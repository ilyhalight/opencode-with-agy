import type { LanguageModelV3Prompt } from "@ai-sdk/provider"
import type { AcpClient } from "./acp/client"
import { contentText } from "./acp/content"
import {
  AcpRpcError,
  type ContentBlock,
  type PromptResult,
  type SessionSetup,
  type SessionUpdate,
  type ToolCallUpdate,
} from "./acp/types"
import { toolIdentity } from "./convert"
import type { CatalogModel } from "./models"
import { thinkingOf } from "./probe"
import { decide, type Decision, type SessionRecord } from "./sync/decide"
import { ANY_REPLY, assistantHash, groups as fingerprint, type Group, type ToolIdentity } from "./sync/fingerprint"
import { catchUpContent, flattenContent, userContent, type Rendered } from "./sync/render"

export interface SessionStore {
  get(key: string): Promise<unknown>
  set(key: string, value: SessionRecord): Promise<void>
}

export interface RunRequest {
  /** OpenCode session id. Without one (or when `primary` is false) a throwaway agy session is used. */
  sessionID?: string
  /** True for the main chat loop; titles, compaction and other helper requests are not primary. */
  primary: boolean
  /** Base agy model id, e.g. `gemini-3.8-flash`. */
  modelId: string
  /** Thinking level from the selected variant, already checked against the catalog. Unset: the model's default. */
  thoughtLevel?: string
  prompt: LanguageModelV3Prompt
  signal: AbortSignal
  onUpdate: (update: SessionUpdate) => void
}

export type RunMode = "continue" | "reattached" | "fresh" | "throwaway"

export interface RunOutcome {
  result: PromptResult
  mode: RunMode
  /** Why a new agy conversation was needed, when one was. */
  reason?: string
  acpSessionId: string
  /** What was sent to agy (useful for tests and debugging). */
  sent: ContentBlock[]
  warnings: string[]
}

export interface SessionsOptions {
  cwd: string
  /** Close an agy conversation after this long without use. It is resumed from disk on demand. 0 disables. */
  idleMs?: number
  debug?: (message: string, data?: unknown) => void
  /** Called whenever a conversation is created for a chat session (not for helper requests). */
  onChatSession?: () => void
  /** Catalog entry of a base model; supplies the default thinking level to apply when none is selected. */
  catalog?: (modelId: string) => CatalogModel | undefined
}

/**
 * The earlier agy conversation could not be checked (resume or listing failed, or the listing was too long to
 * search). Nothing was changed, so the same request can simply be sent again.
 */
export class AgyRetryableError extends Error {
  override name = "AgyRetryableError"
}

interface Live {
  acpSessionId: string
  generation: number
  /** Model last applied on this ACP session, or undefined when unknown (after resume). */
  model?: string
  /** Thinking level last applied (or known to be current), or undefined when unknown. */
  level?: string
  timer?: ReturnType<typeof setTimeout>
}

const STORAGE_PREFIX = "session:"
const MAX_LIST_PAGES = 50

export class AgySessions {
  private live = new Map<string, Live>()
  private locks = new Map<string, Promise<unknown>>()

  constructor(
    private readonly client: AcpClient,
    private readonly store: SessionStore,
    private readonly options: SessionsOptions,
  ) {}

  run(request: RunRequest): Promise<RunOutcome> {
    if (!request.primary || !request.sessionID) return this.throwaway(request)
    return this.serialized(request.sessionID, () => this.primary(request.sessionID!, request))
  }

  /** Cancels the idle timers. The child process itself is owned by the client. */
  dispose() {
    for (const live of this.live.values()) clearTimeout(live.timer)
    this.live.clear()
  }

  private async primary(sessionID: string, request: RunRequest): Promise<RunOutcome> {
    const groups = fingerprint(request.prompt)
    const record = await this.load(sessionID)
    let decision = decide(groups, record)
    // The recorded conversation, when verified not to exist in agy any more (nothing to retire then).
    let recordGone = false
    let reattached = false
    this.stopIdle(sessionID)

    if (decision.kind === "continue" && record) {
      const live = this.live.get(sessionID)
      // After a crash the generation only changes on the next spawn, so a dead process must not look usable.
      const usable =
        this.client.running && live?.acpSessionId === record.acpSessionId && live.generation === this.client.generation
      if (!usable) {
        // The old conversation keeps working in its own directory, so a changed cwd just starts a new one.
        if (record.cwd !== this.options.cwd) decision = { kind: "fresh", reason: "diverged" }
        else if ((await this.reattach(sessionID, record)) === "ok") reattached = true
        else {
          decision = { kind: "fresh", reason: "diverged" }
          recordGone = true
        }
      }
    }

    const continuing = decision.kind === "continue"
    const rendered = decision.kind === "continue" ? continued(request.prompt, groups, decision) : replayed(request.prompt, groups)
    const previous = this.live.get(sessionID)
    const live = continuing ? previous! : await this.create(true)
    if (!continuing) this.live.set(sessionID, live)
    const mode: RunMode = continuing ? (reattached ? "reattached" : "continue") : "fresh"
    const reason = decision.kind === "fresh" ? describeReason(record ? decision.reason : "no-record") : undefined

    let text = ""
    let sent = false
    let engaged = false
    const tools = new Map<string, ToolIdentity>()
    const onUpdate = (update: SessionUpdate) => {
      engaged = true
      if (update.sessionUpdate === "agent_message_chunk") text += contentText((update as { content?: unknown }).content)
      if (update.sessionUpdate === "tool_call" || update.sessionUpdate === "tool_call_update") {
        const call = update as ToolCallUpdate
        if (call.toolCallId && !tools.has(call.toolCallId)) tools.set(call.toolCallId, toolIdentity(call))
      }
      request.onUpdate(update)
    }

    // A new conversation replaces the recorded one only once the turn reached agy; until then the old one stays.
    const abandon = async () => {
      if (!continuing) {
        if (previous) this.live.set(sessionID, previous)
        else this.live.delete(sessionID)
        await this.close(live.acpSessionId)
      }
      this.schedule(sessionID)
    }
    const commit = async (reply: string) => {
      await this.save(sessionID, {
        v: 1,
        acpSessionId: live.acpSessionId,
        model: request.modelId,
        cwd: this.options.cwd,
        history: [...groups.map((group) => group.hash), reply],
      })
      if (!continuing) {
        // Close what this turn replaced, but never the conversation that just answered.
        const retired = new Set([recordGone ? undefined : record?.acpSessionId, previous?.acpSessionId])
        retired.delete(live.acpSessionId)
        for (const id of retired) if (id) await this.close(id)
      }
      this.schedule(sessionID)
    }

    let result: PromptResult
    try {
      await this.applyConfig(live, request)
      result = await this.client.prompt(live.acpSessionId, rendered.blocks, onUpdate, request.signal, () => {
        sent = true
      })
    } catch (error) {
      // An RPC error before agy produced anything is a refusal: it did not take the message. Anything else after the
      // request was written (crash, exit, error mid-stream) may have, and replaying it would show agy the message twice.
      const delivered = sent && (engaged || !(error instanceof AcpRpcError))
      if (!delivered) {
        await abandon()
        throw error
      }
      await commit(ANY_REPLY)
      throw error
    }

    const finished = result.stopReason === "end_turn" && (text.trim() !== "" || tools.size > 0)
    await commit(finished ? assistantHash(text, [...tools.values()]) : ANY_REPLY)
    return { result, mode, reason, acpSessionId: live.acpSessionId, sent: rendered.blocks, warnings: rendered.warnings }
  }

  private async throwaway(request: RunRequest): Promise<RunOutcome> {
    const live = await this.create(false)
    const rendered = flattenContent(request.prompt)
    try {
      await this.applyConfig(live, request)
      const result = await this.client.prompt(live.acpSessionId, rendered.blocks, request.onUpdate, request.signal)
      return { result, mode: "throwaway", acpSessionId: live.acpSessionId, sent: rendered.blocks, warnings: rendered.warnings }
    } finally {
      await this.close(live.acpSessionId)
    }
  }

  private async create(primary: boolean): Promise<Live> {
    const setup = await this.client.request<SessionSetup>("session/new", { cwd: this.options.cwd, mcpServers: [] })
    // Never let catalog bookkeeping break a request, and skip helper sessions.
    if (primary) {
      try {
        this.options.onChatSession?.()
      } catch (error) {
        this.options.debug?.("onChatSession failed", String(error))
      }
    }
    return {
      acpSessionId: setup.sessionId,
      generation: this.client.generation,
      // What a new conversation starts with, so an unchanged choice costs no round trip.
      model: setup.configOptions?.find((option) => option.id === "model")?.currentValue,
      level: thinkingOf(setup.configOptions)?.default,
    }
  }

  /**
   * After a restart (or idle close) the conversation still lives on disk in agy; resume it and check it is real.
   * Resolves "missing" only when the full listing was read and the id is not in it. Anything that stops us from
   * knowing throws a retryable error and changes nothing.
   */
  private async reattach(sessionID: string, record: SessionRecord): Promise<"ok" | "missing"> {
    try {
      await this.client.request("session/resume", { sessionId: record.acpSessionId, cwd: this.options.cwd, mcpServers: [] })
    } catch (error) {
      throw this.unverified("resume failed", error)
    }
    // Resuming an unknown id "succeeds" and yields an empty conversation, so confirm it is a real one.
    let found: boolean | undefined
    try {
      found = await this.exists(record.acpSessionId)
    } catch (error) {
      throw this.unverified("listing conversations failed", error)
    }
    if (found === undefined) throw this.unverified("conversation list too long to search", undefined)
    if (!found) return "missing"
    this.live.set(sessionID, { acpSessionId: record.acpSessionId, generation: this.client.generation })
    return "ok"
  }

  private unverified(what: string, cause: unknown) {
    this.options.debug?.("reattach unverified", { what, cause: cause === undefined ? undefined : String(cause) })
    const detail = cause instanceof Error ? `: ${cause.message.slice(0, 200)}` : ""
    return new AgyRetryableError(`agy could not confirm the earlier conversation (${what}${detail}). Nothing was changed; try again.`)
  }

  /** `undefined` means the listing was cut off before the id could be found or ruled out. */
  private async exists(acpSessionId: string): Promise<boolean | undefined> {
    // agy ignores `cwd` here (it lists every conversation), so one unfiltered, paginated listing is enough.
    let cursor: string | undefined
    for (let page = 0; page < MAX_LIST_PAGES; page++) {
      const listed = await this.client.request<{ sessions?: { sessionId: string }[]; nextCursor?: string }>(
        "session/list",
        cursor ? { cursor } : {},
      )
      if (listed.sessions?.some((session) => session.sessionId === acpSessionId)) return true
      cursor = listed.nextCursor
      if (!cursor) return false
    }
    this.options.debug?.("session list page cap reached", { pages: MAX_LIST_PAGES })
    return undefined
  }

  /**
   * agy forgets the chosen model and thinking level after resume or load, so both are applied whenever they may
   * differ from what the live conversation has. With no level selected the model's own default is applied
   * explicitly: a conversation that ran on "low" must not stay on "low".
   */
  private async applyConfig(live: Live, request: Pick<RunRequest, "modelId" | "thoughtLevel">) {
    if (live.model !== request.modelId) {
      await this.client.request("session/set_config_option", { sessionId: live.acpSessionId, configId: "model", value: request.modelId })
      live.model = request.modelId
      // The new model brings its own level.
      live.level = undefined
    }
    const thinking = this.options.catalog?.(request.modelId)?.thinking
    if (!thinking) return
    const wanted = request.thoughtLevel ?? thinking.default
    if (live.level === wanted) return
    try {
      await this.client.request("session/set_config_option", { sessionId: live.acpSessionId, configId: "thought_level", value: wanted })
      live.level = wanted
    } catch (error) {
      // agy refused this level (the catalog may be stale). The turn is still worth running on whatever level is active.
      if (!(error instanceof AcpRpcError)) throw error
      this.options.debug?.("thought_level rejected", { level: wanted, error: String(error) })
      live.level = undefined
    }
  }

  private async close(acpSessionId: string) {
    try {
      if (this.client.running) await this.client.request("session/close", { sessionId: acpSessionId })
    } catch (error) {
      this.options.debug?.("close failed", String(error))
    }
  }

  private schedule(sessionID: string) {
    const idle = this.options.idleMs ?? 15 * 60_000
    const live = this.live.get(sessionID)
    if (!live || idle <= 0) return
    clearTimeout(live.timer)
    live.timer = setTimeout(() => {
      this.live.delete(sessionID)
      void this.serialized(sessionID, () => this.close(live.acpSessionId))
    }, idle)
    live.timer.unref?.()
  }

  private stopIdle(sessionID: string) {
    const live = this.live.get(sessionID)
    if (live) clearTimeout(live.timer)
  }

  private async load(sessionID: string): Promise<SessionRecord | undefined> {
    const value = await this.store.get(STORAGE_PREFIX + sessionID)
    return isRecord(value) ? value : undefined
  }

  private save(sessionID: string, record: SessionRecord) {
    return this.store.set(STORAGE_PREFIX + sessionID, record)
  }

  private serialized<T>(key: string, task: () => Promise<T>): Promise<T> {
    const previous = this.locks.get(key) ?? Promise.resolve()
    const next = previous.then(task, task)
    const tail = next.catch(() => {})
    this.locks.set(key, tail)
    void tail.then(() => {
      if (this.locks.get(key) === tail) this.locks.delete(key)
    })
    return next
  }
}

function continued(prompt: LanguageModelV3Prompt, groups: readonly Group[], decision: Extract<Decision, { kind: "continue" }>): Rendered {
  return userContent(prompt.slice(groups[decision.from]!.start))
}

/** Replay: everything before the trailing run of user messages becomes history. */
function replayed(prompt: LanguageModelV3Prompt, groups: readonly Group[]): Rendered {
  let latest = groups.length
  while (latest > 0 && groups[latest - 1]!.role === "user") latest--
  const start = latest === groups.length ? prompt.length : groups[latest]!.start
  return catchUpContent(prompt, start)
}

function describeReason(reason: string) {
  switch (reason) {
    case "no-record":
      return "no earlier agy conversation for this session"
    case "diverged":
      return "history changed (revert, edit, compaction or another model replied) or the earlier conversation is gone"
    case "no-new-input":
      return "prompt has no new user input"
    default:
      return "unexpected message sequence"
  }
}

function isRecord(value: unknown): value is SessionRecord {
  if (!value || typeof value !== "object") return false
  const record = value as Partial<SessionRecord>
  return (
    record.v === 1 &&
    typeof record.acpSessionId === "string" &&
    typeof record.cwd === "string" &&
    typeof record.model === "string" &&
    Array.isArray(record.history) &&
    record.history.every((hash) => typeof hash === "string")
  )
}
