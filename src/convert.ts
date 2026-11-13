import type { JSONValue, LanguageModelV3StreamPart, LanguageModelV3Usage } from "@ai-sdk/provider"
import { contentText } from "./acp/content"
import type { PromptResult, PromptUsage, SessionUpdate, ToolCallUpdate } from "./acp/types"
import { canonical } from "./sync/canonical"
import type { ToolIdentity } from "./sync/fingerprint"

type Part = LanguageModelV3StreamPart
type Finish = Extract<Part, { type: "finish" }>

const MAX_STRING = 8_000

/** Turns agy's `session/update` notifications into AI SDK v3 stream parts. */
export class UpdateConverter {
  private seq = 0
  private text: string | undefined
  private reasoning: string | undefined
  private tools = new Map<string, { name: string; resulted: boolean }>()

  /** Parts produced by one update. Unknown or noisy updates produce none. */
  push(update: SessionUpdate): Part[] {
    switch (update.sessionUpdate) {
      case "agent_message_chunk":
        return this.chunk("text", contentText((update as { content?: unknown }).content))
      case "agent_thought_chunk":
        return this.chunk("reasoning", contentText((update as { content?: unknown }).content))
      case "tool_call":
      case "tool_call_update":
        return this.tool(update as ToolCallUpdate)
      default:
        // user_message_chunk (echo), usage_update, session_info_update, available_commands_update, ...
        return []
    }
  }

  /** Closes anything still open. Call before `finish`. */
  close(): Part[] {
    return [...this.endText(), ...this.endReasoning()]
  }

  private chunk(kind: "text" | "reasoning", delta: string | undefined): Part[] {
    if (!delta) return []
    const parts: Part[] = kind === "text" ? this.endReasoning() : this.endText()
    const open = kind === "text" ? this.text : this.reasoning
    if (open === undefined) {
      const id = `${kind}-${this.seq++}`
      if (kind === "text") this.text = id
      else this.reasoning = id
      parts.push({ type: `${kind}-start`, id })
      parts.push({ type: `${kind}-delta`, id, delta })
      return parts
    }
    parts.push({ type: `${kind}-delta`, id: open, delta })
    return parts
  }

  private endText(): Part[] {
    if (this.text === undefined) return []
    const id = this.text
    this.text = undefined
    return [{ type: "text-end", id }]
  }

  private endReasoning(): Part[] {
    if (this.reasoning === undefined) return []
    const id = this.reasoning
    this.reasoning = undefined
    return [{ type: "reasoning-end", id }]
  }

  private tool(update: ToolCallUpdate): Part[] {
    if (!update.toolCallId) return []
    const parts: Part[] = []
    let known = this.tools.get(update.toolCallId)
    if (!known) {
      // Text and thoughts before the call are finished; later text starts a new block.
      parts.push(...this.close())
      const name = toolName(update)
      known = { name, resulted: false }
      this.tools.set(update.toolCallId, known)
      const input = JSON.stringify(update.args ?? update.rawInput ?? {})
      parts.push(
        { type: "tool-input-start", id: update.toolCallId, toolName: name, providerExecuted: true, dynamic: true },
        { type: "tool-input-delta", id: update.toolCallId, delta: input },
        { type: "tool-input-end", id: update.toolCallId },
        { type: "tool-call", toolCallId: update.toolCallId, toolName: name, input, providerExecuted: true, dynamic: true },
      )
    }
    const status = update.status
    if (!known.resulted && (status === "completed" || status === "failed")) {
      known.resulted = true
      parts.push({
        type: "tool-result",
        toolCallId: update.toolCallId,
        toolName: known.name,
        result: toolResult(update),
        isError: status === "failed" ? true : undefined,
        dynamic: true,
      })
    }
    return parts
  }
}

const STOP_REASONS: Record<string, Finish["finishReason"]["unified"]> = {
  end_turn: "stop",
  max_tokens: "length",
  max_turn_requests: "length",
  refusal: "content-filter",
  error: "error",
}

export function finishPart(result: PromptResult, metadata?: Record<string, string>): Finish {
  return {
    type: "finish",
    usage: mapUsage(result.usage),
    finishReason: { unified: STOP_REASONS[result.stopReason] ?? "other", raw: result.stopReason },
    ...(metadata ? { providerMetadata: { agy: metadata } } : {}),
  }
}

/**
 * agy reports `inputTokens` without cached reads, while `totalTokens` includes them.
 * Thought tokens are added to the output total only when agy's own total shows they are not already counted.
 */
export function mapUsage(usage: PromptUsage | undefined): LanguageModelV3Usage {
  const noCache = usage?.inputTokens
  const cacheRead = usage?.cachedReadTokens
  const cacheWrite = usage?.cachedWriteTokens
  const output = usage?.outputTokens
  const thought = usage?.thoughtTokens
  // Fields agy did not report stay undefined; zero would claim a measurement that never happened.
  const known = (noCache ?? 0) + (cacheRead ?? 0) + (cacheWrite ?? 0) + (output ?? 0)
  const separate =
    thought !== undefined && thought > 0 && usage?.totalTokens !== undefined && usage.totalTokens >= known + thought
  return {
    inputTokens: {
      total: noCache === undefined && cacheRead === undefined ? undefined : (noCache ?? 0) + (cacheRead ?? 0),
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: output === undefined ? undefined : separate ? output + thought : output,
      text: output === undefined ? undefined : separate ? output : Math.max(0, output - (thought ?? 0)),
      reasoning: thought,
    },
  }
}

/** Identity of a tool call as it is streamed to OpenCode, for matching OpenCode's echo of it later. */
export function toolIdentity(update: ToolCallUpdate): ToolIdentity {
  return { id: update.toolCallId, name: toolName(update), input: canonical(update.args ?? update.rawInput ?? {}) }
}

export function toolName(update: ToolCallUpdate) {
  const base = update.name ?? update.kind ?? "tool"
  return `agy_${base.replace(/[^a-zA-Z0-9_-]/g, "_")}`
}

/** `/C:/Users/me/x` (agy's URI-style path) becomes `C:\Users\me\x`. */
export function normalizePath(path: string) {
  const match = /^\/([A-Za-z]:)[\\/](.*)$/.exec(path)
  return match ? `${match[1]}\\${match[2]!.replace(/\//g, "\\")}` : path
}

function toolResult(update: ToolCallUpdate) {
  const text: string[] = []
  const diffs: { path?: string; newText?: string; oldText?: string }[] = []
  for (const item of update.content ?? []) {
    if (item.type === "diff") {
      const diff = item as { path?: string; newText?: string; oldText?: string | null }
      diffs.push({
        ...(diff.path ? { path: normalizePath(diff.path) } : {}),
        ...(diff.newText !== undefined ? { newText: diff.newText } : {}),
        ...(diff.oldText ? { oldText: diff.oldText } : {}),
      })
    } else if (item.type === "content") {
      const inner = (item as { content?: { text?: string } }).content
      if (inner?.text) text.push(inner.text)
    }
  }
  const locations = (update.locations ?? []).flatMap((location) => (location.path ? [normalizePath(location.path)] : []))
  return truncate({
    title: update.title?.replace(/(^|\s)\/([A-Za-z]:[\\/])/g, "$1$2"),
    status: update.status,
    ...(text.length ? { text: text.join("\n") } : {}),
    ...(diffs.length ? { diffs } : {}),
    ...(locations.length ? { locations } : {}),
    ...(update.rawOutput !== undefined && !isEmptyObject(update.rawOutput) ? { output: dedupe(update.rawOutput, text) } : {}),
  }) as NonNullable<JSONValue>
}

/** agy repeats a command's output in both `content` and `rawOutput.output`; keep only the structured extras. */
function dedupe(raw: unknown, text: readonly string[]) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return raw
  const { output, ...rest } = raw as Record<string, unknown>
  return typeof output === "string" && text.includes(output) ? rest : raw
}

const isEmptyObject = (value: unknown) =>
  typeof value === "object" && value !== null && !Array.isArray(value) && Object.keys(value).length === 0

function truncate(value: unknown): unknown {
  if (typeof value === "string")
    return value.length > MAX_STRING ? `${value.slice(0, MAX_STRING)}… [truncated ${value.length - MAX_STRING} chars]` : value
  if (Array.isArray(value)) return value.map(truncate)
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .map(([key, item]) => [key, truncate(item)]),
    )
  return value
}
