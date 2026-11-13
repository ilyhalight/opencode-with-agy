// A scripted stand-in for `agy acp`: NDJSON JSON-RPC on stdio.
// The behaviour of a prompt is chosen by a `script:<name>` marker in the prompt text (last one wins).
// FAKE_AGY_LOG   - file that receives one JSON line per request/notification/response seen from the client
// FAKE_AGY_STATE - file standing in for agy's on-disk conversations (survives process restarts)
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs"
import { createInterface } from "node:readline"

const LOG = process.env.FAKE_AGY_LOG
const STATE = process.env.FAKE_AGY_STATE
const CONTROL = process.env.FAKE_AGY_CONTROL

type Conversation = { prompts: number; model: string; level?: string }
const disk: Record<string, Conversation> = STATE && existsSync(STATE) ? JSON.parse(readFileSync(STATE, "utf8")) : {}
const live = new Map<string, Conversation>()
const cancels = new Map<string, () => void>()
const waiters = new Map<number, (message: any) => void>()
let nextRequest = 1000

/** Fault injection, re-read on every use so tests can flip it between requests. */
const control = (): Record<string, any> => (CONTROL && existsSync(CONTROL) ? JSON.parse(readFileSync(CONTROL, "utf8")) : {})
const log = (entry: unknown) => LOG && appendFileSync(LOG, JSON.stringify(entry) + "\n")
const save = () => STATE && writeFileSync(STATE, JSON.stringify(disk))
const send = (message: unknown) => process.stdout.write(JSON.stringify(message) + "\n")
const reply = (id: unknown, result: unknown) => send({ jsonrpc: "2.0", id, result })
const fail = (id: unknown, code: number, message: string) => send({ jsonrpc: "2.0", id, error: { code, message } })
const update = (sessionId: string, body: Record<string, unknown>) =>
  send({ jsonrpc: "2.0", method: "session/update", params: { sessionId, update: body } })
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// Like real agy: base models, and a thought_level option that only some of them offer (each with its own default).
const BASE: Record<string, { name: string; levels: string[]; fallback?: string }> = {
  "gemini-3.8-flash": { name: "Gemini 3.8 Flash", levels: ["low", "medium", "high"], fallback: "high" },
  "claude-sonnet-4-6": { name: "Claude Sonnet 4.6 (Thinking)", levels: [] },
  "deep-thinker": { name: "Deep Thinker", levels: ["low", "high"], fallback: "low" },
}
const FRESH = (): Conversation => ({ prompts: 0, model: "gemini-3.8-flash", level: "high" })
const configOptions = (c: Conversation) => [
  { id: "model", name: "Model", category: "model", type: "select", currentValue: c.model, options: Object.entries(BASE).map(([value, m]) => ({ value, name: m.name })) },
  ...(BASE[c.model]!.levels.length
    ? [{ id: "thought_level", name: "Thinking Level", category: "thought_level", type: "select", currentValue: c.level, options: BASE[c.model]!.levels.map((value) => ({ value })) }]
    : []),
  { id: "mode", name: "Mode", category: "mode", type: "select", currentValue: "default", options: [{ value: "default" }] },
]
const setup = (sessionId: string, c: Conversation) => ({
  sessionId,
  models: { availableModels: [{ modelId: "gemini-3.8-flash-high", name: "per-level id, ignored by the plugin" }], currentModelId: "gemini-3.8-flash-high" },
  configOptions: configOptions(c),
  _meta: { antigravity: { tier: { name: "SECRET-PLAN" } } },
})

function request(method: string, params: unknown) {
  const id = nextRequest++
  send({ jsonrpc: "2.0", id, method, params })
  return new Promise<any>((resolve) => waiters.set(id, resolve))
}

const toolCall = (status: string, extra: Record<string, unknown> = {}) => ({
  sessionUpdate: status === "pending" || status === "in_progress" ? "tool_call" : "tool_call_update",
  toolCallId: "call_1",
  name: "write_file",
  args: { Path: "C:\\work\\a.txt", Content: "hi" },
  status,
  kind: "edit",
  title: "File a.txt",
  locations: [{ path: "/C:/work/a.txt" }],
  rawInput: { Path: "C:\\work\\a.txt", Content: "hi" },
  ...extra,
})

async function prompt(id: unknown, params: any) {
  const sessionId: string = params.sessionId
  const conversation = live.get(sessionId)
  if (!conversation) return fail(id, -32602, `unknown session ${sessionId}`)
  const text: string = (params.prompt ?? []).map((block: any) => block.text ?? "").join("\n")
  const script = [...text.matchAll(/script:([a-z-]+)/g)].at(-1)?.[1] ?? "echo"
  // A refusal that arrives before agy has taken the message (no echo, nothing recorded).
  if (script === "reject") return fail(id, -32000, "not authenticated: please log in")
  conversation.prompts++
  disk[sessionId] = conversation
  save()
  update(sessionId, { sessionUpdate: "user_message_chunk", content: { type: "text", text } })
  const usage = { inputTokens: 100, outputTokens: 5, cachedReadTokens: 50, cachedWriteTokens: 0, thoughtTokens: 0, totalTokens: 155 }
  const chunk = (value: string) => update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: value } })

  switch (script) {
    case "config":
      chunk(`cfg:${conversation.model}/${conversation.level}`)
      return reply(id, { stopReason: "end_turn", usage })
    case "mixed":
      update(sessionId, { sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "thinking" } })
      chunk("before ")
      update(sessionId, toolCall("in_progress"))
      update(sessionId, toolCall("pending"))
      update(sessionId, toolCall("in_progress"))
      update(sessionId, toolCall("completed", { rawOutput: {}, content: [{ type: "diff", path: "/C:/work/a.txt", newText: "hi\n" }] }))
      update(sessionId, toolCall("completed", { rawOutput: {} }))
      chunk("after")
      return reply(id, { stopReason: "end_turn", usage: { ...usage, thoughtTokens: 3, totalTokens: 158 } })
    case "fail-tool":
      update(sessionId, toolCall("in_progress"))
      update(sessionId, toolCall("failed", { content: [{ type: "content", content: { type: "text", text: "boom" } }] }))
      chunk("failed")
      return reply(id, { stopReason: "end_turn", usage })
    case "permission": {
      const response = await request("session/request_permission", {
        sessionId,
        toolCall: toolCall("pending"),
        options: [
          { optionId: "allow", name: "Allow", kind: "allow_once" },
          { optionId: "deny", name: "Deny", kind: "reject_once" },
        ],
      })
      chunk(`permission:${JSON.stringify(response.result?.outcome ?? response.error)}`)
      return reply(id, { stopReason: "end_turn", usage })
    }
    case "unknown-request": {
      const response = await request("fs/read_text_file", { sessionId, path: "x" })
      chunk(`unknown:${response.error?.code}`)
      return reply(id, { stopReason: "end_turn", usage })
    }
    case "slow": {
      chunk("partial")
      const cancelled = new Promise<void>((resolve) => cancels.set(sessionId, resolve))
      await Promise.race([cancelled, sleep(10_000)])
      cancels.delete(sessionId)
      // Like agy: cancelled before generation finished reports no usage.
      return reply(id, { stopReason: "cancelled" })
    }
    case "hold":
      log({ event: "hold-start", sessionId, at: Date.now() })
      await sleep(250)
      log({ event: "hold-end", sessionId, at: Date.now() })
      chunk("held")
      return reply(id, { stopReason: "end_turn", usage })
    case "rpc-error":
      return fail(id, -32000, "not authenticated: please log in")
    case "error-after-chunk":
      chunk("started")
      return fail(id, -32000, "quota exceeded halfway")
    case "array-content":
      update(sessionId, { sessionUpdate: "agent_message_chunk", content: [{ type: "text", text: "ab" }, { type: "text", text: "cd" }] })
      return reply(id, { stopReason: "end_turn", usage })
    case "crash-noisy":
      process.stderr.write("E".repeat(1000) + "\nsecond line that must not be shown\n")
      await sleep(20)
      process.exit(3)
    // eslint-disable-next-line no-fallthrough
    case "crash":
      chunk("dying")
      await sleep(20)
      process.exit(1)
    // eslint-disable-next-line no-fallthrough
    case "max-tokens":
      chunk("cut")
      return reply(id, { stopReason: "max_tokens", usage })
    default:
      chunk(`ok-${conversation.prompts}`)
      return reply(id, { stopReason: "end_turn", usage })
  }
}

log({ event: "started", pid: process.pid })

createInterface({ input: process.stdin }).on("line", (line) => {
  if (!line.trim()) return
  const message = JSON.parse(line)
  log({ direction: "in", ...message })
  if (message.method === undefined) {
    waiters.get(message.id)?.(message)
    waiters.delete(message.id)
    return
  }
  const { id, method, params } = message
  switch (method) {
    case "initialize":
      if (control().hangInit) return
      return reply(id, { protocolVersion: 1, agentCapabilities: { loadSession: true }, agentInfo: { name: "fake", version: "0" } })
    case "session/new": {
      const sessionId = crypto.randomUUID()
      const fresh = FRESH()
      live.set(sessionId, fresh)
      return reply(id, setup(sessionId, fresh))
    }
    case "session/resume": {
      if (control().failResume) return fail(id, -32000, "resume exploded")
      // Like agy: an unknown id "succeeds" and yields an empty conversation.
      // The model and thinking level are not durable: a resumed conversation starts from the defaults again.
      const resumed = { ...FRESH(), prompts: (disk[params.sessionId] ?? FRESH()).prompts }
      live.set(params.sessionId, resumed)
      return reply(id, setup(params.sessionId, resumed))
    }
    case "session/list": {
      const c = control()
      if (c.failList) return fail(id, -32000, "list exploded")
      if (c.endlessPages) return reply(id, { sessions: [], nextCursor: String(Number(params.cursor ?? 0) + 1) })
      const all = Object.entries(disk)
        .filter(([, conversation]) => conversation.prompts > 0)
        .map(([sessionId]) => ({ sessionId, cwd: "/C:/work", title: "t" }))
      if (!c.pageSize) return reply(id, { sessions: all })
      const start = Number(params.cursor ?? 0)
      const next = start + c.pageSize
      return reply(id, { sessions: all.slice(start, next), ...(next < all.length ? { nextCursor: String(next) } : {}) })
    }
    case "session/set_config_option": {
      const c = live.get(params.sessionId)
      if (!c) return fail(id, -32602, "unknown session")
      if (params.configId === "model") {
        if (!BASE[params.value]) return fail(id, -32602, `Unknown model: ${params.value}`)
        c.model = params.value
        c.level = BASE[params.value]!.fallback
      } else if (params.configId === "thought_level") {
        if (!BASE[c.model]!.levels.includes(params.value))
          return fail(id, -32602, `Thinking level "${params.value}" is not offered for model "${BASE[c.model]!.name}"`)
        c.level = params.value
      } else return fail(id, -32602, `Unknown config option: ${params.configId}`)
      return reply(id, { configOptions: configOptions(c) })
    }
    case "session/close":
      live.delete(params.sessionId)
      return reply(id, {})
    case "session/cancel":
      cancels.get(params.sessionId)?.()
      return
    case "session/prompt":
      void prompt(id, params)
      return
    default:
      return fail(id, -32601, `Method not found: ${method}`)
  }
})
