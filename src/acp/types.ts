// Minimal subset of the Agent Client Protocol (https://agentclientprotocol.com) as spoken by `agy acp` v1.5.x.
// Only fields this plugin reads are typed; everything else stays `unknown` and is ignored.

export type JsonObject = Record<string, unknown>

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }

export type StopReason = "end_turn" | "max_tokens" | "max_turn_requests" | "refusal" | "cancelled" | (string & {})

export interface PromptUsage {
  inputTokens?: number
  outputTokens?: number
  cachedReadTokens?: number
  cachedWriteTokens?: number
  thoughtTokens?: number
  totalTokens?: number
}

export interface PromptResult {
  stopReason: StopReason
  usage?: PromptUsage
}

export interface ModelInfo {
  modelId: string
  name: string
}

/** One entry of `configOptions` (`model`, `thought_level`, `mode`, ...). Only flat select lists are understood. */
export interface ConfigOption {
  id: string
  name?: string
  category?: string
  currentValue?: string
  options?: { value: string; name?: string }[]
}

export interface SessionSetup {
  sessionId: string
  models?: { availableModels?: ModelInfo[]; currentModelId?: string }
  configOptions?: ConfigOption[]
}

export type ToolStatus = "pending" | "in_progress" | "completed" | "failed" | (string & {})

export type ToolContent =
  | { type: "diff"; path?: string; newText?: string; oldText?: string | null }
  | { type: "content"; content?: { type?: string; text?: string } }
  | { type: string }

/** `tool_call` and `tool_call_update` share a shape; agy repeats the whole object on every update. */
export interface ToolCallUpdate {
  sessionUpdate: "tool_call" | "tool_call_update"
  toolCallId: string
  /** agy extension: its internal tool name, e.g. `write_file`. */
  name?: string
  /** agy extension: capitalised argument keys, e.g. `Path`, `CommandLine`. */
  args?: JsonObject
  status?: ToolStatus
  kind?: string
  title?: string
  locations?: { path?: string }[]
  rawInput?: unknown
  rawOutput?: unknown
  content?: ToolContent[]
}

export interface TextChunkUpdate {
  sessionUpdate: "agent_message_chunk" | "agent_thought_chunk" | "user_message_chunk"
  content?: { type?: string; text?: string }
}

/** Any other `session/update` (usage, commands, config, title, plan, mode...). Ignored by this plugin. */
export interface OtherUpdate {
  sessionUpdate: string
}

export type SessionUpdate = ToolCallUpdate | TextChunkUpdate | OtherUpdate

export interface PermissionOption {
  optionId: string
  name?: string
  kind: "allow_once" | "allow_always" | "reject_once" | "reject_always" | (string & {})
}

export interface PermissionRequest {
  sessionId: string
  toolCall?: ToolCallUpdate
  options?: PermissionOption[]
}

export class AcpRpcError extends Error {
  constructor(
    readonly code: number,
    message: string,
    readonly data?: unknown,
  ) {
    super(message)
    this.name = "AcpRpcError"
  }
}
