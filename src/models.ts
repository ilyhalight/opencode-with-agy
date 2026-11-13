import type { Model } from "@opencode/plugin"

export const PROVIDER_ID = "agy"
export const PROVIDER_NAME = "Antigravity (agy)"
/** `aisdk:` marks the package as an in-process AI SDK provider; the name itself is only a tag. */
export const PROVIDER_PACKAGE = "aisdk:opencode-with-agy"

/** The key variant settings use, e.g. `{ "thoughtLevel": "medium" }`. Arrives as `providerOptions.agy.thoughtLevel`. */
export const THOUGHT_LEVEL_KEY = "thoughtLevel"

/** The thinking levels agy offers for one base model. */
export interface Thinking {
  levels: string[]
  /** What agy uses when no level is chosen. */
  default: string
}

/** One base agy model. Thinking levels are variants of it, never separate models. */
export interface CatalogModel {
  id: string
  name: string
  /** Absent for models without a thinking option (Claude, GPT-OSS). */
  thinking?: Thinking
}

/** What agy offered when this plugin was written. Replaced by what `session/new` and `set_config_option` report. */
export const DEFAULT_CATALOG: readonly CatalogModel[] = [
  { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", thinking: { levels: ["low", "medium", "high"], default: "high" } },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6 (Thinking)" },
  { id: "claude-opus-4-6-thinking", name: "Claude Opus 4.6 (Thinking)" },
  { id: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium)" },
]

/** agy reports no context size, so these are per-family guesses; only Gemini's is confirmed by the 1048576 that agy sends. */
function contextWindow(id: string) {
  if (id.startsWith("gemini")) return 1_048_576
  if (id.startsWith("claude")) return 200_000
  if (id.startsWith("gpt-oss")) return 131_072
  return 200_000
}

/** Subscription usage is not billed per token, so every model is free as far as OpenCode is concerned. */
export function toModel(model: CatalogModel): Model.Info {
  return {
    id: model.id as Model.ID,
    modelID: model.id as Model.ID,
    providerID: PROVIDER_ID as Model.Info["providerID"],
    name: model.name,
    // agy is agentic and reads images; it only returns text.
    capabilities: { tools: true, input: ["text", "image"], output: ["text"] },
    variants: (model.thinking?.levels ?? []).map((level) => ({
      id: level as Model.VariantID,
      settings: { [THOUGHT_LEVEL_KEY]: level },
    })),
    time: { released: 0 },
    cost: [],
    status: "active",
    enabled: true,
    // Output limit unknown: 0 means "not specified" in OpenCode (same as the Claude Agent provider).
    limit: { context: contextWindow(model.id), output: 0 },
  }
}
