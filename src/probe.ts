import type { AcpClient } from "./acp/client"
import type { ConfigOption, SessionSetup } from "./acp/types"
import type { CatalogModel, Thinking } from "./models"

const findOption = (options: ConfigOption[] | undefined, id: string) =>
  options?.find((option) => option.id === id || option.category === id)

/** The thinking levels in a `configOptions` list, if the current model has any. */
export function thinkingOf(options: ConfigOption[] | undefined): Thinking | undefined {
  const option = findOption(options, "thought_level")
  const levels = option?.options?.map((item) => item.value).filter(Boolean) ?? []
  if (!option || levels.length === 0) return undefined
  return { levels, default: option.currentValue && levels.includes(option.currentValue) ? option.currentValue : levels.at(-1)! }
}

/**
 * Asks agy which base models exist and which thinking levels each offers, without sending a prompt: a scratch session
 * is switched through every model (`set_config_option`) and the reported options are read back. The scratch session
 * is closed afterwards. Returns undefined when agy does not describe its models this way.
 */
export async function probeCatalog(
  client: AcpClient,
  cwd: string,
  debug?: (message: string, data?: unknown) => void,
): Promise<CatalogModel[] | undefined> {
  const setup = await client.request<SessionSetup>("session/new", { cwd, mcpServers: [] })
  try {
    const modelOption = findOption(setup.configOptions, "model")
    if (!modelOption?.options?.length) return undefined
    const models: CatalogModel[] = []
    for (const option of modelOption.options) {
      const base: CatalogModel = { id: option.value, name: option.name || option.value }
      try {
        const configured = await client.request<{ configOptions?: ConfigOption[] }>("session/set_config_option", {
          sessionId: setup.sessionId,
          configId: modelOption.id,
          value: option.value,
        })
        const thinking = thinkingOf(configured.configOptions)
        models.push(thinking ? { ...base, thinking } : base)
      } catch (error) {
        // An odd model must not hide the others; list it without levels.
        debug?.("probe: model skipped levels", { model: option.value, error: String(error) })
        models.push(base)
      }
    }
    return models
  } finally {
    await client.request("session/close", { sessionId: setup.sessionId }).catch(() => {})
  }
}
