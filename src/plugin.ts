import type { Plugin, PluginOptions } from "@opencode/plugin"
import { AcpClient, type AcpClientOptions } from "./acp/client"
import { createCatalog, discoverOnce, readCatalog, serializeCatalog } from "./catalog"
import { AgyLanguageModel, KIND_HEADER, PROVIDER_ID as MODEL_PROVIDER, SESSION_HEADER } from "./model"
import { DEFAULT_CATALOG, PROVIDER_ID, PROVIDER_NAME, PROVIDER_PACKAGE, toModel } from "./models"
import { probeCatalog } from "./probe"
import { install, peek, resolve, runtimeKey, type Runtime } from "./runtime"
import { AgySessions } from "./sessions"
import type { SessionRecord } from "./sync/decide"

type AgyPlugin = Plugin.Plugin

export const PLUGIN_ID = "opencode-with-agy"

const CATALOG_KEY = "catalog"

/** Plugin options, as written in `opencode.jsonc` (`{ "package": "...", "options": { ... } }`). */
export interface AgyOptions {
  /** `allow` (default) approves every agy permission request; `deny` rejects them. */
  permission?: "allow" | "deny"
}

export function readOptions(raw: PluginOptions | undefined): AgyOptions {
  return { permission: (raw as AgyOptions | undefined)?.permission === "deny" ? "deny" : "allow" }
}

export interface PluginDependencies {
  /** How the agy process wrapper is built. Tests substitute a scripted fake agent. */
  createClient?: (options: AcpClientOptions) => AcpClient
}

export function createPlugin(dependencies: PluginDependencies = {}): AgyPlugin {
  const createClient = dependencies.createClient ?? ((options) => new AcpClient(options))
  return {
    id: PLUGIN_ID,
    async setup(ctx) {
      const options = readOptions(ctx.options)
      const cwd = ctx.location.directory

      // The child process is spawned lazily by the first request, never at startup.
      const client = createClient({
        permission: options.permission,
        cwd,
      })

      const catalog = createCatalog(
        readCatalog(await ctx.storage.get(CATALOG_KEY)) ?? [...DEFAULT_CATALOG],
        async (next) => {
          await ctx.storage.set(CATALOG_KEY, serializeCatalog(next))
          await ctx.provider.reload()
        },
      )

      const sessions = new AgySessions(
        client,
        {
          get: (key) => ctx.storage.get(key),
          set: (key, value: SessionRecord) => ctx.storage.set(key, value as unknown as Parameters<typeof ctx.storage.set>[1]),
        },
        {
          cwd,
          catalog: catalog.find,
          // Looks up which thinking levels each model offers, once, without sending any prompt.
          onChatSession: discoverOnce(() => probeCatalog(client, cwd), catalog.update),
        },
      )

      let disposed = false
      const runtime: Runtime = {
        run: (request) => sessions.run(request),
        model: catalog.find,
        disposed: () => disposed,
        dispose: async () => {
          if (disposed) return
          disposed = true
          sessions.dispose()
          await client.dispose()
        },
      }
      // From here on this generation owns the project's runtime; the previous one (if any) is retired.
      const key = runtimeKey(PLUGIN_ID, cwd)
      const release = install(key, runtime)

      const registrations = await Promise.all([
        ctx.provider.transform((editor) => {
          editor.add({
            info: {
              id: PROVIDER_ID,
              name: PROVIDER_NAME,
              activation: "auto",
              package: PROVIDER_PACKAGE,
            } as Parameters<typeof editor.add>[0]["info"],
            models: catalog.current().map(toModel),
          })
        }),
        ctx.aisdk.hook(
          "sdk",
          (event) => {
            // OpenCode keeps this SDK (and the models it makes) for as long as the project is open, across plugin
            // reloads, so nothing here may capture this generation's client or sessions: look them up per call.
            event.sdk = {
              languageModel: (modelID: string) =>
                new AgyLanguageModel(modelID, {
                  run: (request) => resolve(key).run(request),
                  model: (id) => peek(key)?.model(id),
                }),
            }
          },
          { providerID: PROVIDER_ID },
        ),
        ctx.session.hook(
          "model.request",
          (event) => {
            // The kind is the only thing OpenCode does not already send. It sets `x-opencode-session-id` itself
            // (session/model-request.ts); repeating it here is a harmless fallback should that ever change.
            event.headers[KIND_HEADER] = event.kind
            event.headers[SESSION_HEADER] = event.sessionID
          },
          { providerID: MODEL_PROVIDER },
        ),
        // Replaying a failed step would repeat agy's side effects (file writes, commands) and permanent auth errors.
        ctx.session.hook(
          "retry",
          (event) => {
            event.decision = { retry: false }
          },
          { providerID: PROVIDER_ID },
        ),
      ]).catch(async (error) => {
        // A failed setup must not leave a live runtime (and its agy process) behind.
        await release()
        throw error
      })

      return async () => {
        await Promise.all(registrations.map((registration) => registration.dispose().catch(() => {})))
        await release()
      }
    },
  }
}

export default createPlugin()
