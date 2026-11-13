import { afterEach, describe, expect, test } from "bun:test"
import { createCatalog, discoverOnce, readCatalog, serializeCatalog } from "../src/catalog"
import { resolveLevel } from "../src/model"
import { DEFAULT_CATALOG, THOUGHT_LEVEL_KEY, toModel, type CatalogModel } from "../src/models"
import { probeCatalog, thinkingOf } from "../src/probe"
import {
  FAKE_CATALOG,
  assistant,
  fakeClient,
  fakeEnv,
  harness,
  memoryStore,
  methods,
  primaryHeaders,
  run,
  user,
  variant,
} from "./helpers"

const open: { close(): Promise<void> }[] = []
afterEach(async () => {
  await Promise.all(open.splice(0).map((item) => item.close()))
})
function setup(env = fakeEnv(), store = memoryStore(), options: Parameters<typeof harness>[2] = {}) {
  const h = harness(env, store, options)
  open.push(h)
  return { env, ...h }
}
/** Config calls the plugin sent, as `configId=value`. */
const configCalls = (env: ReturnType<typeof fakeEnv>) =>
  methods(env, "session/set_config_option").map((entry) => `${entry.params.configId}=${entry.params.value}`)
const warningsOf = (parts: Awaited<ReturnType<typeof run>>["parts"]) => {
  const start = parts.find((part) => part.type === "stream-start")
  return start && start.type === "stream-start" ? start.warnings : []
}

describe("catalog: thinking levels are variants of one base model", () => {
  test("each base model is one OpenCode model; levels become variants carrying thoughtLevel", () => {
    const flash = toModel(FAKE_CATALOG[0]!)
    expect(String(flash.id)).toBe("gemini-3.8-flash")
    expect(JSON.parse(JSON.stringify(flash.variants))).toEqual([
      { id: "low", settings: { thoughtLevel: "low" } },
      { id: "medium", settings: { thoughtLevel: "medium" } },
      { id: "high", settings: { thoughtLevel: "high" } },
    ])
    expect(THOUGHT_LEVEL_KEY).toBe("thoughtLevel")
  })

  test("models without a thinking option get no variants", () => {
    expect(toModel(FAKE_CATALOG[1]!).variants.length).toBe(0)
    expect(DEFAULT_CATALOG.filter((model) => !model.thinking).map((model) => toModel(model).variants.length)).toEqual([0, 0, 0])
  })

  test("the static fallback lists base models only, never per-level ids", () => {
    expect(DEFAULT_CATALOG.map((model) => model.id)).toEqual([
      "gemini-3.8-flash",
      "claude-sonnet-4-6",
      "claude-opus-4-6-thinking",
      "gpt-oss-120b-medium",
    ])
    expect(toModel(DEFAULT_CATALOG[0]!).variants.map((v) => String(v.id))).toEqual(["low", "medium", "high"])
  })

  test("levels differ per model", () => {
    expect(toModel(FAKE_CATALOG[2]!).variants.map((v) => String(v.id))).toEqual(["low", "high"])
  })
})

describe("resolveLevel: variant settings arrive as providerOptions.agy", () => {
  const lookup = (id: string) => FAKE_CATALOG.find((model) => model.id === id)

  test("nothing selected -> nothing to apply", () => {
    expect(resolveLevel("gemini-3.8-flash", undefined, lookup)).toEqual({})
    expect(resolveLevel("gemini-3.8-flash", { agy: {} }, lookup)).toEqual({})
  })

  test("a level the model offers passes through; thought_level is accepted as an alias", () => {
    expect(resolveLevel("gemini-3.8-flash", variant("medium"), lookup)).toEqual({ level: "medium" })
    expect(resolveLevel("gemini-3.8-flash", { agy: { thought_level: "low" } }, lookup)).toEqual({ level: "low" })
  })

  test("an unknown level is dropped with a warning that names the valid ones and the default", () => {
    const result = resolveLevel("gemini-3.8-flash", variant("xhigh"), lookup)
    expect(result.level).toBeUndefined()
    expect(result.warning).toMatchObject({ type: "unsupported", feature: "thoughtLevel" })
    expect((result.warning as any).details).toMatch(/"xhigh".*low, medium, high.*default "high"/)
  })

  test("a model without thinking levels, or an unknown model, ignores the level with a warning", () => {
    expect(resolveLevel("claude-sonnet-4-6", variant("low"), lookup).warning).toBeDefined()
    expect(resolveLevel("brand-new-model", variant("low"), lookup).warning).toBeDefined()
  })

  test("options for other providers are not ours", () => {
    expect(resolveLevel("gemini-3.8-flash", { other: { thoughtLevel: "low" } }, lookup)).toEqual({})
  })
})

describe("applying the level to the agy conversation", () => {
  test("a selected variant is applied; the unchanged level is not sent twice", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const low = variant("low")
    const one = await run(model(), [user("script:config")], primaryHeaders(), undefined, low)
    expect(one.text).toBe("cfg:gemini-3.8-flash/low")
    const two = await run(model(), [user("script:config"), assistant(one.text), user("script:config")], primaryHeaders(), undefined, low)
    expect(two.text).toBe("cfg:gemini-3.8-flash/low")
    expect(configCalls(env)).toEqual(["thought_level=low"])
  })

  test("no variant afterwards applies the model's default explicitly, so a previous 'low' does not stick", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const one = await run(model(), [user("script:config")], primaryHeaders(), undefined, variant("low"))
    const two = await run(model(), [user("script:config"), assistant(one.text), user("script:config")])
    expect(two.text).toBe("cfg:gemini-3.8-flash/high")
    expect(configCalls(env)).toEqual(["thought_level=low", "thought_level=high"])
  })

  test("switching between variants applies each change once", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const a = await run(model(), [user("script:config")], primaryHeaders(), undefined, variant("medium"))
    const b = await run(model(), [user("script:config"), assistant(a.text), user("script:config")], primaryHeaders(), undefined, variant("low"))
    await run(
      model(),
      [user("script:config"), assistant(a.text), user("script:config"), assistant(b.text), user("script:config")],
      primaryHeaders(),
      undefined,
      variant("medium"),
    )
    expect(configCalls(env)).toEqual(["thought_level=medium", "thought_level=low", "thought_level=medium"])
  })

  test("another base model brings its own default level, which is applied explicitly", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const one = await run(model("deep-thinker"), [user("script:config")])
    expect(one.text).toBe("cfg:deep-thinker/low")
    expect(configCalls(env)).toEqual(["model=deep-thinker", "thought_level=low"])
  })

  test("switching model forgets the old level: the same level name must still be applied on the new model", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const one = await run(model("gemini-3.8-flash"), [user("script:config")]) // conversation is on "high"
    // deep-thinker resets to its own default ("low"), so asking for "high" has to reach agy even though "high" was current before
    const two = await run(model("deep-thinker"), [user("script:config"), assistant(one.text), user("script:config")], primaryHeaders(), undefined, variant("high"))
    expect(two.text).toBe("cfg:deep-thinker/high")
    expect(configCalls(env)).toEqual(["model=deep-thinker", "thought_level=high"])
  })

  test("a model without thinking levels gets no level calls, even with a variant selected", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const result = await run(model("claude-sonnet-4-6"), [user("script:config")], primaryHeaders(), undefined, variant("low"))
    expect(configCalls(env)).toEqual(["model=claude-sonnet-4-6"])
    expect(warningsOf(result.parts)).toHaveLength(1)
    expect(result.text).toBe("cfg:claude-sonnet-4-6/undefined")
  })

  test("an invalid level is ignored with a warning and the default applies", async () => {
    const env = fakeEnv()
    const { model } = setup(env)
    const result = await run(model(), [user("script:config")], primaryHeaders(), undefined, variant("xhigh"))
    expect(warningsOf(result.parts)).toMatchObject([{ type: "unsupported", feature: "thoughtLevel" }])
    expect(result.text).toBe("cfg:gemini-3.8-flash/high")
    expect(configCalls(env)).toEqual([])
  })

  test("a level agy itself rejects (stale catalog) does not fail the turn, and a later valid level still applies", async () => {
    const env = fakeEnv()
    const stale: CatalogModel = { id: "gemini-3.8-flash", name: "x", thinking: { levels: ["low", "ultra"], default: "high" } }
    const { model } = setup(env, memoryStore(), { catalog: () => stale })
    const bad = await run(model(), [user("script:config")], primaryHeaders(), undefined, variant("ultra"))
    expect(bad.text).toBe("cfg:gemini-3.8-flash/high")
    expect(bad.parts.some((part) => part.type === "error")).toBe(false)
    const good = await run(model(), [user("script:config"), assistant(bad.text), user("script:config")], primaryHeaders(), undefined, variant("low"))
    expect(good.text).toBe("cfg:gemini-3.8-flash/low")
  })

  test("after a restart the model and the selected level are applied again", async () => {
    const env = fakeEnv()
    const store = memoryStore()
    const first = setup(env, store)
    const one = await run(first.model(), [user("script:config")], primaryHeaders(), undefined, variant("low"))
    await first.close()
    const second = setup(env, store)
    const two = await run(second.model(), [user("script:config"), assistant(one.text), user("script:config")], primaryHeaders(), undefined, variant("low"))
    expect(second.outcomes[0]!.mode).toBe("reattached")
    expect(two.text).toBe("cfg:gemini-3.8-flash/low")
    expect(configCalls(env)).toEqual(["thought_level=low", "model=gemini-3.8-flash", "thought_level=low"])
  })

  test("after a restart without a variant the default is applied, not whatever the conversation had", async () => {
    const env = fakeEnv()
    const store = memoryStore()
    const first = setup(env, store)
    const one = await run(first.model(), [user("script:config")], primaryHeaders(), undefined, variant("low"))
    await first.close()
    const second = setup(env, store)
    const two = await run(second.model(), [user("script:config"), assistant(one.text), user("script:config")])
    expect(two.text).toBe("cfg:gemini-3.8-flash/high")
    expect(configCalls(env).slice(1)).toEqual(["model=gemini-3.8-flash", "thought_level=high"])
  })

  test("helper requests (title, compaction) honour the variant too", async () => {
    const env = fakeEnv()
    const { model, outcomes } = setup(env)
    await run(model(), [user("script:config")], { "x-opencode-with-agy-kind": "title", "x-opencode-session-id": "ses_1" }, undefined, variant("low"))
    expect(outcomes[0]!.mode).toBe("throwaway")
    expect(configCalls(env)).toEqual(["thought_level=low"])
  })
})

describe("catalog discovery (no prompts)", () => {
  test("probeCatalog reads every base model's levels and default from the scratch session and closes it", async () => {
    const env = fakeEnv()
    const client = fakeClient(env)
    open.push({ close: () => client.dispose() })
    const models = await probeCatalog(client, "C:\\work")
    expect(models).toEqual([
      { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", thinking: { levels: ["low", "medium", "high"], default: "high" } },
      { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6 (Thinking)" },
      { id: "deep-thinker", name: "Deep Thinker", thinking: { levels: ["low", "high"], default: "low" } },
    ])
    expect(methods(env, "session/prompt")).toHaveLength(0)
    expect(methods(env, "session/new")).toHaveLength(1)
    expect(methods(env, "session/close")).toHaveLength(1)
    expect(configCalls(env)).toEqual(["model=gemini-3.8-flash", "model=claude-sonnet-4-6", "model=deep-thinker"])
  })

  test("thinkingOf reads levels and the current value as the default", () => {
    expect(
      thinkingOf([{ id: "thought_level", currentValue: "medium", options: [{ value: "low" }, { value: "medium" }] }]),
    ).toEqual({ levels: ["low", "medium"], default: "medium" })
    expect(thinkingOf([{ id: "model" }])).toBeUndefined()
    expect(thinkingOf(undefined)).toBeUndefined()
  })

  test("a refreshed catalog changes the variants OpenCode shows, and is persisted in the versioned layout", async () => {
    const saved: unknown[] = []
    const catalog = createCatalog([...DEFAULT_CATALOG], async (next) => void saved.push(serializeCatalog(next)))
    const discovered: CatalogModel[] = [
      { id: "gemini-3.8-flash", name: "Gemini 3.8 Flash", thinking: { levels: ["low", "high"], default: "low" } },
      { id: "claude-sonnet-4-6", name: "Claude" },
    ]
    catalog.update(discovered)
    expect(catalog.current().map((model) => toModel(model).variants.map((v) => String(v.id)))).toEqual([["low", "high"], []])
    await Bun.sleep(10)
    expect(saved).toHaveLength(1)
    expect(readCatalog(saved[0])).toEqual(discovered)
    catalog.update(discovered)
    await Bun.sleep(10)
    expect(saved).toHaveLength(1)
  })

  test("readCatalog rejects anything but the current, well-formed layout", () => {
    expect(readCatalog(undefined)).toBeUndefined()
    expect(readCatalog([{ id: "a", name: "A" }])).toBeUndefined()
    expect(readCatalog({ v: 1, models: [{ id: "a", name: "A" }] })).toBeUndefined()
    expect(readCatalog({ v: 2, models: [{ id: 5 }] })).toBeUndefined()
    expect(readCatalog({ v: 2, models: [{ id: "a", name: "A", thinking: { levels: [], default: "x" } }] })).toEqual([{ id: "a", name: "A" }])
    expect(readCatalog({ v: 2, models: [{ id: "a", name: "A", thinking: { levels: ["low", "high"], default: "nope" } }] })).toEqual([
      { id: "a", name: "A", thinking: { levels: ["low", "high"], default: "high" } },
    ])
  })

  test("discoverOnce runs the probe once, applies a non-empty result, and retries after a failure", async () => {
    const applied: number[] = []
    const logs: string[] = []
    let calls = 0
    const trigger = discoverOnce(
      async () => {
        calls++
        if (calls === 1) throw new Error("agy not ready")
        return [{ id: "a", name: "A" }]
      },
      (models) => applied.push(models.length),
      (message) => logs.push(message),
    )
    trigger()
    trigger() // already running
    await Bun.sleep(10)
    expect(logs).toEqual(["model discovery failed"])
    trigger() // retried after the failure
    trigger()
    await Bun.sleep(10)
    trigger() // done: never again
    await Bun.sleep(10)
    expect(calls).toBe(2)
    expect(applied).toEqual([1])
  })

  test("the chat conversation triggers discovery; helper sessions do not", async () => {
    const env = fakeEnv()
    let triggered = 0
    const { model } = setup(env, memoryStore(), { onChatSession: () => void triggered++ })
    await run(model(), [user("title please")], { "x-opencode-with-agy-kind": "title", "x-opencode-session-id": "ses_1" })
    expect(triggered).toBe(0)
    await run(model(), [user("hello")])
    expect(triggered).toBe(1)
  })
})
