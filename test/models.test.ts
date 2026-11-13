import { describe, expect, test } from "bun:test"
import { DEFAULT_CATALOG, PROVIDER_PACKAGE, toModel } from "../src/models"
import { readOptions } from "../src/plugin"

describe("model catalog", () => {
  test("the provider package is an aisdk tag", () => {
    expect(PROVIDER_PACKAGE.startsWith("aisdk:")).toBe(true)
  })

  test("models are free, agentic, accept images and return text", () => {
    for (const entry of DEFAULT_CATALOG) {
      const model = toModel(entry)
      expect(String(model.providerID)).toBe("agy")
      expect(model.cost).toEqual([])
      expect(model.capabilities).toEqual({ tools: true, input: ["text", "image"], output: ["text"] })
      expect(model.limit.output).toBe(0)
    }
  })

  test("context windows follow the model family", () => {
    const limit = (id: string) => toModel({ id, name: id }).limit.context
    expect(limit("gemini-3.8-flash")).toBe(1_048_576)
    expect(limit("claude-sonnet-4-6")).toBe(200_000)
    expect(limit("gpt-oss-120b-medium")).toBe(131_072)
  })
})

describe("plugin options", () => {
  test("permission defaults to allow", () => {
    expect(readOptions(undefined)).toEqual({ permission: "allow" })
  })

  test("only permission is read; anything else is ignored", () => {
    expect(readOptions({ permission: "deny" })).toEqual({ permission: "deny" })
    expect(readOptions({ permission: "whatever", command: "C:\\agy.exe", debug: "x.log" })).toEqual({ permission: "allow" })
  })
})
