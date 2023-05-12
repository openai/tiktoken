import { test, expect, describe, afterAll } from "vitest";
import { get_encoding } from "../../wasm/dist";
import { Tiktoken } from "../src/index";

describe("LiteTokenizer matches the behavior of @dqbd/tiktoken", () => {
  const lite = new Tiktoken();
  const full = get_encoding("cl100k_base");

  afterAll(() => full.free());

  test("Simple test", () => {
    const text = "hello world";
    expect([...lite.encode(text)]).toEqual([...full.encode(text)]);
  });

  test("Magic tokens", () => {
    const text = "<|fim_prefix|>test<|fim_suffix|>";
    expect([...lite.encode(text, "all")]).toEqual([
      ...full.encode(text, "all"),
    ]);
  });

  test("Emojis and non-latin characters", () => {
    const fixtures = [
      "Hello world",
      "New lines\n\n\n\n\n       Spaces",
      "👩‍👦‍👦 👩‍👧‍👦 👩‍👧‍👧 👩‍👩‍👦 👩‍👩‍👧 🇨🇿 Emojis: 🧑🏾‍💻️🧑🏿‍🎓️🧑🏿‍🏭️🧑🏿‍💻️",
      "是美國一個人工智能研究實驗室 由非營利組織OpenAI Inc",
      "<|im_start|>test<|im_end|>",
    ];

    for (const text of fixtures) {
      expect([...lite.encode(text)]).toEqual([...full.encode(text)]);
    }
  });
});
