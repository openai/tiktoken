import { it, expect, describe } from "vitest";
import { encoding_for_model, get_encoding, get_encoding_name_for_model } from "../dist";

it("encoding_for_model initialization", () => {
  expect(() => encoding_for_model("gpt2")).not.toThrowError();
  // @ts-expect-error
  expect(() => encoding_for_model("gpt2-unknown")).toThrowError(
    "Invalid model"
  );
});

it("get_encoding initialization", () => {
  expect(() => get_encoding("cl100k_base")).not.toThrowError();
  // @ts-expect-error
  expect(() => get_encoding("unknown")).toThrowError("Invalid encoding");
});

describe("gpt2", () => {
  const enc = get_encoding("gpt2");

  it("encodes hello world string", () => {
    expect(enc.encode("hello world")).toStrictEqual(
      new Uint32Array([31373, 995])
    );
  });

  it("decodes hello world string", () => {
    expect(
      new TextDecoder().decode(enc.decode(new Uint32Array([31373, 995])))
    ).toStrictEqual("hello world");
  });

  it("encodes hello world string, all allowed special characters", () => {
    expect(enc.encode("hello <|endoftext|>", "all")).toStrictEqual(
      new Uint32Array([31373, 220, 50256])
    );
  });
});

describe("cl100k_base", () => {
  const enc = get_encoding("cl100k_base");

  it("encodes hello world string", () => {
    expect(enc.encode("hello world")).toStrictEqual(
      new Uint32Array([15339, 1917])
    );
  });

  it("decodes hello world string", () => {
    expect(
      new TextDecoder().decode(enc.decode(new Uint32Array([15339, 1917])))
    ).toStrictEqual("hello world");
  });

  it("encodes hello world string, all allowed special characters", () => {
    expect(enc.encode("hello <|endoftext|>", "all")).toStrictEqual(
      new Uint32Array([15339, 220, 100257])
    );
  });
});

describe("o200k_base", () => {
  const enc = get_encoding("o200k_base");

  it("encodes hello world string", () => {
    expect(enc.encode("hello world")).toStrictEqual(
      new Uint32Array([24912, 2375])
    );
  });

  it("decodes hello world string", () => {
    expect(
      new TextDecoder().decode(enc.decode(new Uint32Array([24912, 2375])))
    ).toStrictEqual("hello world");
  });

  it("encodes hello world string, all allowed special characters", () => {
    expect(enc.encode("hello <|endoftext|>", "all")).toStrictEqual(
      new Uint32Array([24912, 220, 199999])
    );
  });
});

it("test_simple", () => {
  const encodings = [
    "gpt2",
    "r50k_base",
    "p50k_base",
    "p50k_edit",
    "cl100k_base",
  ] as const;

  for (const encoding of encodings) {
    const enc = get_encoding(encoding);
    for (let token = 0; token < 10_000; token++) {
      expect(
        enc.encode_single_token(enc.decode_single_token_bytes(token))
      ).toStrictEqual(token);
    }
  }
});

it("test_encoding_for_model", () => {
  expect(encoding_for_model("gpt2").name).toEqual("gpt2");
  expect(encoding_for_model("text-davinci-003").name).toEqual("p50k_base");
  expect(encoding_for_model("gpt-3.5-turbo").name).toEqual("cl100k_base");
});

it("test_get_encoding_name_for_model", () => {
  expect(get_encoding_name_for_model("gpt2")).toEqual("gpt2");
  expect(get_encoding_name_for_model("text-davinci-003")).toEqual("p50k_base");
  expect(get_encoding_name_for_model("gpt-3.5-turbo")).toEqual("cl100k_base");

  // @ts-expect-error - explicitly testing for invalid model
  expect(() => get_encoding_name_for_model("gpt2-unknown")).toThrowError(
      "Invalid model: gpt2-unknown"
  );
})

it("test_custom_tokens", () => {
  const enc = encoding_for_model("gpt2", {
    "<|im_start|>": 100264,
    "<|im_end|>": 100265,
  });
  expect(enc.encode("<|im_start|>test<|im_end|>", "all")).toStrictEqual(
    new Uint32Array([100264, 9288, 100265])
  );
});

it("encode string tokens", () => {
  const enc = get_encoding("gpt2", { "<|im_start|>": 100264 });

  expect(enc.encode("hello world")).toStrictEqual(
    new Uint32Array([31373, 995])
  );

  expect(enc.encode("<|endoftext|>", ["<|endoftext|>"])).toStrictEqual(
    new Uint32Array([50256])
  );

  expect(enc.encode("<|endoftext|>", "all")).toStrictEqual(
    new Uint32Array([50256])
  );

  expect(() => enc.encode("<|endoftext|>")).toThrowError(
    "The text contains a special token that is not allowed"
  );

  expect(() => enc.encode("<|im_start|>")).toThrowError(
    "The text contains a special token that is not allowed"
  );

  expect(enc.encode("<|endoftext|>", [], [])).toStrictEqual(
    new Uint32Array([27, 91, 437, 1659, 5239, 91, 29])
  );
});

it("invalid (dis)allowed_tokens", () => {
  const enc = get_encoding("gpt2");

  // @ts-expect-error
  expect(() => enc.encode("hello world", "invalid-string")).toThrowError(
    "Invalid value for allowed_special"
  );

  // @ts-expect-error
  expect(() => enc.encode("hello world", [], "invalid-string")).toThrowError(
    "Invalid value for disallowed_special"
  );
});

it("invalid");
