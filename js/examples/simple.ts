import assert from "node:assert";
import { getEncoding } from "../dist";

const enc = getEncoding("gpt2");
assert(enc.decode(enc.encode("hello world")) === "hello world");
