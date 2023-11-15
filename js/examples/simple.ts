import assert from "node:assert";
import { getEncoding } from "../dist";

const enc = getEncoding("cl100k_base");
assert(enc.decode(enc.encode("hello world")) === "hello world");
