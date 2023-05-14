export function never(message: string, _: never) {
  throw new Error(message);
}
