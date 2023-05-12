// generate combinations

const strings = "'s|'t|'re|'ve|'m|'ll|'d";

const testRegex =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

function recombine(value: string, acc: string[] = [""]): string[] {
  if (value.length === 0) return acc;
  if (value[0].match(/[a-zA-Z]/)) {
    return recombine(
      value.substring(1),
      acc.flatMap((i) => [
        `${i}${value[0].toLocaleLowerCase()}`,
        `${i}${value[0].toLocaleUpperCase()}`,
      ])
    );
  }

  return recombine(
    value.substring(1),
    acc.map((i) => `${i}${value[0]}`)
  );
}

let match = testRegex.replace(/\(\?i:(.*?)\)/, (_, match: string) => {
  const insensitive = match
    .split("|")
    .flatMap((a) => recombine(a))
    .join("|");
  return `(${insensitive})`;
});

console.log(match);
