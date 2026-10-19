// Checks the practice word lists. Run from web/:  node scripts/check-words.mjs
import { CATEGORIES, LETTERS } from "../src/lib/words.js";

let problems = 0;
for (const c of CATEGORIES) {
  const issues = [];
  const isAlphabet = c.id === "alphabet";
  for (const w of c.items) {
    if ([...w].some((ch) => !LETTERS.includes(ch))) issues.push(`${w}: J/Z or not a letter`);
    if (!isAlphabet && w.includes("T")) issues.push(`${w}: contains T`);
    if (w.includes("P") && w.includes("Q")) issues.push(`${w}: mixes P and Q`);
  }
  const dupes = c.items.filter((w, i) => c.items.indexOf(w) !== i);
  if (dupes.length) issues.push(`duplicates: ${dupes.join(" ")}`);
  if (c.id === "doubles") {
    const noDouble = c.items.filter((w) => !/(.)\1/.test(w));
    if (noDouble.length) issues.push(`no double letter: ${noDouble.join(" ")}`);
  }
  problems += issues.length;
  console.log(`${issues.length ? "FAIL" : "PASS"}  ${c.name.padEnd(15)} ${String(c.items.length).padStart(2)} items` +
              (issues.length ? `  ${issues.join("; ")}` : ""));
}
console.log(problems ? `${problems} problems.` : "All word lists valid.");
process.exit(problems ? 1 : 0);
