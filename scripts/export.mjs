// Builds the catalog records for aisolutionscamp.io from this repository.
//   node scripts/export.mjs <out-dir>
// Writes <out-dir>/solutions/<key>.json and <out-dir>/projects/<id>.json.
// Field names match the `solutions` / `projects` tables of the site.
import fs from "node:fs";
import path from "node:path";

const REPO_URL = "https://github.com/aisolutionscamp/examples";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const outDir = path.resolve(process.argv[2] || "export");
fs.mkdirSync(path.join(outDir, "solutions"), { recursive: true });
fs.mkdirSync(path.join(outDir, "projects"), { recursive: true });

const SOLUTION_FIELDS = ["title", "description", "aiTools", "prompts", "language", "implementationSteps",
  "performance", "userExperience", "accuracyRate", "graphData"];

for (const name of fs.readdirSync(path.join(root, "examples")).sort()) {
  const dir = path.join(root, "examples", name);
  // Skip tool caches (.ruff_cache, ...) and anything that is not an example.
  if (name.startsWith(".") || !fs.existsSync(path.join(dir, "solution.json"))) continue;
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "solution.json"), "utf8"));
  if (!Number.isInteger(meta.projectId)) throw new Error(`${name}: projectId missing`);
  if (meta.solutionId !== null && !Number.isInteger(meta.solutionId)) throw new Error(`${name}: solutionId must be a number or null`);
  for (const f of ["title", "description", "language", "codeSnippetFile"]) {
    if (!meta[f]) throw new Error(`${name}: ${f} missing`);
  }
  const record = {
    key: name,
    solutionId: meta.solutionId,
    projectId: meta.projectId,
    ...Object.fromEntries(SOLUTION_FIELDS.map((f) => [f, meta[f] ?? null])),
    codeSnippet: fs.readFileSync(path.join(dir, meta.codeSnippetFile), "utf8").trimEnd() + "\n",
    externalLinks: [{ title: "Full example on GitHub", url: `${REPO_URL}/tree/main/examples/${name}` }, ...(meta.externalLinks || [])],
  };
  fs.writeFileSync(path.join(outDir, "solutions", `${name}.json`), JSON.stringify(record, null, 2) + "\n");
  console.log(`exported solution ${name}`);
}

for (const file of fs.readdirSync(path.join(root, "use-cases")).filter((f) => f.endsWith(".json")).sort()) {
  const meta = JSON.parse(fs.readFileSync(path.join(root, "use-cases", file), "utf8"));
  if (!Number.isInteger(meta.id)) throw new Error(`${file}: id missing`);
  fs.writeFileSync(path.join(outDir, "projects", `${meta.id}.json`), JSON.stringify(meta, null, 2) + "\n");
  console.log(`exported use case ${meta.id}`);
}
