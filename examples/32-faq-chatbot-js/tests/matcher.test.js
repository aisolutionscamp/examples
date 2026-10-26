import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { FaqMatcher, normalize, stem, tokenize, validateKnowledgeBase } from "../src/matcher.js";

const kb = JSON.parse(readFileSync(new URL("../faq.json", import.meta.url), "utf8"));
const paraphrases = JSON.parse(readFileSync(new URL("./paraphrases.json", import.meta.url), "utf8"));
const matcher = new FaqMatcher(kb);

test("normalize strips accents and punctuation", () => {
  assert.equal(normalize("Café? Öffnungszeiten!"), "cafe  offnungszeiten ");
});

test("stem handles common English suffixes", () => {
  assert.equal(stem("shipping"), "ship");
  assert.equal(stem("prices"), "price");
  assert.equal(stem("boxes"), "box");
  assert.equal(stem("returned"), "return");
  assert.equal(stem("address"), "address");
  assert.equal(stem("glasses"), "glass");
});

test("tokenize removes stopwords and applies synonyms", () => {
  assert.deepEqual(tokenize("What are the delivery costs?", kb.synonyms), ["ship", "price"]);
});

test("the bundled knowledge base is valid", () => {
  assert.deepEqual(validateKnowledgeBase(kb), []);
  assert.deepEqual(validateKnowledgeBase({ entries: [{ id: "a" }, { id: "a", question: "q", answer: "x" }] }), [
    "entry a: question missing",
    "entry a: answer missing",
    "entry 1: duplicate id a",
  ]);
});

test("every FAQ question finds its own answer", () => {
  for (const entry of kb.entries) {
    assert.equal(matcher.ask(entry.question).id, entry.id, entry.question);
  }
});

test("paraphrased questions find the right answer", () => {
  const misses = paraphrases.filter(([q, id]) => matcher.ask(q).id !== id).map(([q, id]) => `${q} -> ${matcher.ask(q).id} (want ${id})`);
  assert.deepEqual(misses, []);
});

test("unrelated questions get the fallback and suggestions stay relevant", () => {
  const result = matcher.ask("Can you repair my bicycle?");
  assert.equal(result.id, null);
  assert.equal(result.answer, matcher.options.fallback);
  assert.equal(matcher.ask("").id, null);
});

test("a partial match below the threshold returns suggestions", () => {
  const strict = new FaqMatcher(kb, { minScore: 0.99 });
  const result = strict.ask("shipping question about something else entirely");
  assert.equal(result.id, null);
  assert.ok(result.suggestions.includes("How much does shipping cost?"));
});

test("rare words weigh more than common ones", () => {
  // "refund" appears in one entry, so it decides the match even next to generic words.
  assert.equal(matcher.ask("order refund").id, "returns");
});
