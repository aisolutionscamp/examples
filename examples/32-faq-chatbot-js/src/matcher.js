// FAQ matcher: finds the best answer for a question in a JSON knowledge base.
// No AI model and no server: keyword matching with IDF weights, synonyms and light stemming.
// Works in the browser and in Node.js (ES module, no dependencies).

const STOPWORDS = new Set(
  ("a about also an and any are be can could do does for from get has have how i in is it just know like me " +
    "my need of on or our please some something take tell the there this to want we what when which who why " +
    "will with would yet you your").split(" "),
);

export const DEFAULT_OPTIONS = {
  minScore: 0.35, // below this, the bot says it does not know
  suggestionCount: 3,
  unknownWeight: 0.5, // weight of question words that appear in no entry (known words: about 1.1 to 2.5)
  fallback: "Sorry, I don't know the answer to that yet. Please contact us and we will reply as soon as possible.",
};

export function normalize(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "") // strip accents
    .replace(/[^a-z0-9\s]/g, " ");
}

// Very small suffix stemmer: "shipping" -> "ship", "prices" -> "price", "returned" -> "return".
export function stem(word) {
  if (word.length <= 4 || word.endsWith("ss")) return word;
  for (const suffix of ["ing", "ed", "es", "s"]) {
    if (word.endsWith(suffix) && word.length - suffix.length >= 3) {
      let base = word.slice(0, -suffix.length);
      if ((suffix === "ing" || suffix === "ed") && base.length >= 4 && /([b-df-hj-np-tv-z])\1$/.test(base)) {
        base = base.slice(0, -1); // shipp -> ship, stopp -> stop
      }
      if (suffix === "es" && !/(ss|sh|ch|x|z)$/.test(base)) base = word.slice(0, -1); // prices -> price
      return base;
    }
  }
  return word;
}

export function tokenize(text, synonyms = {}) {
  const tokens = [];
  for (const word of normalize(text).split(/\s+/)) {
    if (!word || STOPWORDS.has(word)) continue;
    const canonical = synonyms[word] ?? synonyms[stem(word)] ?? word;
    for (const part of canonical.split(" ")) tokens.push(stem(part));
  }
  return tokens;
}

export class FaqMatcher {
  constructor(knowledgeBase, options = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.synonyms = knowledgeBase.synonyms ?? {};
    this.entries = knowledgeBase.entries.map((entry) => {
      const text = [entry.question, ...(entry.alternatives ?? []), ...(entry.keywords ?? [])].join(" ");
      return { ...entry, terms: new Set(tokenize(text, this.synonyms)) };
    });
    // Inverse document frequency: rare terms ("refund") count more than common ones ("order").
    const docCount = new Map();
    for (const entry of this.entries) for (const t of entry.terms) docCount.set(t, (docCount.get(t) ?? 0) + 1);
    this.idf = new Map([...docCount].map(([t, n]) => [t, Math.log(1 + this.entries.length / n)]));
  }

  score(queryTerms, entry) {
    if (queryTerms.length === 0) return 0;
    let matched = 0;
    let total = 0;
    for (const t of new Set(queryTerms)) {
      const weight = this.idf.get(t) ?? this.options.unknownWeight; // words not in any entry count a little
      total += weight;
      if (entry.terms.has(t)) matched += weight;
    }
    return matched / total; // share of the question's meaning covered by this entry, 0..1
  }

  ask(question) {
    const terms = tokenize(question, this.synonyms);
    const ranked = this.entries
      .map((entry) => ({ entry, score: this.score(terms, entry) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score);
    const best = ranked[0];
    const suggestions = ranked.slice(best && best.score >= this.options.minScore ? 1 : 0)
      .slice(0, this.options.suggestionCount)
      .map((r) => r.entry.question);
    if (!best || best.score < this.options.minScore) {
      return { id: null, answer: this.options.fallback, score: best?.score ?? 0, suggestions };
    }
    return { id: best.entry.id, answer: best.entry.answer, score: best.score, suggestions };
  }
}

export function validateKnowledgeBase(kb) {
  const errors = [];
  const ids = new Set();
  if (!Array.isArray(kb?.entries)) return ["entries must be an array"];
  kb.entries.forEach((e, i) => {
    if (!e.id) errors.push(`entry ${i}: id missing`);
    else if (ids.has(e.id)) errors.push(`entry ${i}: duplicate id ${e.id}`);
    ids.add(e.id);
    if (!e.question) errors.push(`entry ${e.id ?? i}: question missing`);
    if (!e.answer) errors.push(`entry ${e.id ?? i}: answer missing`);
  });
  return errors;
}
