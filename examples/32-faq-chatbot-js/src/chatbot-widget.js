// Embeddable chat widget. Usage:
//   <script type="module">
//     import { mountChatbot } from "./src/chatbot-widget.js";
//     mountChatbot({ knowledgeBaseUrl: "./faq.json" });
//   </script>
import { FaqMatcher } from "./matcher.js";

const STORAGE_KEY = "faq-chatbot-history";

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return []; // private mode or blocked storage: start fresh
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-30)));
  } catch {
    /* ignore */
  }
}

export async function mountChatbot({ knowledgeBaseUrl = "./faq.json", title = "Questions? Ask us", root = document.body } = {}) {
  const kb = await (await fetch(knowledgeBaseUrl)).json();
  const matcher = new FaqMatcher(kb, {
    fallback: `Sorry, I don't know that yet. Please write to ${kb.contact} and we will reply soon.`,
  });

  const container = document.createElement("div");
  container.className = "faq-bot";
  container.innerHTML = `
    <button class="faq-bot__toggle" aria-expanded="false" aria-controls="faq-bot-panel">${title}</button>
    <section class="faq-bot__panel" id="faq-bot-panel" hidden aria-label="${kb.business} FAQ chat">
      <ol class="faq-bot__log" aria-live="polite"></ol>
      <div class="faq-bot__suggestions"></div>
      <form class="faq-bot__form">
        <input name="q" autocomplete="off" placeholder="Type your question" aria-label="Your question" required>
        <button type="submit">Send</button>
      </form>
    </section>`;
  root.appendChild(container);

  const toggle = container.querySelector(".faq-bot__toggle");
  const panel = container.querySelector(".faq-bot__panel");
  const log = container.querySelector(".faq-bot__log");
  const suggestionsBox = container.querySelector(".faq-bot__suggestions");
  const form = container.querySelector(".faq-bot__form");
  const history = loadHistory();

  function addMessage(role, text) {
    const li = document.createElement("li");
    li.className = `faq-bot__msg faq-bot__msg--${role}`;
    li.textContent = text; // textContent, never innerHTML: user input is not trusted
    log.appendChild(li);
    log.scrollTop = log.scrollHeight;
  }

  function showSuggestions(questions) {
    suggestionsBox.replaceChildren(...questions.map((q) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = q;
      b.addEventListener("click", () => ask(q));
      return b;
    }));
  }

  function ask(question) {
    const result = matcher.ask(question);
    addMessage("user", question);
    addMessage("bot", result.answer);
    showSuggestions(result.suggestions);
    history.push({ q: question, a: result.answer, id: result.id });
    saveHistory(history);
  }

  for (const item of history) {
    addMessage("user", item.q);
    addMessage("bot", item.a);
  }
  if (history.length === 0) {
    addMessage("bot", `Hi, I can answer common questions about ${kb.business}.`);
    showSuggestions(kb.entries.slice(0, 3).map((e) => e.question));
  }

  toggle.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    toggle.setAttribute("aria-expanded", String(!panel.hidden));
    if (!panel.hidden) form.q.focus();
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const q = form.q.value.trim();
    if (q) ask(q);
    form.reset();
  });
  return { ask, matcher };
}
