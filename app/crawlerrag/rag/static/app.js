// Recall Chatbot page. Everything that comes from the model or the data is inserted as text nodes:
// the answer is split into paragraphs, list items, **bold** runs and [n] citations by hand, never parsed as HTML.
"use strict";

const KEY = "rag.conversation";
const MATCHED = { both: "vector + keyword", vector: "vector", text: "keyword" };

const $ = (id) => document.getElementById(id);
const els = {
  log: $("log"), empty: $("empty"), notice: $("notice"), docTypes: $("doc-types"), filters: $("filters"),
  form: $("composer"), question: $("question"), send: $("send"), status: $("status"), usage: $("usage"),
  newChat: $("new-chat"),
};
let info = null;
let busy = false;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function storedConversation() {
  try { return sessionStorage.getItem(KEY); } catch { return null; }
}
function storeConversation(id) {
  try { id ? sessionStorage.setItem(KEY, id) : sessionStorage.removeItem(KEY); } catch { /* private mode */ }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let body = null;
  try { body = await response.json(); } catch { /* not JSON, e.g. a proxy error page */ }
  if (!response.ok) {
    const message = (body && body.error) || `HTTP ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return body;
}

// ---------------------------------------------------------------- answer rendering
// Inline: **bold** and citations such as [3], [1, 2] or [1][4].
function inline(text, anchorPrefix) {
  const out = [];
  const pattern = /\*\*([^*]+)\*\*|\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let m;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[1] !== undefined) {
      out.push(el("strong", {}, m[1]));
    } else {
      for (const n of m[2].split(",").map((s) => s.trim())) {
        out.push(el("a", { class: "cite", href: `#${anchorPrefix}-${n}`, title: `Source ${n}` }, n));
      }
    }
    last = pattern.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function renderAnswer(text, anchorPrefix) {
  const frag = document.createDocumentFragment();
  let list = null;
  for (const raw of text.split(/\n/)) {
    const line = raw.trim();
    const bullet = line.match(/^(?:[*\-•])\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== tag) {
        list = el(tag, { class: tag === "ol" ? "answer-list" : null });
        frag.append(list);
      }
      list.append(el("li", {}, inline((bullet || numbered)[1], anchorPrefix)));
      continue;
    }
    list = null;
    if (!line) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    frag.append(heading ? el("p", {}, el("strong", {}, heading[1])) : el("p", {}, inline(line, anchorPrefix)));
  }
  return frag;
}

function sourceItem(s, anchorPrefix) {
  const title = s.url
    ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, s.title || s.doc_id)
    : el("span", {}, s.title || s.doc_id);
  // `value` keeps the printed number equal to s.n: the answer says [5], so the item must read 5 even
  // when the four before it are folded away.
  return el("li", { id: `${anchorPrefix}-${s.n}`, value: s.n },
    title,
    el("div", { class: "src-meta" }, s.doc_id,
      s.matched_by ? el("span", { class: "badge", title: "Which retriever found it" }, MATCHED[s.matched_by] || s.matched_by) : null),
    s.snippet ? el("details", {}, el("summary", {}, "Excerpt given to the model"), el("p", {}, s.snippet)) : null);
}

// Retrieval always hands the model RAG_TOP_K excerpts, and the answer usually leans on one of them:
// measured over 14 real answers, 8 were retrieved every time and the median answer cited one. So the
// list shows what was cited, and keeps the rest one click away - the retrieved set is how a wrong
// answer gets explained, so it is never dropped. `cited` is computed server-side (web.py) so the page,
// a reloaded conversation and the CLI all agree.
function renderSources(sources, anchorPrefix) {
  if (!sources || !sources.length) return null;
  const used = sources.filter((s) => s.cited !== false);
  const rest = sources.filter((s) => s.cited === false);
  const box = el("div", { class: "sources" }, el("h3", {}, "Sources"),
                 el("ol", {}, used.map((s) => sourceItem(s, anchorPrefix))));
  if (rest.length) {
    box.append(el("details", { class: "unused" },
      el("summary", {}, `${rest.length} more retrieved, not cited in the answer`),
      el("ol", {}, rest.map((s) => sourceItem(s, anchorPrefix)))));
  }
  return box;
}

function stats(r) {
  const parts = [];
  if (r.duration_ms) parts.push(`${(r.duration_ms / 1000).toFixed(1)} s`);
  if (r.prompt_tokens || r.output_tokens) {
    parts.push(`${(r.prompt_tokens || 0).toLocaleString()} in / ${(r.output_tokens || 0).toLocaleString()} out tokens`);
  }
  if (r.query_id) parts.push(`query #${r.query_id}`);
  const line = el("div", { class: "stats" }, parts.join(" · "));
  if (r.trace_url) line.append(" · ", el("a", { href: r.trace_url, target: "_blank", rel: "noopener noreferrer" }, "MLflow trace"));
  return line;
}

function botMessage(r) {
  const prefix = `q${r.query_id || Math.random().toString(36).slice(2)}`;
  const box = el("article", { class: r.error ? "msg bot error" : "msg bot" });
  if (r.standalone_question) box.append(el("p", { class: "rewrite" }, `Searched as: ${r.standalone_question}`));
  if (r.error) box.append(el("p", { class: "err" }, r.error));
  if (r.answer) box.append(renderAnswer(r.answer, prefix));
  else if (!r.error) box.append(el("p", {}, "(The model returned no text.)"));
  if (r.truncated) box.append(el("p", { class: "err" }, "The answer was cut off at the output token limit."));
  const sources = renderSources(r.sources, prefix);
  if (sources) box.append(sources);
  box.append(stats(r));
  return box;
}

// The turn is paused: no answer yet, a question back, and examples this index can really answer.
// Clicking one sends it on the same thread, so the paused turn finishes instead of starting a new one.
function clarifyMessage(reply, send) {
  const box = el("article", { class: "msg bot clarify" });
  box.append(el("p", {}, reply.pending_question || "Could you be more specific?"));
  if (reply.samples && reply.samples.length) {
    const list = el("ul", { class: "samples" });
    for (const sample of reply.samples) {
      const button = el("button", { type: "button", class: "sample" }, sample.question);
      button.addEventListener("click", () => {
        if (busy) return;
        box.remove();
        send(sample.question, reply.thread_id);
      });
      const item = el("li", {}, button);
      if (sample.about) item.append(el("span", { class: "about" }, sample.about));
      list.append(item);
    }
    box.append(list);
  }
  box.append(el("p", { class: "stats" }, "Pick one, or type your own question."));
  return box;
}

function userMessage(text) {
  return el("div", { class: "msg user" }, text);
}

function showEmpty() {
  els.empty.hidden = els.log.children.length > 0;
}

// ---------------------------------------------------------------- page state
function selectedDocTypes() {
  return [...els.docTypes.querySelectorAll("input:checked")].map((i) => i.value);
}

function parseFilters() {
  const filters = {};
  for (const raw of els.filters.value.split(/\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const at = line.indexOf("=");
    if (at < 1) throw new Error(`Filter "${line}" must look like key=value`);
    const key = line.slice(0, at).trim();
    const value = line.slice(at + 1).trim();
    if (/^-?\d+$/.test(value)) filters[key] = Number(value);
    else if (/^(true|false)$/i.test(value)) filters[key] = value.toLowerCase() === "true";
    else filters[key] = value;
  }
  return Object.keys(filters).length ? filters : null;
}

function setUsage() {
  if (!info) return;
  const models = `${info.chat_model} · ${info.embed_model}`;
  els.usage.textContent = info.daily_limit
    ? `${info.used_today} / ${info.daily_limit} questions today · ${models}`
    : models;
}

function setBusy(on, message = "") {
  busy = on;
  els.send.disabled = on;
  els.status.textContent = message;
}

async function loadInfo() {
  info = await api("/api/info");
  els.docTypes.querySelectorAll("label").forEach((l) => l.remove());
  for (const d of info.doc_types) {
    const box = el("input", { type: "checkbox", value: d.name, checked: d.default });
    els.docTypes.append(el("label", {}, box, d.label, el("span", { class: "count" }, `(${d.documents.toLocaleString()})`)));
  }
  if (!info.ready) {
    els.notice.textContent = "The search index is empty. Load the recall data before asking questions.";
    els.notice.hidden = false;
  }
  setUsage();
}

async function loadConversation(id) {
  try {
    const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
    for (const t of data.turns) {
      els.log.append(userMessage(t.question), botMessage(t));
    }
  } catch (error) {
    if (error.status === 404) storeConversation(null);
    else throw error;
  }
}

// Server-Sent Events over fetch: EventSource cannot POST, and the question is a POST body.
// Yields {event, data} for each complete event, so a chunk that splits an event mid-way is held back.
async function* sseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const payload = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) payload.push(line.slice(5).trim());
      }
      if (!payload.length) continue;
      try { yield { event, data: JSON.parse(payload.join("\n")) }; } catch { /* ignore a half event */ }
    }
  }
}

// A question that is waiting for a correction: the next thing typed is sent on this thread.
let pendingThread = null;

async function ask(question, threadId = null) {
  let filters;
  try { filters = parseFilters(); } catch (error) { els.status.textContent = error.message; return; }

  els.log.append(userMessage(question));
  const pending = el("article", { class: "msg bot pending" }, "Searching the records...");
  els.log.append(pending);
  showEmpty();
  pending.scrollIntoView({ behavior: "smooth", block: "end" });

  const started = Date.now();
  setBusy(true, "Working... 0 s");
  const timer = setInterval(() => { els.status.textContent = `Working... ${Math.round((Date.now() - started) / 1000)} s`; }, 1000);
  try {
    let id = storedConversation();
    if (!id) {
      id = (await api("/api/conversations", { method: "POST", body: "{}" })).conversation_id;
      storeConversation(id);
    }
    const docTypes = selectedDocTypes();
    const thread = threadId || pendingThread;
    pendingThread = null;
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question, conversation_id: id, thread_id: thread,
        doc_types: docTypes.length ? docTypes : null, filters,
      }),
    });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try { message = (await response.json()).error || message; } catch { /* keep the status */ }
      throw new Error(message);
    }

    // While the model writes, show the text as it arrives. The finished bubble is rebuilt by
    // botMessage() so citations and sources render exactly as they do for a past turn.
    let text = "";
    let live = null;
    let failed = null;
    let reply = null;
    let waiting = null;
    for await (const { event, data } of sseEvents(response)) {
      if (event === "delta") {
        text += data.text || "";
        if (!live) {
          pending.replaceWith(live = el("article", { class: "msg bot streaming" }, el("p", {})));
          live.scrollIntoView({ behavior: "smooth", block: "end" });
        }
        live.firstChild.textContent = text;
      } else if (event === "error") {
        failed = data.error || "The request failed.";
      } else if (event === "question") {
        waiting = data;
      } else if (event === "done") {
        reply = data;
      }
    }
    const box = live || pending;
    if (failed) {
      box.replaceWith(el("article", { class: "msg bot error" }, el("p", { class: "err" }, failed)));
    } else if (waiting) {
      pendingThread = waiting.thread_id;
      // ask() posts the bubble itself - appending one here too showed the picked question twice.
      box.replaceWith(clarifyMessage(waiting, (q, t) => ask(q, t)));
      els.log.lastChild.scrollIntoView({ behavior: "smooth", block: "end" });
    } else if (reply) {
      box.replaceWith(botMessage(reply));
      if (info) { info.used_today += 1; setUsage(); }
    } else {
      box.replaceWith(el("article", { class: "msg bot error" },
        el("p", { class: "err" }, "The connection closed before the answer finished.")));
    }
    setBusy(false, "");
  } catch (error) {
    const box = document.querySelector(".msg.bot.streaming") || pending;
    box.replaceWith(el("article", { class: "msg bot error" }, el("p", { class: "err" }, error.message)));
    setBusy(false, "");
  } finally {
    clearInterval(timer);
  }
}

// ---------------------------------------------------------------- events
els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (!question || busy) return;
  els.question.value = "";
  autosize();
  ask(question);
});

els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

function autosize() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight + 2, 200)}px`;
}
els.question.addEventListener("input", autosize);

els.newChat.addEventListener("click", () => {
  if (busy) return;
  storeConversation(null);
  els.log.replaceChildren();
  showEmpty();
  els.question.focus();
});

document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => { if (!busy) ask(button.textContent.trim()); });
});

(async () => {
  try {
    await loadInfo();
    const id = storedConversation();
    if (id) await loadConversation(id);
  } catch (error) {
    els.notice.textContent = `Could not reach the chatbot service: ${error.message}`;
    els.notice.hidden = false;
  }
  showEmpty();
  els.question.focus();
})();
