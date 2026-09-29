const elements = {
  chatForm: document.querySelector("#chat-form"),
  questionInput: document.querySelector("#question-input"),
  sendButton: document.querySelector("#send-button"),
  messages: document.querySelector("#messages"),
  welcome: document.querySelector("#welcome"),
  indexState: document.querySelector("#index-state"),
  chunkCount: document.querySelector("#chunk-count"),
  indexIndicator: document.querySelector("#index-indicator"),
  documentDot: document.querySelector("#document-dot"),
  connection: document.querySelector("#connection"),
  connectionLabel: document.querySelector("#connection-label"),
};

function setConnection(isOnline) {
  elements.connection.classList.toggle("is-online", isOnline);
  elements.connection.classList.toggle("is-offline", !isOnline);
  elements.connectionLabel.textContent = isOnline ? "API connected" : "API unavailable";
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Could not read index status");
    const status = await response.json();
    setConnection(true);
    elements.chunkCount.textContent = status.chunks.toLocaleString();
    elements.indexIndicator.classList.toggle("is-ready", status.indexed);
    elements.documentDot.classList.toggle("is-ready", status.indexed);
    elements.documentDot.setAttribute("aria-label", status.indexed ? "Indexed" : "Not indexed");
    elements.indexState.textContent = status.indexed
      ? "The paper index is ready to search."
      : "Run notebooks/01_pdf_processing.ipynb to create the index.";
  } catch {
    setConnection(false);
    elements.indexState.textContent = "The API could not be reached.";
    elements.chunkCount.textContent = "--";
  }
}

function addMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : "PAPERTRAIL · TINYLLAMA";
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  article.append(label, body);

  if (sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    for (const source of sources) {
      const item = document.createElement("div");
      item.className = "source";
      const citation = document.createElement("strong");
      citation.textContent = `PAGE ${source.page}`;
      const excerpt = document.createElement("span");
      excerpt.textContent = source.excerpt;
      item.append(citation, excerpt);
      sourceList.append(item);
    }
    article.append(sourceList);
  }

  elements.messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return article;
}

function showTyping() {
  const indicator = document.createElement("div");
  indicator.className = "typing";
  indicator.setAttribute("aria-label", "Generating an answer");
  indicator.innerHTML = "<span></span><span></span><span></span>";
  elements.messages.append(indicator);
  indicator.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return indicator;
}

async function askQuestion(question) {
  const cleanQuestion = question.trim();
  if (!cleanQuestion || elements.sendButton.disabled) return;
  elements.welcome.hidden = true;
  addMessage("user", cleanQuestion);
  elements.questionInput.value = "";
  elements.questionInput.style.height = "auto";
  elements.sendButton.disabled = true;
  const typing = showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: cleanQuestion, top_k: 4 }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "The question could not be answered.");
    typing.remove();
    addMessage("assistant", result.answer, result.sources || []);
  } catch (error) {
    typing.remove();
    const message = addMessage("assistant", error.message || "Something went wrong. Please try again.");
    message.classList.add("message-error");
  } finally {
    elements.sendButton.disabled = false;
    elements.questionInput.focus();
  }
}

elements.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(elements.questionInput.value);
});

elements.questionInput.addEventListener("input", () => {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 160)}px`;
});

elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => askQuestion(button.dataset.question));
});

refreshStatus();