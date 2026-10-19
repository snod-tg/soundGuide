"use strict";

const $ = (id) => document.getElementById(id);
const form = $("question-form");
const question = $("question");
const model = $("model");
let active = null;
let available = false;
const labels = {
  answered: "找到依据", insufficient_evidence: "证据不足",
  clarification_needed: "需要补充", loading: "正在查找", error: "暂未完成", idle: "等待提问",
};

function badge(state) {
  $("result-badge").textContent = labels[state];
  $("result-badge").dataset.state = state;
}

function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "helpguide.sony.net" ? url.href : null;
  } catch { return null; }
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function externalLink(url, text) {
  const link = node("a", "", text);
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function clearResult() {
  $("result").hidden = true;
  $("evidence").hidden = true;
  $("answer-text").replaceChildren();
  $("citations").replaceChildren();
}

function setBusy(busy) {
  $("submit").disabled = busy || !available;
  $("submit").firstChild.textContent = busy ? "查找中… " : "查找答案 ";
  model.disabled = busy || !available;
  question.disabled = busy;
  document.querySelectorAll("[data-question]").forEach((button) => { button.disabled = busy; });
  $("cancel").hidden = !busy;
  $("answer-panel").setAttribute("aria-busy", String(busy));
}

function renderAnswer(data, elapsed) {
  if (!Object.hasOwn(labels, data.status) || !["answered", "insufficient_evidence", "clarification_needed"].includes(data.status)
      || typeof data.answer !== "string" || !Array.isArray(data.citations)) {
    throw new Error("服务返回的回答格式异常，请重试。");
  }
  badge(data.status);
  $("status").textContent = "";
  $("result").hidden = false;
  $("result-context").textContent = `${data.model} · ${elapsed.toFixed(1)} 秒 · ${data.question}`;
  const refs = new Set(data.citations.map((citation) => String(citation.number)));
  // Treat all model/source text as text, never as HTML or executable Markdown.
  for (const line of data.answer.split("\n").filter((part) => part.trim())) {
    const paragraph = node("p");
    for (const part of line.split(/(\[\d+\])/g)) {
      const number = /^\[(\d+)\]$/.exec(part)?.[1];
      if (number && refs.has(number)) {
        const link = node("a", "citation-ref", part);
        link.href = `#citation-${number}`;
        link.setAttribute("aria-label", `查看依据 ${number}`);
        link.addEventListener("click", () => { $(`citation-${number}`).open = true; });
        paragraph.append(link);
      } else paragraph.append(document.createTextNode(part));
    }
    $("answer-text").append(paragraph);
  }
  $("answer-note").textContent = data.status === "answered"
    ? "引用可追溯到官方原文；生成内容仍可能误解条件或遗漏步骤，请核对后操作。"
    : data.status === "clarification_needed"
      ? "请在左侧补充完整问题后重新提问；本页不保留上一轮上下文。"
      : "这只表示本次证据不足，不代表完整手册中一定没有相关说明。";
  $("evidence").hidden = data.citations.length === 0;
  $("citation-count").textContent = `${data.citations.length} 条引用`;
  for (const citation of data.citations) {
    const details = node("details", "citation");
    details.id = `citation-${citation.number}`;
    details.open = data.citations.length === 1;
    const location = citation.page ? `PDF 第 ${citation.page} 页` : "网页章节";
    details.append(node("summary", "", `[${citation.number}] ${citation.section_title || "官方帮助指南"} · ${location}`));
    const body = node("div", "citation-body");
    body.append(node("blockquote", "", citation.quote));
    body.append(node("p", "citation-meta", `${citation.model} · 指南 ${citation.document_version} · ${location}${citation.machine_translated ? " · 官方机器翻译" : ""}`));
    const url = safeSourceUrl(citation.url);
    if (url) body.append(externalLink(url, citation.page ? "打开 Sony 官方 PDF ↗" : "打开 Sony 官方章节 ↗"));
    details.append(body);
    $("citations").append(details);
  }
}

question.addEventListener("input", () => {
  $("question-count").textContent = `${question.value.length} / 300`;
  question.setCustomValidity("");
});
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.dataset.question;
    question.dispatchEvent(new Event("input"));
    question.focus();
  });
});
model.addEventListener("change", () => {
  clearResult();
  badge("idle");
  $("empty").hidden = false;
  $("status").textContent = "";
});
$("cancel").addEventListener("click", () => {
  if (active) { active.cancelled = true; active.controller.abort(); }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (active) return;
  if (!question.value.trim()) {
    question.setCustomValidity("请输入具体问题。");
    question.reportValidity();
    return;
  }
  const payload = { model: model.value, question: question.value.trim() };
  const request = { controller: new AbortController(), cancelled: false };
  active = request;
  const started = performance.now();
  clearResult();
  $("empty").hidden = true;
  $("status").className = "status";
  badge("loading");
  setBusy(true);
  const progress = () => {
    $("status").textContent = `正在检索资料并生成回答… ${Math.floor((performance.now() - started) / 1000)} 秒`;
  };
  progress();
  const ticker = setInterval(progress, 1000);
  const timeout = setTimeout(() => request.controller.abort(), 90000);
  if (matchMedia("(max-width: 760px)").matches) $("answer-panel").scrollIntoView({ block: "start" });
  try {
    const response = await fetch("/api/answer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), signal: request.controller.signal,
    });
    if (!response.ok) {
      const messages = {
        409: "资料索引与当前配置不一致，需要维护索引后再试。",
        422: "请选择支持的型号，并将问题控制在 1～300 字以内。",
        502: "模型回答未通过引用或完整性检查，请重新提问。",
        503: "检索、数据库或生成服务暂不可用，请确认相关服务和连接后重试。",
      };
      throw new Error(messages[response.status] || `请求未完成（HTTP ${response.status}），请稍后重试。`);
    }
    renderAnswer(await response.json(), (performance.now() - started) / 1000);
  } catch (error) {
    clearResult();
    badge("error");
    $("status").className = "status error";
    $("status").textContent = request.controller.signal.aborted
      ? request.cancelled
        ? "已停止等待。服务端可能仍在处理本次请求，请稍后再试。"
        : "等待超过 90 秒，已停止等待。服务端可能仍在处理，请稍后再试。"
      : error instanceof TypeError ? "无法连接服务，请检查网络和应用服务后重试。" : error.message;
  } finally {
    clearInterval(ticker);
    clearTimeout(timeout);
    active = null;
    setBusy(false);
  }
});

async function loadSources() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch("/api/sources", { signal: controller.signal });
    if (!response.ok) throw new Error("Source service unavailable");
    const sources = await response.json();
    const indexed = sources.filter((source) => source.status === "indexed");
    available = indexed.length > 0;
    model.replaceChildren();
    const categories = { "over-ear": "头戴式", "true-wireless": "真无线入耳式", neckband: "颈挂式" };
    for (const [category, label] of Object.entries(categories)) {
      const group = node("optgroup");
      group.label = label;
      for (const source of sources.filter((item) => item.category === category)) {
        const option = node("option", "", `Sony ${source.model}${source.status === "indexed" ? "" : "（暂不可用）"}`);
        option.value = source.model;
        option.disabled = source.status !== "indexed";
        group.append(option);
      }
      if (group.children.length) model.append(group);
    }
    if (available) model.value = indexed.some((source) => source.model === "WH-1000XM5") ? "WH-1000XM5" : indexed[0].model;
    else {
      const option = node("option", "", "暂无已索引型号");
      option.value = "";
      model.prepend(option);
      model.value = "";
    }
    setBusy(false);
    $("source-summary").textContent = `${indexed.length} / ${sources.length} 份资料已索引`;
    const states = { indexed: "已索引", candidate: "未下载", downloaded: "待索引", needs_reindex: "需重新索引" };
    for (const source of sources) {
      const card = node("article", "source-card");
      const info = node("div");
      info.append(node("strong", "", source.model));
      info.append(node("p", "", `${states[source.status] || "状态未知"} · ${source.chunks} 个片段 · ${source.format === "html_manual" ? "网页" : "PDF"}指南 ${source.document_version}${source.machine_translated ? " · 官方机器翻译" : ""}`));
      card.append(info);
      const url = safeSourceUrl(source.url);
      if (url) card.append(externalLink(url, "官方指南 ↗"));
      $("source-list").append(card);
    }
  } catch {
    available = false;
    model.replaceChildren(node("option", "", "资料状态读取失败"));
    setBusy(false);
    $("source-summary").textContent = "状态暂不可用";
    $("source-list").replaceChildren();
    $("source-error").hidden = false;
  } finally { clearTimeout(timeout); }
}

loadSources();
