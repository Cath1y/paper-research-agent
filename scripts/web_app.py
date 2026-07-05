#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data/metadata/web_runs"
MEMORY_DIR = ROOT / "data/memory"


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Embodied AI Literature Assistant</title>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [["\\(", "\\)"], ["$", "$"]],
        displayMath: [["\\[", "\\]"], ["$$", "$$"]],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      },
      startup: { typeset: false }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --paper: #fffaf0;
      --ink: #182230;
      --muted: #6b655c;
      --line: #d8d0c2;
      --brand: #173b63;
      --brand-soft: #e8eff6;
      --accent: #8a5a24;
      --ok: #047857;
      --warn: #b45309;
      --bad: #b42318;
      --code: #111827;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      height: 100vh;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
    }
    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      max-height: 100vh;
      border-right: 1px solid var(--line);
      background: #fbf8f1;
      padding: 20px;
      overflow: auto;
    }
    main {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
      padding: 0;
    }
    h1 {
      margin: 0 0 4px;
      color: var(--brand);
      font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
      font-size: 22px;
      line-height: 1.2;
    }
    h2 {
      margin: 22px 0 10px;
      font-size: 15px;
    }
    .subtle { color: var(--muted); }
    label {
      display: block;
      margin: 12px 0 5px;
      font-weight: 650;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
    }
    textarea { min-height: 126px; resize: vertical; }
    #question {
      min-height: 58px;
      max-height: 180px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .checks {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .threads {
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    .thread-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .small-button {
      width: auto;
      margin: 0;
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 12px;
    }
    .thread-list {
      display: grid;
      gap: 6px;
      max-height: 260px;
      overflow: auto;
    }
    .thread-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf8;
      padding: 8px 9px;
      cursor: pointer;
      transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
    }
    .thread-item:hover {
      border-color: #b79568;
      transform: translateY(-1px);
    }
    .thread-item.active {
      border-color: var(--brand);
      background: var(--brand-soft);
      box-shadow: inset 3px 0 0 var(--brand);
    }
    .thread-name {
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .thread-last {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      margin-top: 2px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
    }
    .check input { width: auto; }
    button {
      width: 100%;
      margin-top: 16px;
      border: 0;
      border-radius: 8px;
      background: var(--brand);
      color: #fff;
      padding: 11px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.94);
      backdrop-filter: blur(8px);
    }
    .status {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 999px;
      padding: 10px 12px;
      color: var(--muted);
    }
    .status.ok { color: var(--ok); }
    .status.bad { color: var(--bad); }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric, .section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
    }
    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1.1;
      margin-bottom: 2px;
    }
    .settings {
      margin-top: 16px;
    }
    .settings details {
      margin-top: 8px;
    }
    .chat {
      overflow: auto;
      padding: 24px;
      background:
        linear-gradient(rgba(24, 34, 48, 0.035) 1px, transparent 1px),
        var(--bg);
      background-size: 100% 34px;
    }
    .message {
      max-width: 980px;
      margin: 0 auto 14px;
      display: flex;
    }
    .message.user { justify-content: flex-end; }
    .bubble {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 14px 16px;
      max-width: min(920px, 100%);
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .user .bubble {
      background: #edf3f8;
      border-color: #c5d3df;
    }
    .assistant .bubble {
      background: var(--panel);
    }
    .bubble-head {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .composer {
      border-top: 1px solid var(--line);
      background: #fbf8f1;
      padding: 14px 22px 18px;
    }
    .composer-inner {
      max-width: 980px;
      margin: 0 auto;
    }
    .composer-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 132px;
      gap: 10px;
      align-items: end;
    }
    .composer button {
      margin-top: 0;
      height: 58px;
    }
    .answer {
      white-space: pre-wrap;
      font-size: 14.5px;
    }
    .markdown {
      white-space: normal;
      line-height: 1.72;
    }
    .markdown h1,
    .markdown h2,
    .markdown h3,
    .markdown h4 {
      margin: 12px 0 7px;
      line-height: 1.3;
    }
    .markdown h1 { font-size: 19px; }
    .markdown h2 { font-size: 17px; }
    .markdown h3 { font-size: 15.5px; }
    .markdown h4 { font-size: 14.5px; }
    .markdown p {
      margin: 8px 0 10px;
    }
    .markdown ul,
    .markdown ol {
      margin: 8px 0 8px 22px;
      padding: 0;
    }
    .markdown li {
      margin: 4px 0;
    }
    .markdown blockquote {
      margin: 10px 0;
      padding: 8px 12px;
      border-left: 3px solid var(--line);
      color: var(--muted);
      background: #f9fafb;
    }
    .markdown code {
      background: #eef2f7;
      border-radius: 4px;
      padding: 1px 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }
    .markdown pre {
      white-space: pre;
      max-height: none;
    }
    .markdown pre code {
      background: transparent;
      padding: 0;
      color: inherit;
    }
    .markdown table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 13.5px;
    }
    .markdown th,
    .markdown td {
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }
    .markdown th {
      background: #f2f4f7;
      font-weight: 700;
    }
    .markdown .MathJax {
      font-size: 0.96em !important;
    }
    .papers {
      display: grid;
      gap: 8px;
    }
    .paper {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--panel);
    }
    .paper-title { font-weight: 700; }
    .paper-meta { color: var(--muted); margin-top: 3px; }
    a { color: var(--brand); text-decoration: none; }
    a:hover { text-decoration: underline; }
    details {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      margin-top: 10px;
      padding: 8px 10px;
    }
    summary { cursor: pointer; font-weight: 700; }
    pre {
      overflow: auto;
      background: var(--code);
      color: #e5e7eb;
      border-radius: 6px;
      padding: 12px;
      max-height: 460px;
      white-space: pre-wrap;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 22px;
      color: var(--muted);
      background: #fff;
      text-align: center;
    }
    @media (max-width: 900px) {
      .app {
        height: auto;
        min-height: 100vh;
        overflow: visible;
        grid-template-columns: 1fr;
      }
      aside {
        position: static;
        height: auto;
        max-height: none;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .grid { grid-template-columns: 1fr; }
      main { min-height: 70vh; }
      .composer-row { grid-template-columns: 1fr; }
      .composer button { height: auto; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Literature Assistant</h1>
      <div class="subtle">Evidence-first research desk · LangGraph + PaperQA</div>

      <div class="threads">
        <div class="thread-bar">
          <div>
            <strong>Threads</strong>
            <div id="current-thread" class="subtle">web_demo</div>
          </div>
          <button id="new-thread" class="small-button" type="button">新建会话</button>
        </div>
        <div id="thread-list" class="thread-list"></div>
      </div>

      <form id="ask-form">
        <input id="thread_id" name="thread_id" type="hidden" value="web_demo" />

        <div class="settings">
          <details open>
            <summary>常用设置</summary>
            <div class="row">
              <div>
                <label for="year_from">起始年份</label>
                <input id="year_from" name="year_from" type="number" value="2024" />
              </div>
              <div>
                <label for="year_to">结束年份</label>
                <input id="year_to" name="year_to" type="number" value="2026" />
              </div>
            </div>

            <div class="row">
              <div>
                <label for="paperqa_k">PDF 数</label>
                <input id="paperqa_k" name="paperqa_k" type="number" value="6" min="1" max="20" />
              </div>
              <div>
                <label for="agent_search_count">PaperQA Search</label>
                <input id="agent_search_count" name="agent_search_count" type="number" value="6" min="1" max="30" />
              </div>
            </div>

            <label for="paperqa_reader_mode">Reader 模式</label>
            <select id="paperqa_reader_mode" name="paperqa_reader_mode">
              <option value="explicit-tools">explicit-tools</option>
              <option value="paperqa-agent">paperqa-agent</option>
              <option value="agent-only">agent-only</option>
            </select>

            <div class="checks">
              <label class="check"><input name="include_arxiv" type="checkbox" /> 本地 arXiv 库</label>
              <label class="check"><input name="dry_run" type="checkbox" /> 只检索不精读</label>
              <label class="check"><input name="disable_web_search" type="checkbox" /> 关闭普通 Web Search</label>
            </div>
          </details>

          <details>
            <summary>高级设置</summary>
            <div class="row">
              <div>
                <label for="synthesis_max_tokens">回答 token</label>
                <input id="synthesis_max_tokens" name="synthesis_max_tokens" type="number" value="9000" min="512" />
              </div>
              <div>
                <label for="timeout">超时秒数</label>
                <input id="timeout" name="timeout" type="number" value="420" min="30" />
              </div>
            </div>

            <label for="extra_args">额外命令行参数</label>
            <input id="extra_args" name="extra_args" placeholder="例如：--paper-search-loop-iterations 2" />
          </details>
        </div>
      </form>
    </aside>

    <main>
      <div class="toolbar">
        <div>
          <h1>对话</h1>
          <div class="subtle">每次提问会调用完整 LangGraph 工作流；长调研可能需要几分钟。</div>
        </div>
        <div id="status" class="status">就绪</div>
      </div>
      <div id="chat" class="chat">
        <div class="message assistant">
          <div class="bubble">
            <div class="bubble-head">Assistant</div>
            你好，我可以帮你做论文调研、论文精读、方向综述和已读论文追问。左侧调整参数，在底部输入问题即可。
          </div>
        </div>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <div class="composer-row">
            <textarea id="question" name="question" form="ask-form" placeholder="输入问题，例如：帮我精读 OpenVLA，并调研它的相关方向"></textarea>
            <button id="submit" form="ask-form" type="submit">发送</button>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script>
    const form = document.getElementById("ask-form");
    const statusEl = document.getElementById("status");
    const chatEl = document.getElementById("chat");
    const submit = document.getElementById("submit");
    const questionEl = document.getElementById("question");
    const threadInput = document.getElementById("thread_id");
    const threadListEl = document.getElementById("thread-list");
    const currentThreadEl = document.getElementById("current-thread");
    const newThreadBtn = document.getElementById("new-thread");

    function value(name) {
      return new FormData(form).get(name);
    }
    function checked(name) {
      return Boolean(new FormData(form).get(name));
    }
    function escapeHtml(text) {
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }
    function renderInlineMarkdown(text) {
      let html = escapeHtml(text);
      html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_match, label, url) => {
        return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
      });
      html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
      html = html.replace(/\n/g, "<br>");
      return html;
    }
    function isTableSeparator(line) {
      return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    }
    function splitTableRow(line) {
      return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
    }
    function normalizeMarkdown(markdown) {
      let text = String(markdown ?? "").replace(/\r\n/g, "\n");
      const codeBlocks = [];
      text = text.replace(/```[\s\S]*?```/g, (block) => {
        const token = `\u0000CODE_BLOCK_${codeBlocks.length}\u0000`;
        codeBlocks.push(block);
        return token;
      });
      text = text
        .replace(/\\n(?=\s*(?:[-*]|\d+\.|#{1,4}\s+))/g, "\n")
        .replace(/([^\n])(\s*)(#{1,4}\s+\d*\.?\s*)/g, "$1\n\n$3")
        .replace(/([^\n])(\s*)(#{1,4}\s+)/g, "$1\n\n$3")
        .replace(/([。！？；;])\s*(-\s+)/g, "$1\n$2")
        .replace(/([。！？；;])\s*(\d+\.\s+)/g, "$1\n$2")
        .replace(/(\[PaperQA\/PDF\][^\n]*?)(\s+)(#{1,4}\s+)/g, "$1\n\n$3")
        .replace(/(\[[^\]]+\][^\n]*?)(\s+)(#{1,4}\s+)/g, "$1\n\n$3")
        .replace(/([^\n])\s+(-\s+\*\*[^*]+?\*\*)/g, "$1\n$2")
        .replace(/([^\n])\s+(-\s+[\u4e00-\u9fffA-Za-z0-9][^：:]{0,40}[：:])/g, "$1\n$2");
      for (const [index, block] of codeBlocks.entries()) {
        text = text.replace(`\u0000CODE_BLOCK_${index}\u0000`, block);
      }
      return text;
    }
    function renderMarkdown(markdown) {
      const lines = normalizeMarkdown(markdown).split("\n");
      const out = [];
      let paragraph = [];
      let list = null;
      let inCode = false;
      let codeLines = [];

      function flushParagraph() {
        if (!paragraph.length) return;
        out.push(`<p>${renderInlineMarkdown(paragraph.join("\n"))}</p>`);
        paragraph = [];
      }
      function flushList() {
        if (!list) return;
        out.push(`<${list.type}>${list.items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${list.type}>`);
        list = null;
      }
      function flushBlocks() {
        flushParagraph();
        flushList();
      }

      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
          if (inCode) {
            out.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
            codeLines = [];
            inCode = false;
          } else {
            flushBlocks();
            inCode = true;
            codeLines = [];
          }
          continue;
        }
        if (inCode) {
          codeLines.push(line);
          continue;
        }
        if (!line.trim()) {
          flushBlocks();
          continue;
        }

        if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
          flushBlocks();
          const headers = splitTableRow(line);
          i += 2;
          const rows = [];
          while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
            rows.push(splitTableRow(lines[i]));
            i += 1;
          }
          i -= 1;
          out.push(
            `<table><thead><tr>${headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead>` +
            `<tbody>${rows.map((row) => `<tr>${headers.map((_h, idx) => `<td>${renderInlineMarkdown(row[idx] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`
          );
          continue;
        }

        const heading = line.match(/^(#{1,4})\s+(.+)$/);
        if (heading) {
          flushBlocks();
          const level = heading[1].length;
          out.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }

        const quote = line.match(/^>\s?(.*)$/);
        if (quote) {
          flushBlocks();
          out.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
          continue;
        }

        const unordered = line.match(/^\s*[-*]\s+(.+)$/);
        if (unordered) {
          flushParagraph();
          if (!list || list.type !== "ul") {
            flushList();
            list = {type: "ul", items: []};
          }
          list.items.push(unordered[1]);
          continue;
        }

        const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
        if (ordered) {
          flushParagraph();
          if (!list || list.type !== "ol") {
            flushList();
            list = {type: "ol", items: []};
          }
          list.items.push(ordered[1]);
          continue;
        }

        paragraph.push(line.trim());
      }
      if (inCode) {
        out.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      }
      flushBlocks();
      return out.join("\n");
    }
    function paperRows(items) {
      if (!items || !items.length) return '<div class="empty">没有选中的论文。</div>';
      return `<div class="papers">${items.map((item) => {
        const result = item.result || item;
        const authors = (result.authors || []).slice(0, 8).join(", ");
        const url = result.paper_url || result.pdf_url || "";
        const pdf = result.pdf_url || "";
        return `<div class="paper">
          <div class="paper-title">${escapeHtml(result.title || "Untitled")}</div>
          <div class="paper-meta">${escapeHtml(result.year || "?")} · ${escapeHtml(result.venue || result.corpus || "?")} · ${escapeHtml(authors)}</div>
          <div class="paper-meta">cache: ${escapeHtml(item.cache_status || "unknown")}</div>
          ${url ? `<div><a href="${escapeHtml(url)}" target="_blank">paper</a>${pdf ? ` · <a href="${escapeHtml(pdf)}" target="_blank">pdf</a>` : ""}</div>` : ""}
        </div>`;
      }).join("")}</div>`;
    }
    function traceList(items) {
      if (!items || !items.length) return '<div class="empty">无 trace。</div>';
      return `<pre>${escapeHtml(JSON.stringify(items, null, 2))}</pre>`;
    }
    function evidenceRows(items) {
      if (!items || !items.length) return '<div class="empty">没有 PaperQA 证据。</div>';
      return `<div class="papers">${items.slice(0, 12).map((item, index) => {
        const citation = item.citation || item.text_name || item.docname || `Evidence ${index + 1}`;
        const snippet = item.context || item.snippet || "";
        return `<div class="paper">
          <div class="paper-title">${escapeHtml(citation)}</div>
          <div class="paper-meta">score: ${escapeHtml(item.score ?? "N/A")}</div>
          <div>${escapeHtml(snippet)}</div>
        </div>`;
      }).join("")}</div>`;
    }
    function addMessage(role, html) {
      const node = document.createElement("div");
      node.className = `message ${role}`;
      node.innerHTML = `<div class="bubble"><div class="bubble-head">${role === "user" ? "You" : "Assistant"}</div>${html}</div>`;
      chatEl.appendChild(node);
      chatEl.scrollTop = chatEl.scrollHeight;
      return node;
    }
    function typesetMath(node) {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([node]).catch(() => {});
      }
    }
    function clearChat() {
      chatEl.innerHTML = `<div class="message assistant">
        <div class="bubble">
          <div class="bubble-head">Assistant</div>
          你好，我可以帮你做论文调研、论文精读、方向综述和已读论文追问。左侧可以切换历史 thread，底部输入问题即可。
        </div>
      </div>`;
    }
    function threadTimeLabel(value) {
      if (!value) return "";
      try {
        return new Date(value).toLocaleString();
      } catch (_) {
        return value;
      }
    }
    function renderThreadList(items) {
      const current = value("thread_id") || "web_demo";
      if (!items || !items.length) {
        threadListEl.innerHTML = '<div class="empty">暂无历史 thread。</div>';
        return;
      }
      threadListEl.innerHTML = items.map((item) => `
        <div class="thread-item ${item.thread_id === current ? "active" : ""}" data-thread="${escapeHtml(item.thread_id)}">
          <div class="thread-name">${escapeHtml(item.thread_id)}</div>
          <div class="thread-last">${escapeHtml(item.last_question || "空聊天")}</div>
          <div class="thread-last">${escapeHtml(item.turn_count || 0)} turns · ${escapeHtml(threadTimeLabel(item.updated_at))}</div>
        </div>
      `).join("");
      threadListEl.querySelectorAll(".thread-item").forEach((node) => {
        node.addEventListener("click", () => switchThread(node.dataset.thread));
      });
    }
    async function loadThreads() {
      const response = await fetch("/api/threads");
      const data = await response.json();
      renderThreadList(data.threads || []);
    }
    function renderHistory(records) {
      clearChat();
      for (const record of records || []) {
        if (record.question) {
          addMessage("user", `<div class="answer">${escapeHtml(record.question)}</div>`);
        }
        const answer = record.answer_full || record.answer_excerpt || record.episode_summary || "";
        const papers = record.selected_papers || [];
        const evidence = record.paperqa_evidence_contexts || [];
        if (answer || papers.length || evidence.length) {
          addMessage("assistant", `
            <section class="section"><div class="answer markdown">${renderMarkdown(answer || "这一轮没有保存回答。")}</div></section>
            <div class="grid">
              <div class="metric"><strong>${papers.length}</strong><span>选中 PDF</span></div>
              <div class="metric"><strong>${(record.paperqa || {}).evidence_count ?? evidence.length ?? 0}</strong><span>PaperQA Evidence</span></div>
              <div class="metric"><strong>${(record.web_sources || []).length}</strong><span>网页证据</span></div>
            </div>
            <details>
              <summary>论文与 PDF (${papers.length})</summary>
              ${paperRows(papers)}
            </details>
            <details>
              <summary>PaperQA 证据 (${(record.paperqa || {}).evidence_count ?? evidence.length ?? 0})</summary>
              ${evidenceRows(evidence)}
            </details>
          `);
          typesetMath(chatEl.lastElementChild);
        }
      }
    }
    async function switchThread(threadId) {
      if (!threadId) return;
      threadInput.value = threadId;
      currentThreadEl.textContent = threadId;
      statusEl.textContent = "已切换 thread";
      statusEl.className = "status";
      const response = await fetch(`/api/threads/${encodeURIComponent(threadId)}`);
      const data = await response.json();
      renderHistory(data.records || []);
      await loadThreads();
    }
    function defaultThreadName() {
      const now = new Date();
      const stamp = [
        now.getFullYear(),
        String(now.getMonth() + 1).padStart(2, "0"),
        String(now.getDate()).padStart(2, "0"),
        String(now.getHours()).padStart(2, "0"),
        String(now.getMinutes()).padStart(2, "0")
      ].join("");
      return `research_${stamp}`;
    }
    async function createThread() {
      const rawName = window.prompt("给新会话起一个名字：", defaultThreadName());
      if (rawName === null) return;
      const response = await fetch("/api/threads", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({thread_id: rawName})
      });
      const data = await response.json();
      if (!response.ok) {
        statusEl.textContent = data.error || "新建会话失败。";
        statusEl.className = "status bad";
        return;
      }
      await switchThread(data.thread_id);
      questionEl.focus();
    }
    function renderAssistant(data) {
      const state = data.state || {};
      const finalAnswer = state.final_answer || "";
      const selected = state.selected || [];
      const web = state.web_evidence || [];
      const paperqa = state.paperqa_trace || {};
      const memoryPaper = state.memory_paper_trace || {};
      const evidence = paperqa.evidence_contexts || [];
      return `
        <section class="section">
          <div class="answer markdown">${renderMarkdown(finalAnswer || "没有生成最终回答。")}</div>
        </section>
        <div class="grid">
          <div class="metric"><strong>${selected.length}</strong><span>选中 PDF</span></div>
          <div class="metric"><strong>${paperqa.evidence_count ?? evidence.length ?? 0}</strong><span>PaperQA Evidence</span></div>
          <div class="metric"><strong>${web.length}</strong><span>网页证据</span></div>
        </div>
        <details open>
          <summary>论文与 PDF (${selected.length})</summary>
          ${paperRows(selected)}
        </details>
        <details>
          <summary>PaperQA 证据 (${paperqa.evidence_count ?? evidence.length ?? 0})</summary>
          ${evidenceRows(evidence)}
        </details>
      `;
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = value("question");
      if (!question.trim()) {
        statusEl.textContent = "请先输入问题。";
        statusEl.className = "status bad";
        return;
      }
      addMessage("user", `<div class="answer">${escapeHtml(question)}</div>`);
      submit.disabled = true;
      statusEl.textContent = "运行中";
      statusEl.className = "status";
      const pending = addMessage("assistant", `<div class="subtle">正在调用 LangGraph / PaperQA，请稍等...</div>`);
      const payload = {
        question,
        thread_id: value("thread_id") || "web_demo",
        year_from: value("year_from"),
        year_to: value("year_to"),
        paperqa_k: value("paperqa_k"),
        agent_search_count: value("agent_search_count"),
        synthesis_max_tokens: value("synthesis_max_tokens"),
        timeout: value("timeout"),
        paperqa_reader_mode: value("paperqa_reader_mode"),
        frontier_enabled: true,
        include_arxiv: checked("include_arxiv"),
        dry_run: checked("dry_run"),
        disable_web_search: checked("disable_web_search"),
        extra_args: value("extra_args") || ""
      };
      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          statusEl.textContent = data.error || "运行失败。";
          statusEl.className = "status bad";
          pending.querySelector(".bubble").innerHTML = `<div class="bubble-head">Assistant</div><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
        } else {
          statusEl.textContent = `完成：${data.elapsed_seconds.toFixed(1)}s`;
          statusEl.className = "status ok";
          pending.querySelector(".bubble").innerHTML = `<div class="bubble-head">Assistant</div>${renderAssistant(data)}`;
          typesetMath(pending);
          questionEl.value = "";
          await loadThreads();
        }
      } catch (error) {
        statusEl.textContent = `请求失败：${error}`;
        statusEl.className = "status bad";
        pending.querySelector(".bubble").innerHTML = `<div class="bubble-head">Assistant</div><pre>${escapeHtml(error)}</pre>`;
      } finally {
        submit.disabled = false;
      }
    });
    newThreadBtn.addEventListener("click", createThread);
    loadThreads().then(() => switchThread(value("thread_id") || "web_demo"));
  </script>
</body>
</html>
"""


def _bool_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def _append_value(command: list[str], flag: str, value: Any) -> None:
    text = str(value or "").strip()
    if text:
        command.extend([flag, text])


def _process_output_text(value: Any, *, limit: int = 20000) -> str:
    """Convert subprocess output to JSON-safe text."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    if len(text) > limit:
        return text[-limit:]
    return text


def _safe_thread_id(value: Any) -> str:
    thread_id = re.sub(r"[^\w_.-]+", "_", str(value or "web_demo")).strip("._-")
    return thread_id or "web_demo"


def _thread_path(thread_id: str) -> Path:
    return MEMORY_DIR / f"{_safe_thread_id(thread_id)}.jsonl"


def _unique_thread_id(thread_id: str) -> str:
    base = _safe_thread_id(thread_id)
    candidate = base
    index = 2
    while _thread_path(candidate).exists():
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _read_thread_records(thread_id: str) -> list[dict[str, Any]]:
    path = _thread_path(thread_id)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _thread_summary(path: Path) -> dict[str, Any]:
    thread_id = path.stem
    records = _read_thread_records(thread_id)
    last = records[-1] if records else {}
    updated_at = last.get("created_at")
    if not updated_at:
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return {
        "thread_id": thread_id,
        "turn_count": len(records),
        "updated_at": updated_at,
        "last_question": last.get("question") or "",
        "last_answer": last.get("episode_summary") or last.get("answer_excerpt") or "",
    }


def _list_threads() -> list[dict[str, Any]]:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        path
        for path in MEMORY_DIR.glob("*.jsonl")
        if path.is_file() and not path.name.startswith(".")
    ]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [_thread_summary(path) for path in paths]


def _build_command(
    payload: dict[str, Any],
    run_json: Path,
    paper_search_log_json: Path | None = None,
) -> tuple[list[str], float]:
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("question is required")

    command = [
        sys.executable,
        str(ROOT / "scripts/ask_literature_langgraph.py"),
        question,
        "--run-json",
        str(run_json),
        "--thread-id",
        str(payload.get("thread_id") or "web_demo"),
    ]
    if paper_search_log_json is not None:
        command.extend(["--paper-search-log-json", str(paper_search_log_json)])
    _bool_flag(command, payload.get("frontier_enabled", True) is not False, "--include-frontier")
    _bool_flag(command, bool(payload.get("include_arxiv")), "--include-arxiv")
    _bool_flag(command, bool(payload.get("dry_run")), "--dry-run")
    _bool_flag(command, bool(payload.get("disable_web_search")), "--disable-web-search")
    _bool_flag(
        command,
        bool(payload.get("disable_memory_paper_tool")),
        "--disable-memory-paper-tool",
    )
    _append_value(command, "--year-from", payload.get("year_from"))
    _append_value(command, "--year-to", payload.get("year_to"))
    _append_value(command, "--paperqa-k", payload.get("paperqa_k"))
    _append_value(command, "--agent-search-count", payload.get("agent_search_count"))
    _append_value(command, "--synthesis-max-tokens", payload.get("synthesis_max_tokens"))
    _append_value(command, "--paperqa-reader-mode", payload.get("paperqa_reader_mode"))

    extra_args = str(payload.get("extra_args") or "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))

    timeout = float(payload.get("timeout") or 900)
    return command, timeout


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/threads")
    def threads():
        return jsonify({"threads": _list_threads()})

    @app.post("/api/threads")
    def new_thread():
        payload = request.get_json(force=True, silent=True) or {}
        requested = str(payload.get("thread_id") or "").strip()
        thread_id = _unique_thread_id(requested or f"research_{uuid.uuid4().hex[:8]}")
        path = _thread_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        return jsonify({"thread_id": thread_id, "threads": _list_threads()})

    @app.get("/api/threads/<thread_id>")
    def thread_detail(thread_id: str):
        safe_id = _safe_thread_id(thread_id)
        return jsonify(
            {
                "thread_id": safe_id,
                "records": _read_thread_records(safe_id),
            }
        )

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(force=True, silent=True) or {}
        run_id = uuid.uuid4().hex[:12]
        run_json = RUN_DIR / f"{run_id}.json"
        paper_search_log_json = ROOT / "data/metadata/paper_search_logs" / f"{run_id}.json"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            command, timeout = _build_command(payload, run_json, paper_search_log_json)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        started = __import__("time").monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = __import__("time").monotonic() - started
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"workflow timed out after {timeout:.0f}s",
                        "elapsed_seconds": elapsed,
                        "stdout": _process_output_text(exc.stdout),
                        "stderr": _process_output_text(exc.stderr),
                        "command": command,
                        "run_json": str(run_json),
                        "paper_search_log_json": str(paper_search_log_json),
                    }
                ),
                504,
            )
        elapsed = __import__("time").monotonic() - started

        state: dict[str, Any] = {}
        if run_json.exists():
            try:
                state = json.loads(run_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}

        ok = completed.returncode == 0
        status = 200 if ok else 500
        return (
            jsonify(
                {
                    "ok": ok,
                    "returncode": completed.returncode,
                    "elapsed_seconds": elapsed,
                    "command": command,
                    "run_json": str(run_json),
                    "paper_search_log_json": str(paper_search_log_json),
                    "state": state,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": None if ok else "workflow failed",
                }
            ),
            status,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI for the literature assistant.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
