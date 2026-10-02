#!/usr/bin/env node
/**
 * Review a pull request with an OpenAI-compatible model and post the result.
 *
 * Written against the chat-completions shape rather than any one vendor's SDK,
 * so OpenRouter, DeepInfra, or anything else speaking that protocol works by
 * changing two variables. Nothing here is specific to a model.
 *
 * Configuration, all from the environment:
 *
 *   LLM_API_KEY    the provider key (secret)
 *   LLM_BASE_URL   e.g. https://openrouter.ai/api/v1
 *   LLM_MODEL      the provider's exact model slug
 *   GH_TOKEN       token used to read the diff and post the comment
 *   GITHUB_REPOSITORY, PR_NUMBER
 *
 * The model slug is deliberately not defaulted. A wrong default fails at the
 * provider with an opaque error on somebody else's pull request, and guessing
 * one is worse than refusing to start.
 */

import { readFileSync } from "node:fs";

/** Diffs beyond this are truncated; large ones exhaust the context window. */
const MAX_DIFF_BYTES = 120_000;

const REQUIRED = ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "GH_TOKEN"];

export function missingConfig(env) {
  return REQUIRED.filter((name) => !env[name]);
}

/**
 * Cut a diff down to a size the model can read, at a file boundary.
 *
 * Truncating mid-hunk hands the model a fragment it may read as the whole
 * change, so the cut lands on the last `diff --git` before the limit and the
 * omission is stated in the text.
 */
export function truncateDiff(diff, limit = MAX_DIFF_BYTES) {
  if (diff.length <= limit) return { diff, truncated: false };
  const head = diff.slice(0, limit);
  const lastFile = head.lastIndexOf("\ndiff --git ");
  const cut = lastFile > 0 ? head.slice(0, lastFile) : head;
  return { diff: cut, truncated: true };
}

/** The instructions the model is judged against. */
export function buildPrompt({ number, title, body, diff, truncated, guide }) {
  const omitted = truncated
    ? "\n\nThis diff was truncated. Review only what is shown, and say so."
    : "";
  return [
    `Review pull request #${number}: ${title}`,
    body ? `\nIts description:\n${body}` : "",
    guide ? `\nThe repository's own guidance is binding:\n${guide}` : "",
    "\nReport only defects you can point at in the diff. Say plainly when",
    "you find none; a review that invents work to look useful is worse than",
    "a short one. Rank what you do find, most serious first.",
    "\nWeigh hardest the things this repository has shipped broken:",
    "- a new file under desktop/ missing from the files: list in",
    "  electron-builder.yml, which silently will not be packaged;",
    "- a check that passes while the product is broken, such as asserting a",
    "  server answered rather than that the application works;",
    "- a startup path that can leave the application running with no window;",
    "- public-facing copy stating something the code does not do;",
    "- the hard limits in the guidance: file and function length, `any`,",
    "  non-null assertions, suppression comments, dead code.",
    "\nWrite GitHub-flavoured markdown. Do not approve or request changes;",
    "the required checks decide whether this merges.",
    `\nThe diff:\n\n${diff}${omitted}`,
  ].join("\n");
}

/** Pull the assistant's text out of a chat-completions response. */
export function extractMessage(payload) {
  const text = payload?.choices?.[0]?.message?.content;
  if (typeof text !== "string" || text.trim() === "") {
    throw new Error(
      `the model returned no message: ${JSON.stringify(payload).slice(0, 400)}`,
    );
  }
  return text.trim();
}

async function github(path, { token, accept, method, body }) {
  const response = await fetch(`https://api.github.com${path}`, {
    method: method ?? "GET",
    headers: {
      Accept: accept ?? "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    throw new Error(
      `GitHub ${method ?? "GET"} ${path}: ${response.status} ` +
        `${await response.text()}`,
    );
  }
  return accept?.includes("diff") ? response.text() : response.json();
}

async function complete({ baseUrl, key, model, prompt }) {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.2,
    }),
  });
  if (!response.ok) {
    throw new Error(
      `${baseUrl}: ${response.status} ${await response.text()}`,
    );
  }
  return extractMessage(await response.json());
}

/** The repository's guidance, when it has some, as prompt context. */
function readGuide() {
  for (const name of ["AGENTS.md", "CLAUDE.md", "rules.md"]) {
    try {
      return readFileSync(name, "utf8");
    } catch {
      // Try the next one; a repository without any is fine.
    }
  }
  return "";
}

async function main() {
  const env = process.env;
  const missing = missingConfig(env);
  if (missing.length > 0) {
    console.error(`missing configuration: ${missing.join(", ")}`);
    process.exit(78);
  }
  const repo = env.GITHUB_REPOSITORY;
  const number = env.PR_NUMBER;
  const token = env.GH_TOKEN;

  const pull = await github(`/repos/${repo}/pulls/${number}`, { token });
  const raw = await github(`/repos/${repo}/pulls/${number}`, {
    token,
    accept: "application/vnd.github.v3.diff",
  });
  const { diff, truncated } = truncateDiff(raw);

  const review = await complete({
    baseUrl: env.LLM_BASE_URL,
    key: env.LLM_API_KEY,
    model: env.LLM_MODEL,
    prompt: buildPrompt({
      number,
      title: pull.title,
      body: pull.body,
      diff,
      truncated,
      guide: readGuide(),
    }),
  });

  await github(`/repos/${repo}/issues/${number}/comments`, {
    token,
    method: "POST",
    body: {
      body: `${review}\n\n<sub>Automated review — ${env.LLM_MODEL}. Not an ` +
        `approval; the required checks decide whether this merges.</sub>`,
    },
  });
  console.log(`posted a review on #${number}`);
}

if (process.argv[1] && process.argv[1].endsWith("llm_review.mjs")) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
