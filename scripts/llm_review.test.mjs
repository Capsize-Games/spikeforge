import assert from "node:assert/strict";
import test from "node:test";

import {
  buildPrompt,
  extractMessage,
  missingConfig,
  truncateDiff,
} from "./llm_review.mjs";

test("names every missing setting at once", () => {
  assert.deepEqual(missingConfig({}), [
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "GH_TOKEN",
  ]);
  assert.deepEqual(
    missingConfig({
      LLM_API_KEY: "k",
      LLM_BASE_URL: "u",
      LLM_MODEL: "m",
      GH_TOKEN: "t",
    }),
    [],
  );
});

/**
 * An empty model slug must count as missing. Passing "" to the provider is a
 * 400 on somebody else's pull request, reported as a review failure rather
 * than as the configuration mistake it is.
 */
test("treats an empty setting as missing", () => {
  const missing = missingConfig({
    LLM_API_KEY: "k",
    LLM_BASE_URL: "u",
    LLM_MODEL: "",
    GH_TOKEN: "t",
  });
  assert.deepEqual(missing, ["LLM_MODEL"]);
});

test("leaves a diff that fits alone", () => {
  const diff = "diff --git a/a b/a\n+one\n";
  assert.deepEqual(truncateDiff(diff, 1000), { diff, truncated: false });
});

/**
 * The cut has to land on a file boundary. Handing the model half a hunk
 * invites it to review a fragment as though it were the whole change.
 */
test("truncates a long diff at a file boundary", () => {
  const first = `diff --git a/a b/a\n${"+x\n".repeat(50)}`;
  const second = `diff --git a/b b/b\n${"+y\n".repeat(50)}`;
  const { diff, truncated } = truncateDiff(first + second, first.length + 20);
  assert.equal(truncated, true);
  assert.equal(diff, first.replace(/\n$/, ""));
  assert.ok(!diff.includes("b/b"));
});

test("still truncates when no boundary is reachable", () => {
  const huge = `diff --git a/a b/a\n${"+x\n".repeat(200)}`;
  const { diff, truncated } = truncateDiff(huge, 50);
  assert.equal(truncated, true);
  assert.equal(diff.length, 50);
});

test("tells the model when it is seeing a partial diff", () => {
  const whole = buildPrompt({ number: 1, title: "t", diff: "d" });
  assert.doesNotMatch(whole, /truncated/);
  const part = buildPrompt({ number: 1, title: "t", diff: "d", truncated: true });
  assert.match(part, /truncated/);
});

test("carries the repository's guidance into the prompt", () => {
  const prompt = buildPrompt({
    number: 7,
    title: "Add a thing",
    diff: "d",
    guide: "NEVER use any",
  });
  assert.match(prompt, /#7: Add a thing/);
  assert.match(prompt, /NEVER use any/);
});

test("reads the assistant message", () => {
  const payload = { choices: [{ message: { content: "  looks fine  " } }] };
  assert.equal(extractMessage(payload), "looks fine");
});

/**
 * A provider that answers 200 with an error body, or with an empty message,
 * must not post an empty review that reads like approval.
 */
test("refuses an empty or malformed completion", () => {
  assert.throws(() => extractMessage({}), /no message/);
  assert.throws(
    () => extractMessage({ choices: [{ message: { content: "   " } }] }),
    /no message/,
  );
  assert.throws(
    () => extractMessage({ error: { message: "bad model" } }),
    /no message/,
  );
});
