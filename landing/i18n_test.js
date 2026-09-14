#!/usr/bin/env node
/* Runtime tests for landing/i18n.js.
 *
 * The site has no build step and no node_modules, so this stands up just enough of a DOM by
 * hand -- elements, a localStorage, a navigator -- to exercise locale detection, the binding
 * attributes, RTL, and the switcher. Run it with: node landing/i18n_test.js
 */
// Minimal DOM shim: enough surface for i18n.js, no jsdom dependency.
const path = require("path");
const BASE = __dirname;
const LOCALES = path.join(BASE, "locales.js");
const CTA = path.join(BASE, "cta-locales.js");
const I18N = path.join(BASE, "i18n.js");

function makeEl(tag, attrs = {}) {
  return {
    tagName: tag, attrs: { ...attrs }, children: [], textContent: "", innerHTML: "", value: "",
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    setAttribute(k, v) { this.attrs[k] = v; },
    appendChild(c) { this.children.push(c); },
    addEventListener() {},
  };
}

function run({ languages, stored, storageThrows = false }) {
  const els = {
    nav: makeEl("a", { "data-i18n": "features" }),
    head: makeEl("h2", { "data-i18n-html": "scope" }),
    aria: makeEl("a", { "data-i18n-aria": "language" }),
    meta: makeEl("meta", { "data-i18n-content": "metaDescription" }),
    missing: makeEl("p", { "data-i18n": "noSuchKey" }),
  };
  const select = makeEl("select", { id: "languageSelect" });
  const root = makeEl("html", {});
  const store = {};

  global.window = {};
  // Node 22 ships a read-only built-in `navigator`; a plain assignment is silently dropped.
  Object.defineProperty(globalThis, "navigator", { value: { languages }, configurable: true, writable: true });
  global.localStorage = {
    getItem(k) { if (storageThrows) throw new Error("private mode"); return k in store ? store[k] : (stored ?? null); },
    setItem(k, v) { if (storageThrows) throw new Error("private mode"); store[k] = v; },
  };
  global.CustomEvent = function (name, init) { return { name, ...init }; };
  global.document = {
    documentElement: root,
    title: "",
    getElementById: (id) => (id === "languageSelect" ? select : null),
    querySelectorAll(sel) {
      const attr = sel.slice(1, -1);
      return Object.values(els).filter((e) => attr in e.attrs);
    },
    createElement: (tag) => makeEl(tag),
    dispatchEvent() {},
  };

  delete require.cache[require.resolve(LOCALES)];
  require(LOCALES);
  delete require.cache[require.resolve(CTA)];
  require(CTA);
  delete require.cache[require.resolve(I18N)];
  require(I18N);
  return { els, select, root, api: global.window.SPIKEFORGE_I18N, store };
}

let failures = 0;
function check(name, actual, expected) {
  const pass = actual === expected;
  if (!pass) failures++;
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${pass ? "" : `\n        got:      ${actual}\n        expected: ${expected}`}`);
}

// 1. Exact browser-language match.
let r = run({ languages: ["ja-JP", "en"] });
check("ja-JP -> ja", r.api.locale(), "ja");
check("ja nav text", r.els.nav.textContent, "機能");
check("ja title", global.document.title.startsWith("Spikeforge"), true);

// 2. Region subtag falls back to the base language.
r = run({ languages: ["pt-BR"] });
check("pt-BR -> pt", r.api.locale(), "pt");

// 3. Unsupported language falls through to the next preference.
r = run({ languages: ["zh-CN", "de-AT"] });
check("zh-CN unsupported, de-AT -> de", r.api.locale(), "de");

// 4. Nothing supported -> English.
r = run({ languages: ["sw-KE"] });
check("no match -> en", r.api.locale(), "en");

// 5. Stored preference wins over the browser.
r = run({ languages: ["fr-FR"], stored: "ko" });
check("stored ko beats fr-FR", r.api.locale(), "ko");

// 6. Arabic flips direction; every other locale is ltr.
r = run({ languages: ["ar"] });
check("ar dir", r.root.dir, "rtl");
check("ar lang", r.root.lang, "ar");
r = run({ languages: ["he-IL", "en"] });
check("en dir", r.root.dir, "ltr");

// 7. Missing key falls back to English rather than rendering empty.
r = run({ languages: ["ru"] });
check("unknown key -> empty string", r.els.missing.textContent, "");
check("ru html binding has <br>", r.els.head.innerHTML.includes("<br>β=0,90"), true);
check("ru aria binding", r.els.aria.getAttribute("aria-label"), "Язык");
check("ru meta content binding", r.els.meta.getAttribute("content").startsWith("Создавайте"), true);

// 8. Switcher is populated with every locale that has a catalogue.
r = run({ languages: ["en"] });
check("switcher option count", r.select.children.length, 17);
check("switcher shows native names", r.select.children.map((o) => o.textContent).includes("Українська"), true);
check("switcher value tracks locale", r.select.value, "en");

// 9. Private-mode storage must not throw.
try {
  r = run({ languages: ["it"], storageThrows: true });
  check("storage throws -> still resolves", r.api.locale(), "it");
} catch (e) {
  failures++;
  console.log("FAIL  storage throws -> uncaught:", e.message);
}

console.log(failures ? `\n${failures} failure(s)` : "\nall checks passed");
process.exit(failures ? 1 : 0);
