(function () {
  "use strict";

  var storageKey = "spikeforge.locale";
  var names = {
    en: "English", es: "Español", pt: "Português", fr: "Français", de: "Deutsch",
    it: "Italiano", nl: "Nederlands", pl: "Polski", ru: "Русский", uk: "Українська",
    tr: "Türkçe", ar: "العربية", hi: "हिन्दी", ja: "日本語", ko: "한국어",
    vi: "Tiếng Việt", id: "Bahasa Indonesia",
  };
  var rtl = ["ar"];
  var translations = window.SPIKEFORGE_TRANSLATIONS || {};
  var supported = Object.keys(names).filter(function (code) { return translations[code]; });
  var select = document.getElementById("languageSelect");
  var current = "en";

  function stored() {
    // Safari throws on localStorage reads in private mode rather than returning null.
    try { return localStorage.getItem(storageKey); } catch (error) { return null; }
  }

  function normalise(tag) {
    if (!tag) return null;
    var lower = String(tag).toLowerCase();
    var base = lower.split("-")[0];
    for (var i = 0; i < supported.length; i++) {
      if (supported[i].toLowerCase() === lower || supported[i].toLowerCase().split("-")[0] === base) {
        return supported[i];
      }
    }
    return null;
  }

  function preferredLocale() {
    var saved = normalise(stored());
    if (saved) return saved;
    // Walk the whole preference list, not just the first entry: a reader whose top choice we
    // do not ship may still have a second one that we do.
    var languages = navigator.languages || [navigator.language];
    for (var i = 0; i < languages.length; i++) {
      var match = normalise(languages[i]);
      if (match) return match;
    }
    return "en";
  }

  function message(key) {
    var messages = translations[current] || translations.en;
    return (messages && messages[key]) || (translations.en && translations.en[key]) || "";
  }

  function apply(attribute, setter) {
    document.querySelectorAll("[" + attribute + "]").forEach(function (element) {
      setter(element, message(element.getAttribute(attribute)));
    });
  }

  function translate(locale) {
    current = translations[locale] ? locale : "en";
    document.documentElement.lang = current;
    document.documentElement.dir = rtl.indexOf(current) !== -1 ? "rtl" : "ltr";

    apply("data-i18n", function (el, text) { el.textContent = text; });
    apply("data-i18n-html", function (el, text) { el.innerHTML = text; });
    apply("data-i18n-aria", function (el, text) { el.setAttribute("aria-label", text); });
    apply("data-i18n-alt", function (el, text) { el.setAttribute("alt", text); });
    apply("data-i18n-content", function (el, text) { el.setAttribute("content", text); });

    document.title = message("pageTitle");
    try { localStorage.setItem(storageKey, current); } catch (error) { /* private mode */ }
    if (select) {
      select.value = current;
      select.setAttribute("aria-label", message("language"));
    }
    document.dispatchEvent(new CustomEvent("spikeforge:localechange", { detail: { locale: current } }));
  }

  if (select) {
    supported.forEach(function (locale) {
      var option = document.createElement("option");
      option.value = locale;
      option.textContent = names[locale];
      select.appendChild(option);
    });
    select.addEventListener("change", function () { translate(select.value); });
  }

  window.SPIKEFORGE_I18N = {
    translate: translate,
    message: message,
    locale: function () { return current; },
  };

  translate(preferredLocale());
})();
