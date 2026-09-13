(function () {
  "use strict";

  var storageKey = "spikeforge.locale";
  var supported = ["en", "ja", "es", "ko", "de", "fr", "pt", "it"];
  var names = {
    en: "English",
    ja: "日本語",
    es: "Español",
    ko: "한국어",
    de: "Deutsch",
    fr: "Français",
    pt: "Português",
    it: "Italiano",
  };
  var translations = window.SPIKEFORGE_TRANSLATIONS;
  var select = document.getElementById("languageSelect");

  function preferredLocale() {
    var stored = localStorage.getItem(storageKey);
    if (supported.indexOf(stored) !== -1) return stored;
    var browserLocale = navigator.language.toLowerCase().split("-")[0];
    return supported.indexOf(browserLocale) !== -1 ? browserLocale : "en";
  }

  function translate(locale) {
    var messages = translations[locale] || translations.en;
    document.documentElement.lang = locale;
    document.querySelectorAll("[data-i18n]").forEach(function (element) {
      var key = element.getAttribute("data-i18n");
      element.textContent = messages[key] || translations.en[key];
    });
    document.querySelectorAll("[data-i18n-html]").forEach(function (element) {
      var key = element.getAttribute("data-i18n-html");
      element.innerHTML = messages[key] || translations.en[key];
    });
    document.querySelectorAll("[data-i18n-aria]").forEach(function (element) {
      var key = element.getAttribute("data-i18n-aria");
      element.setAttribute("aria-label", messages[key] || translations.en[key]);
    });
    document.querySelectorAll("[data-i18n-alt]").forEach(function (element) {
      var key = element.getAttribute("data-i18n-alt");
      element.setAttribute("alt", messages[key] || translations.en[key]);
    });
    document.querySelectorAll("[data-i18n-content]").forEach(function (element) {
      var key = element.getAttribute("data-i18n-content");
      element.setAttribute("content", messages[key] || translations.en[key]);
    });
    document.title = messages.pageTitle || translations.en.pageTitle;
    localStorage.setItem(storageKey, locale);
    select.value = locale;
    select.setAttribute("aria-label", messages.language);
  }

  supported.forEach(function (locale) {
    var option = document.createElement("option");
    option.value = locale;
    option.textContent = names[locale];
    select.appendChild(option);
  });
  select.addEventListener("change", function () {
    translate(select.value);
  });
  translate(preferredLocale());
})();
