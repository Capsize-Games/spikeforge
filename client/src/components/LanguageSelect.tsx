import { Languages } from "lucide-react";

import { useI18n } from "../i18n/I18nProvider";
import type { Locale } from "../i18n/types";

const OPTIONS: ReadonlyArray<{ value: Locale; label: string }> = [
  { value: "en", label: "English" },
  { value: "ja", label: "日本語" },
  { value: "es", label: "Español" },
  { value: "ko", label: "한국어" },
  { value: "de", label: "Deutsch" },
  { value: "fr", label: "Français" },
  { value: "pt", label: "Português" },
  { value: "it", label: "Italiano" },
];

/** Compact, persistent locale picker shared by every dashboard panel. */
export function LanguageSelect() {
  const { locale, setLocale, t } = useI18n();
  return (
    <label className="language-select" title={t("language.label")}>
      <Languages size={15} aria-hidden="true" />
      <span className="sr-only">{t("language.label")}</span>
      <select
        value={locale}
        aria-label={t("language.label")}
        onChange={(event) => setLocale(event.target.value as Locale)}
      >
        {OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
