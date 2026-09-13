import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { EN, TRANSLATIONS } from "./translations";
import type { TranslationKey } from "./translations";
import type { Locale } from "./types";

const STORAGE_KEY = "spikeforge.locale";
const SUPPORTED = new Set<Locale>([
  "en",
  "ja",
  "es",
  "ko",
  "de",
  "fr",
  "pt",
  "it",
]);

interface I18nValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey) => string;
}

const I18nContext = createContext<I18nValue | null>(null);

function isLocale(value: string | null): value is Locale {
  return value !== null && SUPPORTED.has(value as Locale);
}

function initialLocale(): Locale {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (isLocale(stored)) return stored;
  const browserLocale = navigator.language.toLowerCase().split("-")[0] ?? "en";
  return isLocale(browserLocale) ? browserLocale : "en";
}

interface Props {
  children: ReactNode;
}

export function I18nProvider({ children }: Props) {
  const [locale, setLocale] = useState<Locale>(initialLocale);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, locale);
    document.documentElement.lang = locale;
  }, [locale]);

  const value = useMemo<I18nValue>(
    () => ({
      locale,
      setLocale,
      t: (key) => TRANSLATIONS[locale][key] ?? EN[key],
    }),
    [locale],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nValue {
  const value = useContext(I18nContext);
  if (value === null) {
    throw new Error("useI18n must be used inside I18nProvider");
  }
  return value;
}
