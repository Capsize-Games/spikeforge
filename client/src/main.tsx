import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { I18nProvider } from "./i18n/I18nProvider";
import { applyStoredTheme, ThemeProvider } from "./theme";
import "./styles.css";

// Apply the saved theme before the first paint to avoid a flash.
applyStoredTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </I18nProvider>
  </StrictMode>,
);
