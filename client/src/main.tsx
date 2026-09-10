import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { applyStoredTheme, ThemeProvider } from "./theme";
import "./styles.css";

// Apply the saved theme before the first paint to avoid a flash.
applyStoredTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
);
