import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyStoredTheme } from "./lib/settingsStore";
import "./styles.css";

// Before the first render: the account is the source of truth, but it takes a
// request to fetch, and a flash of the wrong palette is worse than a stale one.
applyStoredTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
