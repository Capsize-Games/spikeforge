import type { StatusPayload } from "../types";

interface Props {
  connected: boolean;
  status: StatusPayload | null;
}

export function TopBar({ connected, status }: Props) {
  return (
    <header className="topbar">
      <h1>SNN Interpreter</h1>
      <span className={`dot ${connected ? "ok" : "bad"}`} />
      <span>{connected ? "connected" : "disconnected"}</span>
      {status && (
        <span className="status">
          {status.coding} · {status.num_steps} steps
          {status.target !== null && status.target !== undefined
            ? ` · target ${status.target}`
            : ""}
        </span>
      )}
    </header>
  );
}
