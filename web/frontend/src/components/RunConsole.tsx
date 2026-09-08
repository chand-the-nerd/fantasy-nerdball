import { useEffect, useRef } from "react";
import type { Run } from "../lib/types";

interface Props {
  run: Run;
}

export function RunConsole({ run }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const lines = (run.log || "").split("\n").filter(Boolean);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [run.log]);

  const active = run.status === "queued" || run.status === "running";

  return (
    <div className="panel">
      <h3>
        {run.status === "queued" && "Waiting for the optimiser"}
        {run.status === "running" && `Optimising gameweek ${run.gameweek}`}
        {run.status === "failed" && "The run didn't finish"}
        {run.status === "cancelled" && "Run cancelled"}
        {run.status === "complete" && `Gameweek ${run.gameweek} optimised`}
      </h3>

      {active && (
        <div className="progress-strip">
          <i />
        </div>
      )}

      {run.status === "failed" && run.error && (
        <div className="notice bad">{run.error}</div>
      )}

      <div className="console">
        {lines.length === 0 ? (
          <p>Fetching the latest player data.</p>
        ) : (
          lines.map((line, i) => <p key={i}>{line}</p>)
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
