import { useState } from "react";
import { cancelJob, type JobEvent, type JobInfo } from "../api/client";

const STAGES = [
  { id: "analyze", label: "Analyze audio" },
  { id: "generate", label: "Generate notes" },
  { id: "lighting", label: "Lighting" },
  { id: "rules", label: "Playability check" },
  { id: "package", label: "Package" },
];

export function ProgressBar({ info }: { info: JobInfo }) {
  const pct = Math.round((info.progress ?? 0) * 100);
  return (
    <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

interface Props {
  info: JobInfo;
  events: JobEvent[];
  /** Songs that run before this one, while it is queued. */
  ahead?: number | null;
}

export function ProgressView({ info, events, ahead }: Props) {
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const current = STAGES.findIndex((s) => s.id === info.stage);
  const done = info.status === "done";
  const log = events.filter((e) => e.message).slice(-8);
  const waiting = ahead ? ` · ${ahead} song${ahead > 1 ? "s" : ""} ahead` : "";
  const cancel = () => {
    setCancelling(true);
    setError(null);
    cancelJob(info.id).catch((e) => {
      setError((e as Error).message);
      setCancelling(false);
    });
  };
  return (
    <div className="card progress-card">
      <div className="progress-head">
        <div>
          <div className="eyebrow">{info.status === "queued" ? `Queued${waiting}` : "Generating"}</div>
          <h2>{info.filename}</h2>
        </div>
        <div className="pct">{Math.round((info.progress ?? 0) * 100)}%</div>
      </div>
      <ProgressBar info={info} />
      <ol className="stages">
        {STAGES.map((s, i) => {
          const state = done || i < current ? "done" : i === current ? "active" : "";
          return (
            <li key={s.id} className={state}>
              <span className="dot" />
              {s.label}
            </li>
          );
        })}
      </ol>
      <ul className="log">
        {log.map((e) => (
          <li key={e.seq}>
            <span className="muted">{e.stage || e.status}</span> {e.message}
          </li>
        ))}
      </ul>
      {info.install_when_done && <div className="muted">Installs to the game when done.</div>}
      {error && <div className="error">{error}</div>}
      {(info.status === "queued" || info.status === "running") && (
        <button
          className="ghost cancel"
          disabled={cancelling}
          onClick={cancel}
          title={info.status === "running" ? "Stops at the next pipeline step" : "Take it off the queue"}
        >
          {cancelling || info.message === "Cancelling…" ? "Cancelling…" : "Cancel"}
        </button>
      )}
    </div>
  );
}
