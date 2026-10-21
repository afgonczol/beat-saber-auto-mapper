import type { JobEvent, JobInfo } from "../api/client";

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

export function ProgressView({ info, events }: { info: JobInfo; events: JobEvent[] }) {
  const current = STAGES.findIndex((s) => s.id === info.stage);
  const done = info.status === "done";
  const log = events.filter((e) => e.message).slice(-8);
  return (
    <div className="card progress-card">
      <div className="progress-head">
        <div>
          <div className="eyebrow">{info.status === "queued" ? "Queued" : "Generating"}</div>
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
    </div>
  );
}
