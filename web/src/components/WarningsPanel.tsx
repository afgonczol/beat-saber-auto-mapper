import type { DifficultyWarning } from "../api/client";
import { formatTime, type TempoMap } from "../preview/timing";

interface Props {
  warnings: DifficultyWarning[];
  notes: string[];
  tempo: TempoMap;
  onSeek: (seconds: number) => void;
}

/** Playability issues for the current difficulty; click one to jump just before it. */
export function WarningsPanel({ warnings, notes, tempo, onSeek }: Props) {
  const sorted = [...warnings].sort((a, b) => a.issue.beat - b.issue.beat);
  const errors = sorted.filter((w) => w.issue.severity === "error").length;
  return (
    <div className="card warnings">
      <h3>
        Warnings
        <span className={`count ${errors ? "error" : sorted.length ? "warning" : "ok"}`}>
          {sorted.length ? `${sorted.length}${errors ? ` · ${errors} error${errors > 1 ? "s" : ""}` : ""}` : "none"}
        </span>
      </h3>
      {sorted.length === 0 && <div className="muted">No playability issues for this difficulty.</div>}
      <ul>
        {sorted.map((w, i) => {
          const t = tempo.beatToSeconds(w.issue.beat);
          return (
            <li key={i}>
              <button className={`warning ${w.issue.severity}`} onClick={() => onSeek(t - 1.5)}>
                <span className="when">{formatTime(t)}</span>
                <span className="rule">{w.issue.rule.replace(/_/g, " ")}</span>
                <span className="msg">{w.issue.message}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {notes.length > 0 && (
        <ul className="job-notes">
          {notes.map((n, i) => (
            <li key={i} className="muted">
              {n}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
