import { useState, type ReactNode } from "react";
import {
  DIFFICULTY_LABEL,
  MODE_LABEL,
  TERMINAL,
  cancelJob,
  downloadUrl,
  removeJob,
  type JobInfo,
} from "../api/client";
import { ProgressBar } from "./ProgressView";

const STATUS_LABEL: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

const STAGE_LABEL: Record<string, string> = {
  analyze: "Analysing audio",
  generate: "Generating notes",
  lighting: "Lighting",
  rules: "Playability check",
  package: "Packaging",
};

function optionsSummary(job: JobInfo): string {
  const o = job.options;
  const modes = (o.modes ?? []).map((m) => MODE_LABEL[m] ?? m).join(", ");
  const diffs = (o.difficulties ?? []).map((d) => DIFFICULTY_LABEL[d] ?? d).join(", ");
  return `${modes} · ${diffs}`;
}

interface Props {
  jobs: JobInfo[];
  error: string | null;
  refresh: () => void;
  open: (id: string) => void;
}

/** Every song on the server: queued, running, finished. Lasts until the server restarts. */
export function QueuePanel({ jobs, error, refresh, open }: Props) {
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [actionError, setActionError] = useState<string | null>(null);

  const act = async (ids: string[], fn: (id: string) => Promise<unknown>) => {
    setActionError(null);
    setBusy((b) => new Set([...b, ...ids]));
    try {
      for (const id of ids) await fn(id);
    } catch (e) {
      setActionError((e as Error).message);
    } finally {
      setBusy((b) => new Set([...b].filter((id) => !ids.includes(id))));
      refresh();
    }
  };

  const queued = jobs.filter((j) => j.status === "queued");
  const finished = jobs.filter((j) => TERMINAL.includes(j.status));
  const running = jobs.filter((j) => j.status === "running");
  const done = jobs.filter((j) => j.status === "done").length;
  const failed = jobs.filter((j) => j.status === "failed").length;

  return (
    <section className="card queue" aria-label="Song queue">
      <div className="queue-head">
        <div>
          <h2>Queue</h2>
          <div className="muted">
            {running.length > 0 ? "1 running" : "Idle"}
            {queued.length > 0 && ` · ${queued.length} waiting`}
            {done > 0 && ` · ${done} done`}
            {failed > 0 && ` · ${failed} failed`}
          </div>
        </div>
        <div className="queue-bulk">
          {queued.length > 1 && (
            <button
              className="ghost small"
              onClick={() => act(queued.map((j) => j.id), cancelJob)}
              title="Cancel every song that has not started yet"
            >
              Cancel waiting
            </button>
          )}
          {finished.length > 0 && (
            <button
              className="ghost small"
              onClick={() => act(finished.map((j) => j.id), removeJob)}
              title="Take finished, failed and cancelled songs off the list (installed maps stay in the game)"
            >
              Clear finished
            </button>
          )}
        </div>
      </div>
      {(error || actionError) && <div className="error">{actionError ?? error}</div>}
      <ol className="queue-list">
        {jobs.map((job) => (
          <QueueRow
            key={job.id}
            job={job}
            busy={busy.has(job.id)}
            open={open}
            cancel={() => act([job.id], cancelJob)}
            remove={() => act([job.id], removeJob)}
          />
        ))}
      </ol>
    </section>
  );
}

function QueueRow({
  job,
  busy,
  open,
  cancel,
  remove,
}: {
  job: JobInfo;
  busy: boolean;
  open: (id: string) => void;
  cancel: () => void;
  remove: () => void;
}) {
  const status = job.status;
  const pct = Math.round((job.progress ?? 0) * 100);
  let detail: ReactNode = optionsSummary(job);
  if (status === "running") {
    detail = `${STAGE_LABEL[job.stage ?? ""] ?? "Starting"} · ${pct}%${job.message ? ` · ${job.message}` : ""}`;
  } else if (status === "failed") {
    detail = <span className="error">{job.error || job.message}</span>;
  }
  return (
    <li className={`queue-row ${status}`}>
      <div className="queue-main">
        <div className="queue-title">
          <span className={`pill ${status}`}>{STATUS_LABEL[status] ?? status}</span>
          <a
            className="queue-name"
            href={`#job=${job.id}`}
            onClick={(e) => {
              e.preventDefault();
              open(job.id);
            }}
            title={status === "done" ? "Open the preview" : "Open its progress page"}
          >
            {job.filename}
          </a>
        </div>
        <div className="queue-detail muted">{detail}</div>
        {status === "running" && <ProgressBar info={job} />}
        {job.installed && (
          <div className="queue-detail ok" title={job.installed.path}>
            Installed to the game
          </div>
        )}
        {job.install_error && <div className="queue-detail warn">Install failed: {job.install_error}</div>}
        {!job.installed && !job.install_error && job.install_when_done && !TERMINAL.includes(status) && (
          <div className="queue-detail muted">Installs to the game when done</div>
        )}
      </div>
      <div className="queue-actions">
        {status === "done" && (
          <>
            <button className="button small" onClick={() => open(job.id)}>
              Open
            </button>
            <a className="button small" href={downloadUrl(job.id)} download>
              .zip
            </a>
          </>
        )}
        {status === "running" && (
          <button className="button small" disabled={busy} onClick={cancel} title="Stops at the next pipeline step">
            {busy || job.message === "Cancelling…" ? "Cancelling…" : "Cancel"}
          </button>
        )}
        {status !== "running" && (
          <button
            className="ghost small"
            disabled={busy}
            onClick={remove}
            title={status === "queued" ? "Take it off the queue" : "Take it off the list"}
            aria-label={`Remove ${job.filename}`}
          >
            ✕
          </button>
        )}
      </div>
    </li>
  );
}
