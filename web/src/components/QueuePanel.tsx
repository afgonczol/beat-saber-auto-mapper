import { useState, type ReactNode } from "react";
import {
  DIFFICULTY_LABEL,
  MODE_LABEL,
  TERMINAL,
  cancelJob,
  downloadUrl,
  installJob,
  jobName,
  removeJob,
  rerunJob,
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

const MAX_RUN_S = 30 * 60; // longer runs were interrupted or waited on something; skip them

/** Seconds a whole song takes here: the median of the finished runs, null before the first. */
export function typicalRunSeconds(jobs: JobInfo[]): number | null {
  const runs = jobs
    .filter((j) => j.status === "done" && j.revision === 1 && j.started_at && j.finished_at)
    .map((j) => (Date.parse(j.finished_at!) - Date.parse(j.started_at!)) / 1000)
    .filter((s) => s > 0 && s < MAX_RUN_S)
    .sort((a, b) => a - b);
  return runs.length ? runs[Math.floor(runs.length / 2)] : null;
}

/** Rough time until the queue is empty, e.g. "about 4 min left". */
export function queueEta(jobs: JobInfo[]): string | null {
  const run = typicalRunSeconds(jobs);
  if (run == null) return null;
  const running = jobs.find((j) => j.status === "running");
  const waiting = jobs.filter((j) => j.status === "queued").length;
  const left = waiting * run + (running ? run * (1 - (running.progress ?? 0)) : 0);
  if (left <= 0) return null;
  return left < 60 ? "under a minute left" : `about ${Math.round(left / 60)} min left`;
}

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
  /** Make a song again with other options. */
  remake: (id: string) => void;
}

/** Every song on the server: queued, running, finished. Lasts until the server restarts. */
export function QueuePanel({ jobs, error, refresh, open, remake }: Props) {
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
  const failedJobs = jobs.filter((j) => j.status === "failed");
  const failed = failedJobs.length;
  const notInstalled = jobs.filter((j) => j.status === "done" && !j.installed);
  const eta = queueEta(jobs);
  // A retry queues the song again with the same options and drops the failed entry.
  const retry = (id: string) => rerunJob(id).then(() => removeJob(id));

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
            {eta && ` · ${eta}`}
          </div>
        </div>
        <div className="queue-bulk">
          {failed > 0 && (
            <button
              className="ghost small"
              onClick={() => act(failedJobs.map((j) => j.id), retry)}
              title="Queue every failed song again with the same options"
            >
              Retry failed
            </button>
          )}
          {notInstalled.length > 1 && (
            <button
              className="ghost small"
              onClick={() => act(notInstalled.map((j) => j.id), installJob)}
              title="Copy every finished map that is not in the game yet into CustomLevels"
            >
              Install all ({notInstalled.length})
            </button>
          )}
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
            retry={() => act([job.id], retry)}
            install={() => act([job.id], installJob)}
            remake={() => remake(job.id)}
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
  retry,
  install,
  remake,
}: {
  job: JobInfo;
  busy: boolean;
  open: (id: string) => void;
  cancel: () => void;
  remove: () => void;
  retry: () => void;
  install: () => void;
  remake: () => void;
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
            title={`${job.filename} · ${status === "done" ? "open the preview" : "open its progress page"}`}
          >
            {jobName(job)}
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
            {!job.installed && (
              <button className="button small" disabled={busy} onClick={install} title="Copy the map into CustomLevels">
                Install
              </button>
            )}
            <button className="ghost small" onClick={remake} title="Make this song again with other options or a new seed">
              Again…
            </button>
          </>
        )}
        {(status === "failed" || status === "cancelled") && (
          <>
            <button className="button small" disabled={busy} onClick={retry} title="Queue it again with the same options">
              Retry
            </button>
            <button className="ghost small" onClick={remake} title="Queue it again with other options">
              Edit…
            </button>
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
