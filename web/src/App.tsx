import { useEffect, useState } from "react";
import { TERMINAL, getSettings, jobName, type ServerSettings } from "./api/client";
import { Preview } from "./components/Preview";
import { ProgressView } from "./components/ProgressView";
import { QueuePanel } from "./components/QueuePanel";
import { UploadPanel } from "./components/UploadPanel";
import { useJob } from "./useJob";
import { jobsAhead, useQueue } from "./useQueue";

// The current page lives in the URL hash so a reload stays on it: #job=<id> for a song,
// #remake=<id> to make a song again with other options. The queue lives on the server.
function readHash(): { job: string | null; remake: string | null } {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return { job: params.get("job"), remake: params.get("remake") };
}

export function App() {
  const [route, setRoute] = useState(readHash);
  const jobId = route.job;
  const { info, events, map, error, regenerate, reload } = useJob(jobId);
  const queue = useQueue();
  const [settings, setSettings] = useState<ServerSettings | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    getSettings().then(setSettings, () => setSettings(null));
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (hash: string) => {
    window.location.hash = hash;
    setRoute(readHash());
    window.scrollTo({ top: 0 });
  };
  const open = (id: string | null) => go(id ? `job=${id}` : "");
  const remake = (id: string) => go(`remake=${id}`);

  const back = queue.jobs.length > 0 ? "Back to the queue" : "Try another song";
  const active = queue.jobs.filter((j) => !TERMINAL.includes(j.status)).length;
  const running = queue.jobs.find((j) => j.status === "running");

  // The tab title shows the queue while you do something else.
  useEffect(() => {
    if (running) document.title = `${Math.round((running.progress ?? 0) * 100)}% · ${active} left · bsmapper`;
    else if (active) document.title = `${active} waiting · bsmapper`;
    else document.title = "bsmapper";
  }, [running?.progress, running?.id, active]);

  const remakeJob = route.remake ? queue.jobs.find((j) => j.id === route.remake) : null;

  let body;
  if (!jobId) {
    body = (
      <div className="home">
        {queue.jobs.length > 0 && !route.remake && (
          <QueuePanel {...queue} open={open} remake={remake} />
        )}
        {route.remake && !remakeJob ? (
          <div className="card center muted">{queue.jobs.length ? "That song is no longer on the server." : "Loading…"}</div>
        ) : (
          <UploadPanel
            key={route.remake ?? "new"}
            settings={settings}
            remake={remakeJob}
            onCancelRemake={() => open(route.remake)}
            onQueued={(jobs) => {
              queue.refresh();
              // One song: follow it as before. A batch: stay here and watch the queue.
              if (jobs.length === 1) open(jobs[0].id);
              else window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}
      </div>
    );
  } else if (map && info) {
    body = <Preview info={info} map={map} regenerate={regenerate} reload={reload} remake={() => remake(info.id)} />;
  } else if (info?.status === "failed" || info?.status === "cancelled") {
    const failed = info.status === "failed";
    body = (
      <div className="card center">
        <h2>{failed ? "Generation failed" : "Cancelled"}</h2>
        <p className={failed ? "error" : "muted"}>{failed ? info.error : jobName(info)}</p>
        <div className="button-row">
          <button className="button primary" onClick={() => remake(info.id)} title="Same song, change options if you like">
            {failed ? "Try again…" : "Make it after all…"}
          </button>
          <button className="button" onClick={() => open(null)}>
            {back}
          </button>
        </div>
      </div>
    );
  } else if (info) {
    body = <ProgressView info={info} events={events} ahead={jobsAhead(queue.jobs, info.id)} />;
  } else if (error) {
    body = (
      <div className="card center">
        <h2>Job not found</h2>
        <p className="error">{error}</p>
        <p className="muted">It may have been cleared from the queue.</p>
        <button className="button primary" onClick={() => open(null)}>
          Start over
        </button>
      </div>
    );
  } else {
    body = <div className="card center muted">Loading…</div>;
  }

  return (
    <div className="app">
      <header>
        <a className="brand" href="#" onClick={() => open(null)}>
          <span className="cube red" />
          <span className="cube blue" />
          bsmapper
        </a>
        {settings && settings.generator !== "model" && settings.generator !== "not loaded" && (
          <span className="badge" title={settings.generator_detail}>
            notes: {settings.generator}
          </span>
        )}
        {(jobId || route.remake) && (
          <button className="ghost new" onClick={() => open(null)}>
            {queue.jobs.length > 1 ? `Queue${active ? ` (${active} left)` : ""} · new songs` : "New song"}
          </button>
        )}
      </header>
      {settings?.generator === "stub" && (
        <div className="banner warn">
          Notes come from the placeholder stub generator, not the trained model
          {settings.generator_detail ? ` (${settings.generator_detail})` : ""}. Check that the checkpoint is in{" "}
          <code>checkpoints\</code> and restart the server.
        </div>
      )}
      {error && map && <div className="banner error">{error}</div>}
      <main>{body}</main>
    </div>
  );
}
