import { useEffect, useState } from "react";
import { TERMINAL } from "./api/client";
import { Preview } from "./components/Preview";
import { ProgressView } from "./components/ProgressView";
import { QueuePanel } from "./components/QueuePanel";
import { UploadPanel } from "./components/UploadPanel";
import { useJob } from "./useJob";
import { jobsAhead, useQueue } from "./useQueue";

// The current job lives in the URL hash (#job=<id>) so a reload resumes it. The queue lives on
// the server, so a reload finds it again too.
function readHash(): string | null {
  return new URLSearchParams(window.location.hash.slice(1)).get("job");
}

export function App() {
  const [jobId, setJobId] = useState<string | null>(readHash);
  const { info, events, map, error, regenerate } = useJob(jobId);
  const queue = useQueue();

  useEffect(() => {
    const onHash = () => setJobId(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const open = (id: string | null) => {
    window.location.hash = id ? `job=${id}` : "";
    setJobId(id);
  };

  const back = queue.jobs.length > 0 ? "Back to the queue" : "Try another song";
  const active = queue.jobs.filter((j) => !TERMINAL.includes(j.status)).length;

  let body;
  if (!jobId) {
    body = (
      <div className="home">
        {queue.jobs.length > 0 && <QueuePanel {...queue} open={open} />}
        <UploadPanel
          onQueued={(jobs) => {
            queue.refresh();
            // One song: follow it as before. A batch: stay here and watch the queue.
            if (jobs.length === 1) open(jobs[0].id);
            else window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
      </div>
    );
  } else if (map && info) {
    body = <Preview info={info} map={map} regenerate={regenerate} />;
  } else if (info?.status === "failed" || info?.status === "cancelled") {
    const failed = info.status === "failed";
    body = (
      <div className="card center">
        <h2>{failed ? "Generation failed" : "Cancelled"}</h2>
        <p className={failed ? "error" : "muted"}>{failed ? info.error : info.filename}</p>
        <button className="button primary" onClick={() => open(null)}>
          {back}
        </button>
      </div>
    );
  } else if (info) {
    body = <ProgressView info={info} events={events} ahead={jobsAhead(queue.jobs, info.id)} />;
  } else if (error) {
    body = (
      <div className="card center">
        <h2>Job not found</h2>
        <p className="error">{error}</p>
        <p className="muted">Jobs are kept until the server restarts.</p>
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
        {jobId && (
          <button className="ghost new" onClick={() => open(null)}>
            {queue.jobs.length > 1 ? `Queue${active ? ` (${active} left)` : ""} · new songs` : "New song"}
          </button>
        )}
      </header>
      {error && map && <div className="banner error">{error}</div>}
      <main>{body}</main>
    </div>
  );
}
