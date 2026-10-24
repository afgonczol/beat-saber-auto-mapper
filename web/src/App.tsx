import { useEffect, useState } from "react";
import { Preview } from "./components/Preview";
import { ProgressView } from "./components/ProgressView";
import { UploadPanel } from "./components/UploadPanel";
import { useJob } from "./useJob";

// The current job lives in the URL hash (#job=<id>) so a reload resumes it.
function readHash(): string | null {
  return new URLSearchParams(window.location.hash.slice(1)).get("job");
}

export function App() {
  const [jobId, setJobId] = useState<string | null>(readHash);
  const { info, events, map, error, regenerate } = useJob(jobId);

  useEffect(() => {
    const onHash = () => setJobId(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const open = (id: string | null) => {
    window.location.hash = id ? `job=${id}` : "";
    setJobId(id);
  };

  let body;
  if (!jobId) {
    body = <UploadPanel onCreated={(job) => open(job.id)} />;
  } else if (map && info) {
    body = <Preview info={info} map={map} regenerate={regenerate} />;
  } else if (info?.status === "failed") {
    body = (
      <div className="card center">
        <h2>Generation failed</h2>
        <p className="error">{info.error}</p>
        <button className="button primary" onClick={() => open(null)}>
          Try another song
        </button>
      </div>
    );
  } else if (info) {
    body = <ProgressView info={info} events={events} />;
  } else if (error) {
    body = (
      <div className="card center">
        <h2>Job not found</h2>
        <p className="error">{error}</p>
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
            New song
          </button>
        )}
      </header>
      {error && map && <div className="banner error">{error}</div>}
      <main>{body}</main>
    </div>
  );
}
