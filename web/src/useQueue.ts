import { useCallback, useEffect, useRef, useState } from "react";
import { TERMINAL, listJobs, type JobInfo } from "./api/client";

// Polled rather than one SSE stream per job: a long queue would use up the browser's
// connections to the server.
const ACTIVE_POLL_MS = 1000;
const IDLE_POLL_MS = 5000;

export interface QueueState {
  jobs: JobInfo[];
  error: string | null;
  refresh: () => void;
}

/** Every job on the server (oldest first), refreshed while any is queued or running. */
export function useQueue(): QueueState {
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const alive = useRef(true);
  const latest = useRef(0); // only the newest request schedules the next poll

  const refresh = useCallback(() => {
    window.clearTimeout(timer.current);
    const call = ++latest.current;
    listJobs().then(
      (list) => {
        if (!alive.current || call !== latest.current) return;
        setJobs(list);
        setError(null);
        const active = list.some((j) => !TERMINAL.includes(j.status));
        timer.current = window.setTimeout(refresh, active ? ACTIVE_POLL_MS : IDLE_POLL_MS);
      },
      (e) => {
        if (!alive.current || call !== latest.current) return;
        setError((e as Error).message);
        timer.current = window.setTimeout(refresh, IDLE_POLL_MS);
      },
    );
  }, []);

  useEffect(() => {
    alive.current = true;
    refresh();
    return () => {
      alive.current = false;
      window.clearTimeout(timer.current);
    };
  }, [refresh]);

  return { jobs, error, refresh };
}

/** How many jobs run before this one (null if it is not waiting). */
export function jobsAhead(jobs: JobInfo[], id: string): number | null {
  const i = jobs.findIndex((j) => j.id === id);
  if (i < 0 || jobs[i].status !== "queued") return null;
  return jobs.slice(0, i).filter((j) => !TERMINAL.includes(j.status)).length;
}
