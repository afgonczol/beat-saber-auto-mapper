import { useCallback, useEffect, useRef, useState } from "react";
import {
  TERMINAL,
  followJob,
  getJob,
  getMap,
  regenerate as postRegenerate,
  type JobEvent,
  type JobInfo,
  type MapPreview,
  type RegenerateRequest,
} from "./api/client";

export interface JobState {
  info: JobInfo | null;
  events: JobEvent[];
  map: MapPreview | null;
  error: string | null;
  regenerate: (req: RegenerateRequest) => Promise<void>;
  /** Fetch the job and its map again (after changing it outside a run, e.g. renaming). */
  reload: () => Promise<void>;
}

/** Load a job, follow its progress over SSE, and fetch the map whenever a run finishes. */
export function useJob(id: string | null): JobState {
  const [info, setInfo] = useState<JobInfo | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [map, setMap] = useState<MapPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastSeq = useRef(0);
  const stop = useRef<(() => void) | null>(null);

  const follow = useCallback((jobId: string) => {
    stop.current?.();
    stop.current = followJob(
      jobId,
      lastSeq.current,
      (e) => {
        lastSeq.current = Math.max(lastSeq.current, e.seq);
        setEvents((prev) => [...prev, e]);
        setInfo((prev) =>
          prev ? { ...prev, status: e.status, stage: e.stage, progress: e.progress, message: e.message } : prev,
        );
        if (e.type === "status" && TERMINAL.includes(e.status)) {
          getJob(jobId).then(
            (job) => {
              lastSeq.current = Math.max(lastSeq.current, job.last_seq);
              setInfo(job);
            },
            (err) => setError(String(err.message ?? err)),
          );
          if (e.status === "done") getMap(jobId).then(setMap, (err) => setError(String(err.message ?? err)));
        }
      },
      setError,
    );
  }, []);

  useEffect(() => {
    setInfo(null);
    setEvents([]);
    setMap(null);
    setError(null);
    lastSeq.current = 0;
    if (!id) return;
    let cancelled = false;
    getJob(id).then(
      (job) => {
        if (cancelled) return;
        setInfo(job);
        if (TERMINAL.includes(job.status)) lastSeq.current = job.last_seq;
        if (job.status === "done") getMap(id).then((m) => !cancelled && setMap(m), (e) => setError(e.message));
        if (!TERMINAL.includes(job.status)) follow(id);
      },
      (e) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
      stop.current?.();
      stop.current = null;
    };
  }, [id, follow]);

  const regenerate = useCallback(
    async (req: RegenerateRequest) => {
      if (!id) return;
      setError(null);
      const after = lastSeq.current;
      const job = await postRegenerate(id, req);
      setInfo(job);
      lastSeq.current = after; // replay this run's "queued" event too
      follow(id);
    },
    [id, follow],
  );

  const reload = useCallback(async () => {
    if (!id) return;
    const [job, m] = await Promise.all([getJob(id), getMap(id)]);
    setInfo(job);
    setMap(m);
  }, [id]);

  return { info, events, map, error, regenerate, reload };
}
