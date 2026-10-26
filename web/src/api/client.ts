// Thin client for the jobs API. Types come from openapi.json (`npm run gen:api`).
import type { components } from "./schema";

type S = components["schemas"];
export type JobInfo = S["JobInfo"];
export type JobOptions = S["JobOptions"];
export type JobEvent = S["JobEvent"];
export type JobStatus = S["JobStatus"];
export type MapPreview = S["MapPreview"];
export type DifficultyWarning = S["DifficultyWarning"];
export type RegenerateRequest = S["RegenerateRequest"];
export type InstallResult = S["InstallResult"];
export type EnvironmentInfo = S["EnvironmentInfo"];
export type Beatmap = S["Beatmap"];
export type DifficultyMap = S["DifficultyMap"];
export type Lightshow = S["Lightshow"];
export type Characteristic = S["Characteristic"];
export type Difficulty = S["Difficulty"];
export type LightingStyle = JobOptions["lighting"];

export const MODES: Characteristic[] = ["Standard", "OneSaber", "NoArrows", "90Degree", "360Degree"];
export const DIFFICULTIES: Difficulty[] = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"];
export const TERMINAL: JobStatus[] = ["done", "failed", "cancelled"];
export const AUDIO_SUFFIXES = [".mp3", ".ogg", ".egg", ".wav", ".flac", ".m4a", ".aac", ".opus"];

export const isAudioFile = (name: string) => AUDIO_SUFFIXES.some((s) => name.toLowerCase().endsWith(s));

export const MODE_LABEL: Record<string, string> = {
  Standard: "Standard",
  OneSaber: "One Saber",
  NoArrows: "No Arrows",
  "90Degree": "90°",
  "360Degree": "360°",
  Lightshow: "Lightshow",
};
export const DIFFICULTY_LABEL: Record<string, string> = {
  Easy: "Easy",
  Normal: "Normal",
  Hard: "Hard",
  Expert: "Expert",
  ExpertPlus: "Expert+",
};

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

/** Queue one song. `install`: copy the map into the game as soon as it is generated. */
export async function createJob(audio: File, options: JobOptions, install = false): Promise<JobInfo> {
  const form = new FormData();
  form.append("audio", audio, audio.name);
  form.append("install", String(install));
  for (const m of options.modes ?? []) form.append("modes", m);
  for (const d of options.difficulties ?? []) form.append("difficulties", d);
  form.append("lighting", options.lighting ?? "v3");
  if (options.style != null) form.append("style", String(options.style));
  form.append("seed", String(options.seed ?? 0));
  if (options.intensity != null) form.append("intensity", String(options.intensity));
  if (options.repetition != null) form.append("repetition", String(options.repetition));
  if (options.light_style != null) form.append("light_style", options.light_style);
  if (options.arcs != null) form.append("arcs", String(options.arcs));
  if (options.chains != null) form.append("chains", String(options.chains));
  if (options.njs != null) form.append("njs", String(options.njs));
  if (options.reaction_time != null) form.append("reaction_time", String(options.reaction_time));
  if (options.environment != null) form.append("environment", options.environment);
  return json(await fetch(`${BASE}/jobs`, { method: "POST", body: form }));
}

export async function getJob(id: string): Promise<JobInfo> {
  return json(await fetch(`${BASE}/jobs/${id}`));
}

/** Environments v3 lighting can use: human light effects (most source maps first), then the template ones. */
export async function listEnvironments(): Promise<EnvironmentInfo[]> {
  return json(await fetch(`${BASE}/environments`));
}

/** Every job on the server, oldest first (the order they run in). */
export async function listJobs(): Promise<JobInfo[]> {
  return json(await fetch(`${BASE}/jobs`));
}

/** Cancel a queued job, or stop a running one at its next step. */
export async function cancelJob(id: string): Promise<JobInfo> {
  return json(await fetch(`${BASE}/jobs/${id}/cancel`, { method: "POST" }));
}

/** Take a job off the list (and the queue) and delete its files on the server. */
export async function removeJob(id: string): Promise<void> {
  const res = await fetch(`${BASE}/jobs/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) await json(res);
}

export async function getMap(id: string): Promise<MapPreview> {
  return json(await fetch(`${BASE}/jobs/${id}/map`));
}

export async function regenerate(id: string, req: RegenerateRequest): Promise<JobInfo> {
  return json(
    await fetch(`${BASE}/jobs/${id}/regenerate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  );
}

/** Copy the finished map into Beat Saber's CustomLevels (path from ~/.bsmapper/config.toml). */
export async function installJob(id: string): Promise<InstallResult> {
  return json(await fetch(`${BASE}/jobs/${id}/install`, { method: "POST" }));
}

export const audioUrl = (id: string) => `${BASE}/jobs/${id}/audio`;
export const downloadUrl = (id: string) => `${BASE}/jobs/${id}/download`;

/** Follow a job's SSE stream until it reports done/failed. Returns an unsubscribe function. */
export function followJob(
  id: string,
  after: number,
  onEvent: (e: JobEvent) => void,
  onError: (message: string) => void,
): () => void {
  const es = new EventSource(`${BASE}/jobs/${id}/events?after=${after}`);
  let closed = false;
  const handle = (msg: MessageEvent) => {
    const event = JSON.parse(msg.data) as JobEvent;
    onEvent(event);
    if (event.type === "status" && TERMINAL.includes(event.status)) {
      closed = true;
      es.close();
    }
  };
  es.addEventListener("status", handle);
  es.addEventListener("progress", handle);
  es.onerror = () => {
    // EventSource retries by itself; only give up if the server refuses outright.
    if (!closed && es.readyState === EventSource.CLOSED) onError("Lost the connection to the server.");
  };
  return () => {
    closed = true;
    es.close();
  };
}
