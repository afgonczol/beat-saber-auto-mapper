// The generation options form: its state, the JobOptions it sends, and the copy kept in the
// browser so the next visit starts with the options used last.
import type { Characteristic, Difficulty, JobOptions, LightingStyle } from "./api/client";

export type LightStyle = NonNullable<JobOptions["light_style"]>;

export interface FormOptions {
  modes: Characteristic[];
  difficulties: Difficulty[];
  lighting: LightingStyle;
  lightStyle: LightStyle | ""; // "" = the server's configured default
  environment: string; // v3 environment; "" = picked for each song
  useStyle: boolean;
  style: number;
  useIntensity: boolean; // false = the server's configured default
  intensity: number;
  useRepetition: boolean; // false = the learned amount
  repetition: number;
  arcs: boolean | null; // null = the server's default
  chains: boolean | null;
  seed: number;
  reactionTime: string; // "" = the server's defaults
  njs: string;
  install: boolean; // install to the game when done
}

export const DEFAULT_OPTIONS: FormOptions = {
  modes: ["Standard"],
  difficulties: ["Hard", "Expert", "ExpertPlus"],
  lighting: "v3",
  lightStyle: "",
  environment: "",
  useStyle: true,
  style: 0.5,
  useIntensity: false,
  intensity: 0.8,
  useRepetition: false,
  repetition: 0.5,
  arcs: null,
  chains: null,
  seed: 0,
  reactionTime: "",
  njs: "",
  install: false,
};

const KEY = "bsmapper.options.v1";

/** The options used last in this browser, else the defaults. */
export function loadOptions(): FormOptions {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (!saved || typeof saved !== "object") return DEFAULT_OPTIONS;
    const merged = { ...DEFAULT_OPTIONS };
    for (const k of Object.keys(DEFAULT_OPTIONS) as (keyof FormOptions)[]) {
      const d = DEFAULT_OPTIONS[k];
      const v = saved[k];
      // Keep only values of the right type; anything else falls back to the default.
      const ok =
        Array.isArray(d) ? Array.isArray(v) && v.length > 0 : d === null ? v === null || typeof v === "boolean" : typeof v === typeof d;
      if (ok) (merged as Record<string, unknown>)[k] = v;
    }
    return merged;
  } catch {
    return DEFAULT_OPTIONS;
  }
}

export function saveOptions(options: FormOptions): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(options));
  } catch {
    /* storage unavailable: the options just are not remembered */
  }
}

export function toJobOptions(f: FormOptions): JobOptions {
  return {
    modes: f.modes,
    difficulties: f.difficulties,
    lighting: f.lighting,
    style: f.useStyle ? f.style : null,
    intensity: f.useIntensity ? f.intensity : null,
    repetition: f.useRepetition ? f.repetition : null,
    light_style: f.lightStyle === "" ? null : f.lightStyle,
    arcs: f.arcs,
    chains: f.chains,
    seed: f.seed,
    reaction_time: f.reactionTime === "" ? null : Number(f.reactionTime),
    njs: f.njs === "" ? null : Number(f.njs),
    environment: f.lighting === "v3" && f.environment !== "" ? f.environment : null,
  };
}

/** The form for a job's options (to make the song again with a few changes). */
export function fromJobOptions(o: JobOptions, install = false): FormOptions {
  return {
    modes: o.modes ?? DEFAULT_OPTIONS.modes,
    difficulties: o.difficulties ?? DEFAULT_OPTIONS.difficulties,
    lighting: o.lighting ?? "v3",
    lightStyle: o.light_style ?? "",
    environment: o.environment ?? "",
    useStyle: o.style != null,
    style: o.style ?? DEFAULT_OPTIONS.style,
    useIntensity: o.intensity != null,
    intensity: o.intensity ?? DEFAULT_OPTIONS.intensity,
    useRepetition: o.repetition != null,
    repetition: o.repetition ?? DEFAULT_OPTIONS.repetition,
    arcs: o.arcs ?? null,
    chains: o.chains ?? null,
    seed: o.seed ?? 0,
    reactionTime: o.reaction_time != null ? String(o.reaction_time) : "",
    njs: o.njs != null ? String(o.njs) : "",
    install,
  };
}
