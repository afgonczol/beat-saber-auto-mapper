import { useRef, useState, type DragEvent } from "react";
import {
  DIFFICULTIES,
  DIFFICULTY_LABEL,
  MODES,
  MODE_LABEL,
  createJob,
  type Characteristic,
  type Difficulty,
  type JobInfo,
  type JobOptions,
  type LightingStyle,
} from "../api/client";

const ACCEPT = ".mp3,.ogg,.egg,.wav,.flac,.m4a,.aac,.opus,audio/*";
const LIGHTING: { value: LightingStyle; label: string; hint: string }[] = [
  { value: "vanilla", label: "Vanilla", hint: "Classic events, any environment" },
  { value: "v3", label: "v3 groups", hint: "Group lighting for v3 environments" },
  { value: "chroma", label: "Chroma", hint: "Custom RGB colors (needs the Chroma mod)" },
];

type LightStyle = NonNullable<JobOptions["light_style"]>;

// Look of the model's lights (light_v2 checkpoints); "" = the server's configured default.
const LIGHT_STYLES: { value: LightStyle | ""; label: string; hint: string }[] = [
  { value: "", label: "Default", hint: "The configured default (auto unless set)" },
  { value: "auto", label: "Auto", hint: "Picked for the song; another seed may pick another" },
  { value: "pulse", label: "Pulse", hint: "Beat-pulsed fades with laser sweeps" },
  { value: "calm", label: "Calm", hint: "Slow fade pulses, long holds, little laser motion" },
  { value: "scenes", label: "Scenes", hint: "Sustained on/off scenes, few colour flips" },
  { value: "flash", label: "Flash", hint: "Fast flashes with ring zooms and spins" },
  { value: "accent", label: "Accent", hint: "White accents and colour boosts" },
  { value: "none", label: "Model's choice", hint: "No style token: the model picks by chance" },
];

function toggle<T>(list: T[], item: T): T[] {
  return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
}

// Object types: null = the server's default (config, else arcs from Hard and chains from Expert).
const TRISTATE: { value: boolean | null; label: string }[] = [
  { value: null, label: "Default" },
  { value: true, label: "On" },
  { value: false, label: "Off" },
];

function TriState({
  name,
  value,
  onChange,
  hint,
}: {
  name: string;
  value: boolean | null;
  onChange: (v: boolean | null) => void;
  hint: string;
}) {
  return (
    <div className="segmented" title={hint}>
      {TRISTATE.map((t) => (
        <label key={t.label} className={value === t.value ? "on" : ""}>
          <input type="radio" name={name} checked={value === t.value} onChange={() => onChange(t.value)} />
          {t.label}
        </label>
      ))}
    </div>
  );
}

export function UploadPanel({ onCreated }: { onCreated: (job: JobInfo) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [modes, setModes] = useState<Characteristic[]>(["Standard"]);
  const [difficulties, setDifficulties] = useState<Difficulty[]>(["Hard", "Expert", "ExpertPlus"]);
  const [lighting, setLighting] = useState<LightingStyle>("v3");
  const [lightStyle, setLightStyle] = useState<LightStyle | "">("");
  const [useStyle, setUseStyle] = useState(true);
  const [style, setStyle] = useState(0.5);
  const [seed, setSeed] = useState(0);
  // Note timing; empty = the server's defaults.
  const [reactionTime, setReactionTime] = useState("");
  // Intensity; "default" = the server's configured default.
  const [useIntensity, setUseIntensity] = useState(false);
  const [intensity, setIntensity] = useState(0.8);
  // Pattern reuse on repeated music; "learned" = what the model learned from human maps.
  const [useRepetition, setUseRepetition] = useState(false);
  const [repetition, setRepetition] = useState(0.5);
  const [arcs, setArcs] = useState<boolean | null>(null);
  const [chains, setChains] = useState<boolean | null>(null);
  const [njs, setNjs] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    const options: JobOptions = {
      modes,
      difficulties,
      lighting,
      style: useStyle ? style : null,
      intensity: useIntensity ? intensity : null,
      repetition: useRepetition ? repetition : null,
      light_style: lightStyle === "" ? null : lightStyle,
      arcs,
      chains,
      seed,
      reaction_time: reactionTime === "" ? null : Number(reactionTime),
      njs: njs === "" ? null : Number(njs),
    };
    try {
      onCreated(await createJob(file, options));
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  const ready = file && modes.length > 0 && difficulties.length > 0 && !busy;

  return (
    <div className="upload">
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        aria-label="Choose a song"
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
        />
        <div className="drop-icon" aria-hidden>
          <span className="cube red" />
          <span className="cube blue" />
        </div>
        {file ? (
          <>
            <div className="drop-title">{file.name}</div>
            <div className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB · click or drop to replace</div>
          </>
        ) : (
          <>
            <div className="drop-title">Drop a song here</div>
            <div className="muted">or click to choose · mp3, ogg, wav, flac, m4a</div>
          </>
        )}
      </div>

      <div className="card options">
        <fieldset>
          <legend>Modes</legend>
          <div className="chips">
            {MODES.map((m) => (
              <label key={m} className={`chip ${modes.includes(m) ? "on" : ""}`}>
                <input type="checkbox" checked={modes.includes(m)} onChange={() => setModes(toggle(modes, m))} />
                {MODE_LABEL[m]}
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>Difficulties</legend>
          <div className="chips">
            {DIFFICULTIES.map((d) => (
              <label key={d} className={`chip diff-${d} ${difficulties.includes(d) ? "on" : ""}`}>
                <input
                  type="checkbox"
                  checked={difficulties.includes(d)}
                  onChange={() => setDifficulties(toggle(difficulties, d))}
                />
                {DIFFICULTY_LABEL[d]}
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>Lighting</legend>
          <div className="segmented">
            {LIGHTING.map((l) => (
              <label key={l.value} className={lighting === l.value ? "on" : ""} title={l.hint}>
                <input
                  type="radio"
                  name="lighting"
                  checked={lighting === l.value}
                  onChange={() => setLighting(l.value)}
                />
                {l.label}
              </label>
            ))}
          </div>
          <label className="light-style" title={LIGHT_STYLES.find((l) => l.value === lightStyle)?.hint}>
            Light style{" "}
            <select value={lightStyle} onChange={(e) => setLightStyle(e.target.value as LightStyle | "")}>
              {LIGHT_STYLES.map((l) => (
                <option key={l.value} value={l.value} title={l.hint}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
        </fieldset>

        <fieldset>
          <legend>
            Style
            <label className="inline-toggle">
              <input type="checkbox" checked={!useStyle} onChange={() => setUseStyle(!useStyle)} />
              no preference
            </label>
          </legend>
          <div className={`style-slider ${useStyle ? "" : "disabled"}`}>
            <span>Flowy</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={style}
              disabled={!useStyle}
              onChange={(e) => setStyle(Number(e.target.value))}
              aria-label="Flowy to technical"
            />
            <span>Technical</span>
            <output>{useStyle ? style.toFixed(2) : "—"}</output>
          </div>
        </fieldset>

        <fieldset>
          <legend>
            Intensity
            <label className="inline-toggle">
              <input type="checkbox" checked={!useIntensity} onChange={() => setUseIntensity(!useIntensity)} />
              default
            </label>
          </legend>
          <div className={`style-slider ${useIntensity ? "" : "disabled"}`}>
            <span>Typical</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={intensity}
              disabled={!useIntensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              aria-label="Intensity within each difficulty"
              title="0.5 = a typical human map for the difficulty, 1 = about the hardest 10%"
            />
            <span>Hardest</span>
            <output>{useIntensity ? intensity.toFixed(2) : "—"}</output>
          </div>
        </fieldset>

        <fieldset>
          <legend>
            Repetition
            <label className="inline-toggle">
              <input type="checkbox" checked={!useRepetition} onChange={() => setUseRepetition(!useRepetition)} />
              learned
            </label>
          </legend>
          <div className={`style-slider ${useRepetition ? "" : "disabled"}`}>
            <span>Varied</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={repetition}
              disabled={!useRepetition}
              onChange={(e) => setRepetition(Number(e.target.value))}
              aria-label="Pattern reuse on repeated music"
              title="How strongly a repeated chorus reuses the earlier patterns; 0.5 = as learned"
            />
            <span>Repeated</span>
            <output>{useRepetition ? repetition.toFixed(2) : "—"}</output>
          </div>
        </fieldset>

        <fieldset className="row">
          <legend>Arcs</legend>
          <TriState name="arcs" value={arcs} onChange={setArcs} hint="Default: on from Hard up" />
        </fieldset>

        <fieldset className="row">
          <legend>Chains</legend>
          <TriState name="chains" value={chains} onChange={setChains} hint="Default: on from Expert up" />
        </fieldset>

        <fieldset className="row">
          <legend>Seed</legend>
          <input
            type="number"
            className="seed"
            value={seed}
            onChange={(e) => setSeed(Math.trunc(Number(e.target.value) || 0))}
          />
          <button type="button" className="ghost" onClick={() => setSeed(Math.floor(Math.random() * 1e6))}>
            Random
          </button>
        </fieldset>

        <fieldset className="row">
          <legend>Note timing</legend>
          <label title="Seconds from a note appearing to reaching you, for Expert and Expert+">
            Reaction time (Expert, Expert+)
            <input
              type="number"
              className="seed"
              min={0.3}
              max={2}
              step={0.05}
              placeholder="0.8 / 0.7 s"
              value={reactionTime}
              onChange={(e) => setReactionTime(e.target.value)}
            />
          </label>
          <label title="Note jump speed for every difficulty">
            NJS
            <input
              type="number"
              className="seed"
              min={1}
              max={40}
              step={0.5}
              placeholder="default"
              value={njs}
              onChange={(e) => setNjs(e.target.value)}
            />
          </label>
        </fieldset>

        {error && <div className="error">{error}</div>}
        <button type="button" className="primary big" disabled={!ready} onClick={submit}>
          {busy ? "Uploading…" : "Generate map"}
        </button>
      </div>
    </div>
  );
}
