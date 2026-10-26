import { useEffect, useRef, useState, type DragEvent } from "react";
import {
  DIFFICULTIES,
  DIFFICULTY_LABEL,
  MODES,
  MODE_LABEL,
  createJob,
  isAudioFile,
  listEnvironments,
  type Characteristic,
  type Difficulty,
  type EnvironmentInfo,
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

// Folders dropped on the page: read the audio files inside (and in their subfolders).
async function filesFromEntry(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) {
    return new Promise((resolve) => (entry as FileSystemFileEntry).file((f) => resolve([f]), () => resolve([])));
  }
  if (!entry.isDirectory) return [];
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const entries: FileSystemEntry[] = [];
  // readEntries returns at most ~100 entries per call; keep reading until it returns none.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve) => reader.readEntries(resolve, () => resolve([])));
    if (batch.length === 0) break;
    entries.push(...batch);
  }
  const nested = await Promise.all(entries.map(filesFromEntry));
  return nested.flat().sort((a, b) => a.name.localeCompare(b.name));
}

async function droppedFiles(e: DragEvent): Promise<File[]> {
  const entries = [...e.dataTransfer.items]
    .map((item) => item.webkitGetAsEntry?.())
    .filter((x): x is FileSystemEntry => !!x);
  if (entries.length === 0) return [...e.dataTransfer.files];
  return (await Promise.all(entries.map(filesFromEntry))).flat();
}

const fileKey = (f: File) => `${f.name}:${f.size}:${f.lastModified}`;

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

interface Props {
  /** Called once per batch with the queued jobs, in queue order. */
  onQueued: (jobs: JobInfo[]) => void;
}

export function UploadPanel({ onQueued }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [installWhenDone, setInstallWhenDone] = useState(false);
  const [uploaded, setUploaded] = useState(0);
  const [modes, setModes] = useState<Characteristic[]>(["Standard"]);
  const [difficulties, setDifficulties] = useState<Difficulty[]>(["Hard", "Expert", "ExpertPlus"]);
  const [lighting, setLighting] = useState<LightingStyle>("v3");
  const [lightStyle, setLightStyle] = useState<LightStyle | "">("");
  // v3 environment (look and colours); "" = picked for each song.
  const [environment, setEnvironment] = useState("");
  const [environments, setEnvironments] = useState<EnvironmentInfo[]>([]);
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
  const folderInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Not in React's attribute types; lets the second picker choose a whole folder.
    folderInput.current?.setAttribute("webkitdirectory", "");
    listEnvironments().then(setEnvironments, () => setEnvironments([]));
  }, []);

  // Add songs to the batch (duplicates ignored); anything that is not audio is skipped.
  const addFiles = (incoming: File[]) => {
    const audio = incoming.filter((f) => isAudioFile(f.name));
    setSkipped(incoming.filter((f) => !isAudioFile(f.name)).map((f) => f.name));
    setFiles((prev) => {
      const seen = new Set(prev.map(fileKey));
      return [...prev, ...audio.filter((f) => !seen.has(fileKey(f)) && seen.add(fileKey(f)))];
    });
  };

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(await droppedFiles(e));
  };

  const submit = async () => {
    if (files.length === 0) return;
    setBusy(true);
    setError(null);
    setUploaded(0);
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
      environment: lighting === "v3" && environment !== "" ? environment : null,
    };
    // One upload at a time: the first song starts generating while the rest upload.
    const queued: JobInfo[] = [];
    let failed: File[] = [];
    for (const f of files) {
      try {
        queued.push(await createJob(f, options, installWhenDone));
      } catch (e) {
        failed = [...failed, f];
        setError(`${f.name}: ${(e as Error).message}`);
      }
      setUploaded((n) => n + 1);
    }
    setBusy(false);
    setFiles(failed); // keep what did not upload, to retry
    if (queued.length > 0) onQueued(queued);
  };

  const ready = files.length > 0 && modes.length > 0 && difficulties.length > 0 && !busy;
  const totalMb = files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
  let label = "Generate map";
  if (busy) label = files.length > 1 ? `Uploading ${Math.min(uploaded + 1, files.length)} of ${files.length}…` : "Uploading…";
  else if (files.length > 1) label = `Queue ${files.length} songs`;

  return (
    <div className="upload">
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${files.length > 0 ? "has-file" : ""}`}
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
        aria-label="Choose songs"
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(e) => {
            addFiles([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <input
          ref={folderInput}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            addFiles([...(e.target.files ?? [])]);
            e.target.value = "";
          }}
        />
        <div className="drop-icon" aria-hidden>
          <span className="cube red" />
          <span className="cube blue" />
        </div>
        {files.length === 1 ? (
          <>
            <div className="drop-title">{files[0].name}</div>
            <div className="muted">{totalMb.toFixed(1)} MB · click or drop to add more songs</div>
          </>
        ) : files.length > 1 ? (
          <>
            <div className="drop-title">{files.length} songs</div>
            <div className="muted">{totalMb.toFixed(1)} MB · click or drop to add more</div>
          </>
        ) : (
          <>
            <div className="drop-title">Drop songs here</div>
            <div className="muted">or click to choose · one or many · mp3, ogg, wav, flac, m4a</div>
          </>
        )}
        <button
          type="button"
          className="ghost small"
          onClick={(e) => {
            e.stopPropagation();
            folderInput.current?.click();
          }}
        >
          Choose a folder…
        </button>
        {files.length > 1 && (
          <ol className="file-list" onClick={(e) => e.stopPropagation()}>
            {files.map((f) => (
              <li key={fileKey(f)}>
                <span title={f.name}>{f.name}</span>
                <button
                  type="button"
                  className="ghost small"
                  disabled={busy}
                  aria-label={`Remove ${f.name}`}
                  onClick={() => setFiles(files.filter((x) => x !== f))}
                >
                  ✕
                </button>
              </li>
            ))}
          </ol>
        )}
        {files.length > 0 && !busy && (
          <button
            type="button"
            className="ghost small"
            onClick={(e) => {
              e.stopPropagation();
              setFiles([]);
              setSkipped([]);
            }}
          >
            Clear
          </button>
        )}
        {skipped.length > 0 && (
          <div className="muted" title={skipped.join("\n")}>
            Skipped {skipped.length} file{skipped.length > 1 ? "s" : ""} that {skipped.length > 1 ? "are" : "is"} not
            audio
          </div>
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
          {lighting === "v3" && environments.length > 0 && (
            <label className="light-style" title="The game environment: its look and colours">
              Environment{" "}
              <select value={environment} onChange={(e) => setEnvironment(e.target.value)}>
                <option value="">Auto (picked for each song)</option>
                <optgroup label="Human light effects">
                  {environments
                    .filter((e) => e.maps > 0)
                    .map((e) => (
                      <option key={e.name} value={e.name} title={`Effects from ${e.maps} human maps`}>
                        {e.label}
                        {e.auto ? "" : " (few maps)"}
                      </option>
                    ))}
                </optgroup>
                {environments.some((e) => e.maps === 0) && (
                  <optgroup label="Built-in lighting template">
                    {environments
                      .filter((e) => e.maps === 0)
                      .map((e) => (
                        <option key={e.name} value={e.name}>
                          {e.label}
                        </option>
                      ))}
                  </optgroup>
                )}
              </select>
            </label>
          )}
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

        <label className="install-toggle" title="Copy each map into Beat Saber's CustomLevels as soon as it is generated">
          <input type="checkbox" checked={installWhenDone} onChange={() => setInstallWhenDone(!installWhenDone)} />
          Install to game when done
        </label>

        {error && <div className="error">{error}</div>}
        <button type="button" className="primary big" disabled={!ready} onClick={submit}>
          {label}
        </button>
        {files.length > 1 && !busy && (
          <div className="muted center-text">Every song gets the options above; they run one after another.</div>
        )}
      </div>
    </div>
  );
}
