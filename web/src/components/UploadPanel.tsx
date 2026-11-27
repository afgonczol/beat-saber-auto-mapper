import { useEffect, useRef, useState, type DragEvent } from "react";
import {
  DIFFICULTIES,
  DIFFICULTY_LABEL,
  MODES,
  MODE_LABEL,
  createJob,
  isAudioFile,
  jobName,
  listEnvironments,
  rerunJob,
  type EnvironmentInfo,
  type JobInfo,
  type LightingStyle,
  type ServerSettings,
} from "../api/client";
import {
  DEFAULT_OPTIONS,
  fromJobOptions,
  loadOptions,
  saveOptions,
  toJobOptions,
  type FormOptions,
  type LightStyle,
} from "../options";

const ACCEPT = ".mp3,.ogg,.egg,.wav,.flac,.m4a,.aac,.opus,audio/*";
const LIGHTING: { value: LightingStyle; label: string; hint: string }[] = [
  { value: "vanilla", label: "Vanilla", hint: "Classic events, any environment" },
  { value: "v3", label: "v3 groups", hint: "Group lighting for v3 environments" },
  { value: "chroma", label: "Chroma", hint: "Custom RGB colors (needs the Chroma mod)" },
];

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

const onOff = (v: boolean | null | undefined, unset: string) => (v == null ? unset : v ? "on" : "off");

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

/** Title and artist fields for one song (what the game's song list shows). */
function SongName({
  title,
  artist,
  setTitle,
  setArtist,
  placeholder,
}: {
  title: string;
  artist: string;
  setTitle: (v: string) => void;
  setArtist: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="song-name" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <label>
        Title
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder={placeholder} maxLength={200} />
      </label>
      <label>
        Artist
        <input value={artist} onChange={(e) => setArtist(e.target.value)} placeholder={placeholder} maxLength={200} />
      </label>
    </div>
  );
}

interface Props {
  /** Called once per batch with the queued jobs, in queue order. */
  onQueued: (jobs: JobInfo[]) => void;
  settings: ServerSettings | null;
  /** Make this job's song again (its upload is on the server) instead of uploading songs. */
  remake?: JobInfo | null;
  onCancelRemake?: () => void;
}

export function UploadPanel({ onQueued, settings, remake, onCancelRemake }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [uploaded, setUploaded] = useState(0);
  // A remake starts from that job's options; otherwise from the options used last.
  const [opts, setOpts] = useState<FormOptions>(() =>
    remake ? fromJobOptions(remake.options, loadOptions().install) : loadOptions(),
  );
  const set = (patch: Partial<FormOptions>) => setOpts((o) => ({ ...o, ...patch }));
  // The song's name in the map; "" = from the file (one song) or as before (a remake).
  const [title, setTitle] = useState(remake?.title ?? "");
  const [artist, setArtist] = useState(remake?.artist ?? "");
  const [environments, setEnvironments] = useState<EnvironmentInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Not in React's attribute types; lets the second picker choose a whole folder.
    folderInput.current?.setAttribute("webkitdirectory", "");
    listEnvironments().then(setEnvironments, () => setEnvironments([]));
  }, []);

  useEffect(() => {
    if (!remake) saveOptions(opts);
  }, [opts, remake]);

  // A typed song name belongs to that one song.
  const single = files.length === 1 ? fileKey(files[0]) : null;
  useEffect(() => {
    if (!remake) {
      setTitle("");
      setArtist("");
    }
  }, [single, remake]);

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
    setBusy(true);
    setError(null);
    setUploaded(0);
    const options = toJobOptions(opts);
    if (songs === 1 && title.trim()) {
      options.title = title.trim();
      options.artist = artist.trim();
    }
    if (remake) {
      try {
        onQueued([await rerunJob(remake.id, options, opts.install)]);
      } catch (e) {
        setError((e as Error).message);
      }
      setBusy(false);
      return;
    }
    // One upload at a time: the first song starts generating while the rest upload.
    const queued: JobInfo[] = [];
    let failed: File[] = [];
    for (const f of files) {
      try {
        queued.push(await createJob(f, options, opts.install));
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

  const songs = remake ? 1 : files.length;
  const ready = songs > 0 && opts.modes.length > 0 && opts.difficulties.length > 0 && !busy;
  const totalMb = files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
  let label = remake ? "Make it again" : "Generate map";
  if (busy && remake) label = "Queueing…";
  else if (busy) label = files.length > 1 ? `Uploading ${Math.min(uploaded + 1, files.length)} of ${files.length}…` : "Uploading…";
  else if (files.length > 1) label = `Queue ${files.length} songs`;
  const maps = songs * opts.modes.length * opts.difficulties.length;

  const jump = settings?.jump;
  const rtPlaceholder = jump
    ? `${jump.Expert?.reaction_time ?? 0.8} / ${jump.ExpertPlus?.reaction_time ?? 0.7} s`
    : "0.8 / 0.7 s";
  const defaultLightStyle = settings ? `Default (${settings.light_style})` : "Default";
  const install = settings?.install;

  return (
    <div className="upload">
      {remake ? (
        <div className="dropzone has-file remake">
          <div className="eyebrow">Make again</div>
          <div className="drop-title">{jobName(remake)}</div>
          <div className="muted">
            Uses the song already on the server, so nothing is uploaded. Change any option, then click{" "}
            <b>Make it again</b>. The earlier map stays in the queue.
          </div>
          <SongName title={title} artist={artist} setTitle={setTitle} setArtist={setArtist} />
          <button type="button" className="ghost small" onClick={onCancelRemake}>
            Cancel
          </button>
        </div>
      ) : (
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
              <SongName
                title={title}
                artist={artist}
                setTitle={setTitle}
                setArtist={setArtist}
                placeholder="from the file"
              />
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
      )}

      <div className="card options">
        <div className="options-head">
          <span className="muted">
            {remake ? "Options of the earlier map" : "Your options are remembered in this browser."}
          </span>
          <button
            type="button"
            className="ghost small"
            onClick={() => setOpts(remake ? fromJobOptions(remake.options, opts.install) : DEFAULT_OPTIONS)}
            title={remake ? "Back to the earlier map's options" : "Back to the standard options"}
          >
            Reset
          </button>
        </div>

        <fieldset>
          <legend>Modes</legend>
          <div className="chips">
            {MODES.map((m) => (
              <label key={m} className={`chip ${opts.modes.includes(m) ? "on" : ""}`}>
                <input type="checkbox" checked={opts.modes.includes(m)} onChange={() => set({ modes: toggle(opts.modes, m) })} />
                {MODE_LABEL[m]}
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>Difficulties</legend>
          <div className="chips">
            {DIFFICULTIES.map((d) => (
              <label key={d} className={`chip diff-${d} ${opts.difficulties.includes(d) ? "on" : ""}`}>
                <input
                  type="checkbox"
                  checked={opts.difficulties.includes(d)}
                  onChange={() => set({ difficulties: toggle(opts.difficulties, d) })}
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
              <label key={l.value} className={opts.lighting === l.value ? "on" : ""} title={l.hint}>
                <input
                  type="radio"
                  name="lighting"
                  checked={opts.lighting === l.value}
                  onChange={() => set({ lighting: l.value })}
                />
                {l.label}
              </label>
            ))}
          </div>
          {opts.lighting === "chroma" && install && !install.chroma && install.found && (
            <div className="hint warn">Chroma is not installed in the game; these lights play in the default colours.</div>
          )}
          <label className="light-style" title={LIGHT_STYLES.find((l) => l.value === opts.lightStyle)?.hint}>
            Light style{" "}
            <select value={opts.lightStyle} onChange={(e) => set({ lightStyle: e.target.value as LightStyle | "" })}>
              {LIGHT_STYLES.map((l) => (
                <option key={l.value} value={l.value} title={l.hint}>
                  {l.value === "" ? defaultLightStyle : l.label}
                </option>
              ))}
            </select>
          </label>
          {opts.lighting === "v3" && environments.length > 0 && (
            <label className="light-style" title="The game environment: its look and colours">
              Environment{" "}
              <select value={opts.environment} onChange={(e) => set({ environment: e.target.value })}>
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
              <input type="checkbox" checked={!opts.useStyle} onChange={() => set({ useStyle: !opts.useStyle })} />
              no preference
            </label>
          </legend>
          <div className={`style-slider ${opts.useStyle ? "" : "disabled"}`}>
            <span>Flowy</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opts.style}
              disabled={!opts.useStyle}
              onChange={(e) => set({ style: Number(e.target.value) })}
              aria-label="Flowy to technical"
            />
            <span>Technical</span>
            <output>{opts.useStyle ? opts.style.toFixed(2) : "—"}</output>
          </div>
        </fieldset>

        <fieldset>
          <legend>
            Intensity
            <label className="inline-toggle">
              <input
                type="checkbox"
                checked={!opts.useIntensity}
                onChange={() => set({ useIntensity: !opts.useIntensity })}
              />
              default{settings ? ` (${settings.intensity.toFixed(2)})` : ""}
            </label>
          </legend>
          <div className={`style-slider ${opts.useIntensity ? "" : "disabled"}`}>
            <span>Typical</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opts.useIntensity ? opts.intensity : (settings?.intensity ?? opts.intensity)}
              disabled={!opts.useIntensity}
              onChange={(e) => set({ intensity: Number(e.target.value) })}
              aria-label="Intensity within each difficulty"
              title="0.5 = a typical human map for the difficulty, 1 = about the hardest 10%"
            />
            <span>Hardest</span>
            <output>{opts.useIntensity ? opts.intensity.toFixed(2) : "—"}</output>
          </div>
        </fieldset>

        <fieldset>
          <legend>
            Repetition
            <label className="inline-toggle">
              <input
                type="checkbox"
                checked={!opts.useRepetition}
                onChange={() => set({ useRepetition: !opts.useRepetition })}
              />
              {settings?.repetition != null ? `default (${settings.repetition.toFixed(2)})` : "learned"}
            </label>
          </legend>
          <div className={`style-slider ${opts.useRepetition ? "" : "disabled"}`}>
            <span>Varied</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opts.repetition}
              disabled={!opts.useRepetition}
              onChange={(e) => set({ repetition: Number(e.target.value) })}
              aria-label="Pattern reuse on repeated music"
              title="How strongly a repeated chorus reuses the earlier patterns; 0.5 = as learned"
            />
            <span>Repeated</span>
            <output>{opts.useRepetition ? opts.repetition.toFixed(2) : "—"}</output>
          </div>
        </fieldset>

        <fieldset className="row">
          <legend>Arcs</legend>
          <TriState
            name="arcs"
            value={opts.arcs}
            onChange={(arcs) => set({ arcs })}
            hint={`Default: ${onOff(settings?.arcs, "on from Hard up")}`}
          />
        </fieldset>

        <fieldset className="row">
          <legend>Chains</legend>
          <TriState
            name="chains"
            value={opts.chains}
            onChange={(chains) => set({ chains })}
            hint={`Default: ${onOff(settings?.chains, "on from Expert up")}`}
          />
        </fieldset>

        <fieldset className="row">
          <legend>Seed</legend>
          <input
            type="number"
            className="seed"
            value={opts.seed}
            onChange={(e) => set({ seed: Math.trunc(Number(e.target.value) || 0) })}
            title="Another seed gives another map for the same song and options"
          />
          <button type="button" className="ghost" onClick={() => set({ seed: Math.floor(Math.random() * 1e6) })}>
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
              placeholder={rtPlaceholder}
              value={opts.reactionTime}
              onChange={(e) => set({ reactionTime: e.target.value })}
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
              value={opts.njs}
              onChange={(e) => set({ njs: e.target.value })}
            />
          </label>
        </fieldset>

        <label className="install-toggle" title="Copy each map into Beat Saber's CustomLevels as soon as it is generated">
          <input type="checkbox" checked={opts.install} onChange={() => set({ install: !opts.install })} />
          Install to game when done
        </label>
        {install && !install.found && (
          <div className="hint warn">
            {install.error
              ? `The config file has an error: ${install.error}`
              : `Beat Saber's CustomLevels folder was not found (${install.path}).`}{" "}
            Set <code>[game] custom_levels</code> in <code>{install.config}</code>, then restart the server.
          </div>
        )}

        {error && <div className="error">{error}</div>}
        <button type="button" className="primary big" disabled={!ready} onClick={submit}>
          {label}
        </button>
        {songs > 0 && !busy && (
          <div className="muted center-text">
            {maps} map{maps === 1 ? "" : "s"}
            {files.length > 1 ? " · every song gets the options above; they run one after another" : ""}
          </div>
        )}
      </div>
    </div>
  );
}
