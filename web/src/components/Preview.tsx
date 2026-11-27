import { useEffect, useMemo, useRef, useState } from "react";
import {
  DIFFICULTIES,
  DIFFICULTY_LABEL,
  MODE_LABEL,
  TERMINAL,
  audioUrl,
  downloadUrl,
  installJob,
  setFeedback,
  setSongInfo,
  type Characteristic,
  type Difficulty,
  type DifficultyMap,
  type InstallResult,
  type JobInfo,
  type MapPreview,
  type RegenerateRequest,
} from "../api/client";
import { TrackScene } from "../preview/TrackScene";
import { TempoMap, formatTime, reactionTime } from "../preview/timing";
import { ProgressBar } from "./ProgressView";
import { Timeline } from "./Timeline";
import { WarningsPanel } from "./WarningsPanel";

const SPEEDS = [0.5, 0.75, 1];
const SEEK_S = 5;
const SPAWN_KEY = "bsmapper.preview.gameSpawn";

interface Props {
  info: JobInfo;
  map: MapPreview;
  regenerate: (req: RegenerateRequest) => Promise<void>;
  /** Fetch the job and map again after renaming the song. */
  reload: () => Promise<void>;
  /** Go to the page that makes this song again with other options. */
  remake: () => void;
}

interface DiffStats {
  notes: number;
  nps: number;
  njs: number;
  reaction: number;
}

function diffStats(dmap: DifficultyMap, tempo: TempoMap, bpm: number): DiffStats {
  const notes = dmap.notes ?? [];
  const beats = notes.map((n) => n.beat);
  const first = beats.length ? Math.min(...beats) : 0;
  const last = beats.length ? Math.max(...beats) : 0;
  const span = tempo.beatToSeconds(last) - tempo.beatToSeconds(first);
  return {
    notes: notes.length,
    nps: notes.length / Math.max(1, span),
    njs: dmap.njs,
    reaction: reactionTime(dmap.njs, dmap.njs_offset ?? 0, bpm),
  };
}

function readSpawn(): boolean {
  try {
    return localStorage.getItem(SPAWN_KEY) !== "far";
  } catch {
    return true;
  }
}

export function Preview({ info, map, regenerate, reload, remake }: Props) {
  const beatmap = map.beatmap;
  const diffs = beatmap.difficulties ?? [];
  const modes = useMemo(() => [...new Set(diffs.map((d) => d.characteristic))], [diffs]);
  const [mode, setMode] = useState<Characteristic>(modes[0]);
  const available = useMemo(
    () =>
      diffs
        .filter((d) => d.characteristic === mode)
        .map((d) => d.difficulty)
        .sort((a, b) => DIFFICULTIES.indexOf(a) - DIFFICULTIES.indexOf(b)),
    [diffs, mode],
  );
  const [difficulty, setDifficulty] = useState<Difficulty>(available[available.length - 1]);
  const dmap = diffs.find((d) => d.characteristic === mode && d.difficulty === difficulty) ?? diffs[0];
  const tempo = useMemo(() => new TempoMap(beatmap), [beatmap]);
  const stats = useMemo(() => {
    const out = new Map<string, DiffStats>();
    for (const d of diffs) out.set(`${d.characteristic}/${d.difficulty}`, diffStats(d, tempo, beatmap.bpm));
    return out;
  }, [diffs, tempo, beatmap.bpm]);
  const current = dmap ? stats.get(`${dmap.characteristic}/${dmap.difficulty}`) : undefined;

  const audio = useRef<HTMLAudioElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const scene = useRef<TrackScene | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [gameSpawn, setGameSpawn] = useState(readSpawn);
  const [webglError, setWebglError] = useState<string | null>(null);

  // Keep the difficulty tab valid when the mode changes.
  useEffect(() => {
    if (!available.includes(difficulty)) setDifficulty(available[available.length - 1]);
  }, [available, difficulty]);

  useEffect(() => {
    if (!stage.current) return;
    try {
      scene.current = new TrackScene(stage.current, () => audio.current?.currentTime ?? 0);
    } catch (e) {
      setWebglError(`3D preview unavailable: ${(e as Error).message}`);
    }
    return () => {
      scene.current?.dispose();
      scene.current = null;
    };
  }, []);

  useEffect(() => {
    if (dmap) scene.current?.setMap(beatmap, dmap, dmap.lightshow ?? beatmap.lightshow ?? { uses_chroma: false });
  }, [beatmap, dmap]);

  useEffect(() => {
    scene.current?.setSpawn(gameSpawn && current ? current.reaction : null);
    try {
      localStorage.setItem(SPAWN_KEY, gameSpawn ? "game" : "far");
    } catch {
      /* not remembered */
    }
  }, [gameSpawn, current?.reaction]);

  useEffect(() => {
    if (audio.current) audio.current.playbackRate = speed;
  }, [speed]);

  const toggle = () => {
    const a = audio.current;
    if (!a) return;
    if (a.paused) void a.play();
    else a.pause();
  };
  const seek = (seconds: number) => {
    if (audio.current) audio.current.currentTime = Math.max(0, Math.min(map.duration_s, seconds));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
      const a = audio.current;
      if (e.code === "Space" && tag !== "BUTTON") {
        e.preventDefault();
        toggle();
      } else if (a && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        e.preventDefault();
        // Shift: one beat, for looking at a pattern closely.
        const step = e.shiftKey ? 60 / beatmap.bpm : SEEK_S;
        seek(a.currentTime + (e.key === "ArrowLeft" ? -step : step));
      } else if (a && e.key === "Home") {
        e.preventDefault();
        seek(0);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [beatmap.bpm, map.duration_s]);

  const busy = !TERMINAL.includes(info.status);
  const allWarnings = map.warnings ?? [];
  const warnings = allWarnings.filter((w) => w.characteristic === mode && w.difficulty === difficulty);
  const warningCount = (m: string, d: string) =>
    allWarnings.filter((w) => w.characteristic === m && w.difficulty === d).length;
  const rows = diffs
    .slice()
    .sort(
      (a, b) =>
        modes.indexOf(a.characteristic) - modes.indexOf(b.characteristic) ||
        DIFFICULTIES.indexOf(a.difficulty) - DIFFICULTIES.indexOf(b.difficulty),
    );

  return (
    <div className="preview">
      <div className="preview-main">
        <div className="tabs-row">
          <div className="tabs" role="tablist" aria-label="Mode">
            {modes.map((m) => (
              <button key={m} role="tab" aria-selected={m === mode} className={m === mode ? "on" : ""} onClick={() => setMode(m)}>
                {MODE_LABEL[m] ?? m}
              </button>
            ))}
          </div>
          <div className="tabs" role="tablist" aria-label="Difficulty">
            {available.map((d) => {
              const n = warningCount(mode, d);
              return (
                <button
                  key={d}
                  role="tab"
                  aria-selected={d === difficulty}
                  className={`diff-${d} ${d === difficulty ? "on" : ""}`}
                  onClick={() => setDifficulty(d)}
                  aria-label={`${DIFFICULTY_LABEL[d]}${n ? `, ${n} warning${n > 1 ? "s" : ""}` : ""}`}
                >
                  {DIFFICULTY_LABEL[d]}
                  {n > 0 && <span className="tab-count">{n}</span>}
                </button>
              );
            })}
          </div>
        </div>

        <div className="stage" ref={stage} onClick={toggle}>
          {webglError && <div className="stage-msg">{webglError}</div>}
          {!playing && !webglError && (
            <div className="stage-hint">Click or press Space to play · ← → seek 5 s (Shift: 1 beat)</div>
          )}
        </div>

        <div className="transport">
          <button className="play" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
            {playing ? "❚❚" : "▶"}
          </button>
          <Timeline
            audio={audio}
            duration={map.duration_s}
            tempo={tempo}
            dmap={dmap}
            warnings={warnings}
            onSeek={seek}
          />
        </div>
        <div className="transport-options">
          <div className="segmented small" title="Playback speed (to look at fast patterns)">
            {SPEEDS.map((s) => (
              <label key={s} className={speed === s ? "on" : ""}>
                <input type="radio" name="speed" checked={speed === s} onChange={() => setSpeed(s)} />
                {s === 1 ? "1×" : `${s}×`}
              </label>
            ))}
          </div>
          <label
            className="inline-toggle"
            title="Show notes only from when they appear in the game (the reaction time), instead of the whole track ahead"
          >
            <input type="checkbox" checked={gameSpawn} onChange={() => setGameSpawn(!gameSpawn)} />
            Spawn like the game
          </label>
        </div>
        <audio
          ref={audio}
          src={audioUrl(map.job_id)}
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onLoadedMetadata={(e) => (e.currentTarget.playbackRate = speed)}
        />

        <DifficultyTable
          rows={rows}
          stats={stats}
          warningCount={warningCount}
          multiMode={modes.length > 1}
          selected={dmap}
          onSelect={(d) => {
            setMode(d.characteristic);
            setDifficulty(d.difficulty);
          }}
        />
      </div>

      <aside className="side">
        <SongCard info={info} map={map} busy={busy} reload={reload} stats={current} />

        <WarningsPanel warnings={warnings} notes={map.notes ?? []} tempo={tempo} onSeek={seek} />

        <RegenerateBox
          key={`${mode}/${difficulty}`}
          busy={busy}
          info={info}
          label={`${MODE_LABEL[mode] ?? mode} ${DIFFICULTY_LABEL[difficulty]}`}
          onRegenerate={(seed, temperature) =>
            regenerate({ characteristic: mode, difficulty, seed, temperature })
          }
        />

        <div className="actions">
          <a className={`button primary ${busy ? "disabled" : ""}`} href={busy ? undefined : downloadUrl(map.job_id)} download>
            Download .zip
          </a>
          <InstallButton
            jobId={map.job_id}
            revision={map.revision}
            busy={busy}
            installed={info.installed ?? null}
            installError={info.install_error ?? null}
          />
          <button className="button" onClick={remake} title="Make the whole song again, with other options or a new seed">
            Make again with other options…
          </button>
        </div>

        <PlaytestCard key={info.id} info={info} />

        <SettingsUsed info={info} />
      </aside>
    </div>
  );
}

function DifficultyTable({
  rows,
  stats,
  warningCount,
  multiMode,
  selected,
  onSelect,
}: {
  rows: DifficultyMap[];
  stats: Map<string, DiffStats>;
  warningCount: (m: string, d: string) => number;
  multiMode: boolean;
  selected: DifficultyMap | undefined;
  onSelect: (d: DifficultyMap) => void;
}) {
  return (
    <div className="card diff-table">
      <table>
        <thead>
          <tr>
            <th>Difficulty</th>
            <th title="Notes, without bombs">Notes</th>
            <th title="Average notes per second from the first note to the last">NPS</th>
            <th title="Note jump speed">NJS</th>
            <th title="Seconds from a note appearing to reaching you">Reaction</th>
            <th>Warnings</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => {
            const s = stats.get(`${d.characteristic}/${d.difficulty}`);
            const n = warningCount(d.characteristic, d.difficulty);
            const on = d === selected;
            return (
              <tr key={`${d.characteristic}/${d.difficulty}`} className={on ? "on" : ""} onClick={() => onSelect(d)}>
                <td>
                  <span className={`diff-dot diff-${d.difficulty}`} />
                  {multiMode ? `${MODE_LABEL[d.characteristic] ?? d.characteristic} ` : ""}
                  {DIFFICULTY_LABEL[d.difficulty]}
                </td>
                <td>{s?.notes}</td>
                <td>{s?.nps.toFixed(2)}</td>
                <td>{s?.njs}</td>
                <td>{s ? `${s.reaction.toFixed(2)} s` : ""}</td>
                <td className={n ? "warn" : "muted"}>{n || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SongCard({
  info,
  map,
  busy,
  reload,
  stats,
}: {
  info: JobInfo;
  map: MapPreview;
  busy: boolean;
  reload: () => Promise<void>;
  stats: DiffStats | undefined;
}) {
  const beatmap = map.beatmap;
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(beatmap.song_name);
  const [artist, setArtist] = useState(beatmap.song_author);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startEdit = () => {
    setTitle(beatmap.song_name);
    setArtist(beatmap.song_author);
    setError(null);
    setEditing(true);
  };
  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      await setSongInfo(info.id, title, artist);
      await reload();
      setEditing(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card song">
      <div className="eyebrow">{beatmap.mapper}</div>
      {editing ? (
        <form
          className="song-edit"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <label>
            Title
            <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} autoFocus />
          </label>
          <label>
            Artist
            <input value={artist} onChange={(e) => setArtist(e.target.value)} maxLength={200} />
          </label>
          <div className="muted">This is what the game's song list shows{info.installed ? "; the installed copy is replaced" : ""}.</div>
          {error && <div className="error">{error}</div>}
          <div className="button-row">
            <button className="button primary small" type="submit" disabled={saving || !title.trim()}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button className="button small" type="button" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className="song-title">
            <h2 title={beatmap.song_name}>{beatmap.song_name}</h2>
            <button
              className="ghost small"
              onClick={startEdit}
              disabled={busy}
              title="Fix the song title and artist (e.g. drop “Official Video”)"
            >
              Edit
            </button>
          </div>
          <div className="muted">{beatmap.song_author || "Unknown artist"}</div>
        </>
      )}
      <dl className="stats">
        <div><dt>BPM</dt><dd>{beatmap.bpm}</dd></div>
        <div><dt>Length</dt><dd>{formatTime(map.duration_s)}</dd></div>
        <div><dt>Notes</dt><dd>{stats?.notes}</dd></div>
        <div><dt>NPS</dt><dd>{stats?.nps.toFixed(2)}</dd></div>
        <div><dt>NJS</dt><dd>{stats?.njs}</dd></div>
        <div title="Seconds from a note appearing to reaching you">
          <dt>Reaction</dt>
          <dd>{stats ? `${stats.reaction.toFixed(2)} s` : ""}</dd>
        </div>
      </dl>
    </div>
  );
}

/** The player's verdict after playing the map, kept with the job (docs/PLAYTEST.md). */
function PlaytestCard({ info }: { info: JobInfo }) {
  const [rating, setRating] = useState<-1 | 0 | 1>((info.feedback?.rating ?? 0) as -1 | 0 | 1);
  const [note, setNote] = useState(info.feedback?.note ?? "");
  const [saved, setSaved] = useState<string | null>(info.feedback?.note || info.feedback?.rating ? "Saved" : null);
  const [error, setError] = useState<string | null>(null);

  const save = (r: -1 | 0 | 1, n: string) => {
    setError(null);
    setSaved("Saving…");
    setFeedback(info.id, r, n).then(
      () => setSaved("Saved"),
      (e) => {
        setSaved(null);
        setError((e as Error).message);
      },
    );
  };
  const rate = (r: -1 | 1) => {
    const next = rating === r ? 0 : r;
    setRating(next);
    save(next, note);
  };

  return (
    <div className="card playtest">
      <h3>
        Playtest
        {saved && <span className="muted small-text">{saved}</span>}
      </h3>
      <div className="button-row left">
        <button className={`button small ${rating === 1 ? "good" : ""}`} onClick={() => rate(1)} aria-pressed={rating === 1}>
          👍 Good
        </button>
        <button className={`button small ${rating === -1 ? "bad" : ""}`} onClick={() => rate(-1)} aria-pressed={rating === -1}>
          👎 Needs work
        </button>
      </div>
      <textarea
        rows={3}
        value={note}
        placeholder="What did you notice? e.g. “Expert felt too easy”, “lights jitter at 1:40”"
        maxLength={4000}
        onChange={(e) => {
          setNote(e.target.value);
          setSaved(null);
        }}
        onBlur={() => note !== (info.feedback?.note ?? "") && save(rating, note)}
      />
      {error && <div className="error">{error}</div>}
    </div>
  );
}

/** The options this map was made with, so a good one can be made again. */
function SettingsUsed({ info }: { info: JobInfo }) {
  const o = info.options;
  const num = (v: number | null | undefined, unset: string) => (v == null ? unset : v.toFixed(2));
  const flag = (v: boolean | null | undefined) => (v == null ? "default" : v ? "on" : "off");
  const items: [string, string][] = [
    ["Seed", String(o.seed ?? 0)],
    ["Intensity", num(o.intensity, "default")],
    ["Style", num(o.style, "no preference")],
    ["Repetition", num(o.repetition, "learned")],
    ["Lighting", `${o.lighting ?? "v3"}${o.light_style ? ` · ${o.light_style}` : ""}`],
    ["Environment", info.options.lighting === "v3" ? (o.environment ?? "auto") : "—"],
    ["Arcs / chains", `${flag(o.arcs)} / ${flag(o.chains)}`],
    ["Timing", `${o.njs != null ? `NJS ${o.njs}` : "NJS default"}${o.reaction_time != null ? ` · ${o.reaction_time} s` : ""}`],
  ];
  return (
    <details className="card settings-used">
      <summary>Settings used</summary>
      <dl>
        {items.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

function InstallButton({
  jobId,
  revision,
  busy,
  installed,
  installError,
}: {
  jobId: string;
  revision: number;
  busy: boolean;
  installed: InstallResult | null; // installed by the queue ("install to game when done")
  installError: string | null; // why that automatic install failed
}) {
  const [installing, setInstalling] = useState(false);
  const [result, setResult] = useState<InstallResult | null>(installed);
  const [error, setError] = useState<string | null>(installError);
  // A regenerated map is not in the game until it is installed again (the server then
  // clears `installed`, which may arrive after the new revision).
  useEffect(() => {
    setResult(installed);
    setError(installError);
  }, [jobId, revision, installed?.path, installError]);

  const install = () => {
    setInstalling(true);
    setError(null);
    installJob(jobId)
      .then(setResult, (e) => {
        setResult(null);
        setError((e as Error).message);
      })
      .finally(() => setInstalling(false));
  };
  return (
    <>
      <button
        className="button"
        disabled={busy || installing}
        onClick={install}
        title="Copy the map into Beat Saber's CustomLevels folder (set in ~/.bsmapper/config.toml)."
      >
        {installing ? "Installing…" : result ? "Reinstall" : "Install to game"}
      </button>
      {(result || error) && (
        <div className="install-msg">
          {result && (
            <div className="ok" title={result.path}>
              Installed to {result.path}
            </div>
          )}
          {result?.warnings?.map((w) => (
            <div key={w} className="warn">
              {w}
            </div>
          ))}
          {error && <div className="error">{error}</div>}
        </div>
      )}
    </>
  );
}

function RegenerateBox({
  busy,
  info,
  label,
  onRegenerate,
}: {
  busy: boolean;
  info: JobInfo;
  label: string;
  onRegenerate: (seed: number, temperature: number) => Promise<void>;
}) {
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1e6));
  const [temperature, setTemperature] = useState(1);
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="card regen">
      <h3>Regenerate {label}</h3>
      <div className="regen-row">
        <label>
          Seed
          <input type="number" className="seed" value={seed} onChange={(e) => setSeed(Math.trunc(Number(e.target.value) || 0))} />
        </label>
        <button className="ghost" onClick={() => setSeed(Math.floor(Math.random() * 1e6))} title="New random seed">
          ⟳
        </button>
      </div>
      <label className="temp" title="Lower: safer, more common patterns. Higher: more varied and surprising.">
        Temperature <output>{temperature.toFixed(2)}</output>
        <input
          type="range"
          min={0.5}
          max={1.5}
          step={0.05}
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
        />
      </label>
      {busy ? (
        <div className="regen-busy">
          <div className="muted">{info.message || "Working…"}</div>
          <ProgressBar info={info} />
        </div>
      ) : (
        <button
          className="button"
          onClick={() => onRegenerate(seed, temperature).catch((e) => setError((e as Error).message))}
        >
          Regenerate
        </button>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
