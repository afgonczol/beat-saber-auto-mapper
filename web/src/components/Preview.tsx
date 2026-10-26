import { useEffect, useMemo, useRef, useState } from "react";
import {
  DIFFICULTIES,
  DIFFICULTY_LABEL,
  MODE_LABEL,
  TERMINAL,
  audioUrl,
  downloadUrl,
  installJob,
  type Characteristic,
  type Difficulty,
  type InstallResult,
  type JobInfo,
  type MapPreview,
  type RegenerateRequest,
} from "../api/client";
import { TrackScene } from "../preview/TrackScene";
import { TempoMap, formatTime } from "../preview/timing";
import { ProgressBar } from "./ProgressView";
import { Timeline } from "./Timeline";
import { WarningsPanel } from "./WarningsPanel";

interface Props {
  info: JobInfo;
  map: MapPreview;
  regenerate: (req: RegenerateRequest) => Promise<void>;
}

export function Preview({ info, map, regenerate }: Props) {
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

  const audio = useRef<HTMLAudioElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const scene = useRef<TrackScene | null>(null);
  const [playing, setPlaying] = useState(false);
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

  const toggle = () => {
    const a = audio.current;
    if (!a) return;
    if (a.paused) void a.play();
    else a.pause();
  };
  const seek = (seconds: number) => {
    if (audio.current) audio.current.currentTime = Math.max(0, seconds);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tag)) {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const busy = !TERMINAL.includes(info.status);
  const warnings = (map.warnings ?? []).filter((w) => w.characteristic === mode && w.difficulty === difficulty);
  const notes = dmap?.notes?.length ?? 0;
  const lastBeat = Math.max(0, ...(dmap?.notes ?? []).map((n) => n.beat));
  const nps = notes / Math.max(1, tempo.beatToSeconds(lastBeat) - tempo.beatToSeconds(dmap?.notes?.[0]?.beat ?? 0));

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
            {available.map((d) => (
              <button
                key={d}
                role="tab"
                aria-selected={d === difficulty}
                className={`diff-${d} ${d === difficulty ? "on" : ""}`}
                onClick={() => setDifficulty(d)}
              >
                {DIFFICULTY_LABEL[d]}
              </button>
            ))}
          </div>
        </div>

        <div className="stage" ref={stage} onClick={toggle}>
          {webglError && <div className="stage-msg">{webglError}</div>}
          {!playing && !webglError && <div className="stage-hint">Click or press Space to play</div>}
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
        <audio
          ref={audio}
          src={audioUrl(map.job_id)}
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />
      </div>

      <aside className="side">
        <div className="card song">
          <div className="eyebrow">{beatmap.mapper}</div>
          <h2 title={beatmap.song_name}>{beatmap.song_name}</h2>
          <div className="muted">{beatmap.song_author || "Unknown artist"}</div>
          <dl className="stats">
            <div><dt>BPM</dt><dd>{beatmap.bpm}</dd></div>
            <div><dt>Length</dt><dd>{formatTime(map.duration_s)}</dd></div>
            <div><dt>Notes</dt><dd>{notes}</dd></div>
            <div><dt>NPS</dt><dd>{nps.toFixed(2)}</dd></div>
            <div><dt>NJS</dt><dd>{dmap?.njs}</dd></div>
            <div><dt>Lights</dt><dd>{info.options.lighting}</dd></div>
          </dl>
        </div>

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
          <InstallButton jobId={map.job_id} revision={map.revision} busy={busy} installed={info.installed ?? null} />
        </div>
      </aside>
    </div>
  );
}

function InstallButton({
  jobId,
  revision,
  busy,
  installed,
}: {
  jobId: string;
  revision: number;
  busy: boolean;
  installed: InstallResult | null; // installed by the queue ("install to game when done")
}) {
  const [installing, setInstalling] = useState(false);
  const [result, setResult] = useState<InstallResult | null>(installed);
  const [error, setError] = useState<string | null>(null);
  // A regenerated map is not in the game until it is installed again (the server then
  // clears `installed`, which may arrive after the new revision).
  useEffect(() => {
    setResult(installed);
    setError(null);
  }, [jobId, revision, installed?.path]);

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
      <label className="temp">
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
