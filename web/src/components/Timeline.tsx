import { useEffect, useMemo, useRef, type PointerEvent, type RefObject } from "react";
import type { DifficultyMap, DifficultyWarning } from "../api/client";
import { formatTime, type TempoMap } from "../preview/timing";

const BINS = 160;

interface Props {
  audio: RefObject<HTMLAudioElement>;
  duration: number;
  tempo: TempoMap;
  dmap: DifficultyMap | undefined;
  warnings: DifficultyWarning[];
  onSeek: (seconds: number) => void;
}

/** Note-density strip with warning markers and a playhead; click or drag to seek. */
export function Timeline({ audio, duration, tempo, dmap, warnings, onSeek }: Props) {
  const bar = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const time = useRef<HTMLSpanElement>(null);

  const bins = useMemo(() => {
    const counts = new Array<number>(BINS).fill(0);
    for (const n of dmap?.notes ?? []) {
      const i = Math.floor((tempo.beatToSeconds(n.beat) / duration) * BINS);
      if (i >= 0 && i < BINS) counts[i]++;
    }
    const max = Math.max(1, ...counts);
    return counts.map((c) => c / max);
  }, [dmap, tempo, duration]);

  // The playhead moves every frame without re-rendering React.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const t = audio.current?.currentTime ?? 0;
      if (head.current) head.current.style.left = `${Math.min(100, (t / duration) * 100)}%`;
      if (time.current) time.current.textContent = `${formatTime(t)} / ${formatTime(duration)}`;
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [audio, duration]);

  const seekAt = (e: PointerEvent) => {
    const r = bar.current?.getBoundingClientRect();
    if (!r) return;
    onSeek(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * duration);
  };

  return (
    <div className="timeline">
      <div
        className="timeline-bar"
        ref={bar}
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          seekAt(e);
        }}
        onPointerMove={(e) => e.buttons === 1 && seekAt(e)}
      >
        <div className="density">
          {bins.map((v, i) => (
            <span key={i} style={{ height: `${Math.max(4, v * 100)}%` }} />
          ))}
        </div>
        {warnings.map((w, i) => (
          <span
            key={i}
            className={`marker ${w.issue.severity}`}
            style={{ left: `${(tempo.beatToSeconds(w.issue.beat) / duration) * 100}%` }}
            title={`${w.issue.rule}: ${w.issue.message}`}
          />
        ))}
        <div className="playhead" ref={head} />
      </div>
      <span className="time" ref={time} />
    </div>
  );
}
