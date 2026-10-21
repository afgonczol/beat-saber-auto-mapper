// Beat <-> seconds, the same math as schema.Beatmap.beat_to_seconds / seconds_to_beat.
import type { Beatmap } from "../api/client";

interface Segment {
  beat: number;
  seconds: number;
  bpm: number;
}

export class TempoMap {
  private segments: Segment[];

  constructor(beatmap: Pick<Beatmap, "bpm" | "bpm_changes">) {
    const segments: Segment[] = [{ beat: 0, seconds: 0, bpm: beatmap.bpm }];
    const changes = [...(beatmap.bpm_changes ?? [])].sort((a, b) => a.beat - b.beat);
    for (const c of changes) {
      if (c.beat <= 0) {
        segments[0] = { beat: 0, seconds: 0, bpm: c.bpm };
        continue;
      }
      const last = segments[segments.length - 1];
      segments.push({ beat: c.beat, seconds: last.seconds + ((c.beat - last.beat) * 60) / last.bpm, bpm: c.bpm });
    }
    this.segments = segments;
  }

  beatToSeconds(beat: number): number {
    const s = this.find((seg) => seg.beat <= beat);
    return s.seconds + ((beat - s.beat) * 60) / s.bpm;
  }

  secondsToBeat(seconds: number): number {
    const s = this.find((seg) => seg.seconds <= seconds);
    return s.beat + ((seconds - s.seconds) * s.bpm) / 60;
  }

  private find(pred: (s: Segment) => boolean): Segment {
    let found = this.segments[0];
    for (const s of this.segments) {
      if (pred(s)) found = s;
      else break;
    }
    return found;
  }
}

/** Index of the first item with key >= value in a sorted array. */
export function lowerBound<T>(items: T[], value: number, key: (t: T) => number): number {
  let lo = 0;
  let hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(items[mid]) < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
