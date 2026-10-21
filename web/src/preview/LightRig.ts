// A minimal environment that reacts to classic (v2-style) basic events:
// 0 back lasers, 1 ring lights, 2 left lasers, 3 right lasers, 4 center lights,
// 8 ring spin, 12/13 left/right laser speed, plus color boost. Chroma RGBA overrides color.
// v3 event box groups are not rendered yet.
import * as THREE from "three";
import type { Lightshow } from "../api/client";
import { lowerBound } from "./timing";

const BASE = { red: new THREE.Color("#ff3048"), blue: new THREE.Color("#2d8cff") };
const BOOST = { red: new THREE.Color("#ff2ae0"), blue: new THREE.Color("#20f0d0") };
const WHITE = new THREE.Color("#ffffff");
const LIGHT_TYPES = [0, 1, 2, 3, 4] as const;
const OFF_LEVEL = 0.025;
// Per-type opacity, so the big floor strips do not drown out the notes.
const OPACITY: Record<number, number> = { 0: 1, 1: 0.8, 2: 1, 3: 1, 4: 0.4 };

interface LightEvent { t: number; value: number; f: number; color: THREE.Color | null }

function brightness(e: LightEvent | undefined, dt: number): number {
  if (!e || e.value === 0) return 0;
  const kind = (e.value - 1) % 4; // 0 on, 1 flash, 2 fade, 3 transition
  const f = e.f ?? 1;
  if (kind === 1) return f * (1 + 0.9 * Math.exp(-dt * 7));
  if (kind === 2) return f * 1.6 * Math.exp(-dt * 2.4);
  return f;
}

export class LightRig {
  readonly group = new THREE.Group();
  private events = new Map<number, LightEvent[]>();
  private boosts: { t: number; on: boolean }[] = [];
  private speeds = new Map<number, { t: number; value: number }[]>();
  private spins: number[] = [];
  private materials = new Map<number, THREE.MeshBasicMaterial>();
  private leftLasers: THREE.Mesh[] = [];
  private rightLasers: THREE.Mesh[] = [];
  private rings: THREE.Mesh[] = [];
  private ringAngle = 0;
  private time = 0;
  private color = new THREE.Color();
  private glow = new THREE.Color();

  constructor() {
    for (const type of LIGHT_TYPES) {
      this.materials.set(
        type,
        new THREE.MeshBasicMaterial({
          color: "#000000",
          transparent: true,
          opacity: OPACITY[type],
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          fog: false,
        }),
      );
    }
    const mat = (t: number) => this.materials.get(t)!;
    const bar = (w: number, h: number, d: number, m: THREE.Material) =>
      new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);

    // Back lasers: a fan far behind the track.
    for (let i = 0; i < 5; i++) {
      const b = bar(0.12, 30, 0.12, mat(0));
      b.position.set((i - 2) * 6, 6, -62);
      b.rotation.z = (i - 2) * 0.28;
      this.group.add(b);
    }
    // Ring lights: square rings around the track.
    for (let i = 0; i < 4; i++) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(9, 0.08, 4, 4), mat(1));
      ring.position.set(0, 3, -22 - i * 9);
      this.rings.push(ring);
      this.group.add(ring);
    }
    // Rotating lasers on each side.
    for (const side of [-1, 1]) {
      for (let i = 0; i < 4; i++) {
        const b = bar(0.1, 26, 0.1, mat(side < 0 ? 2 : 3));
        b.position.set(side * (9 + i * 1.5), 4, -14 - i * 9);
        (side < 0 ? this.leftLasers : this.rightLasers).push(b);
        this.group.add(b);
      }
    }
    // Center lights: strips along the track edges and one across the far end.
    for (const x of [-1.6, 1.6]) {
      const s = bar(0.06, 0.04, 40, mat(4));
      s.position.set(x, 0.02, -20);
      this.group.add(s);
    }
    const end = bar(8, 0.2, 0.2, mat(4));
    end.position.set(0, 0.1, -44);
    this.group.add(end);
  }

  setLightshow(ls: Lightshow, sec: (beat: number) => number): void {
    this.events.clear();
    this.speeds.clear();
    this.spins = [];
    for (const type of LIGHT_TYPES) this.events.set(type, []);
    this.speeds.set(12, []);
    this.speeds.set(13, []);
    const sorted = [...(ls.basic_events ?? [])].sort((a, b) => a.beat - b.beat);
    for (const e of sorted) {
      const t = sec(e.beat);
      const list = this.events.get(e.type);
      if (list) {
        const c = e.color ? new THREE.Color(e.color[0], e.color[1], e.color[2]) : null;
        list.push({ t, value: e.value, f: e.float_value ?? 1, color: c });
      } else if (e.type === 12 || e.type === 13) {
        this.speeds.get(e.type)!.push({ t, value: e.value });
      } else if (e.type === 8) {
        this.spins.push(t);
      }
    }
    this.boosts = [...(ls.boost_events ?? [])].sort((a, b) => a.beat - b.beat).map((b) => ({ t: sec(b.beat), on: b.on }));
  }

  /** Update for song time `now`; returns the average light color (for ambient tint). */
  update(now: number, dt: number): THREE.Color {
    this.time += dt;
    const bi = lowerBound(this.boosts, now + 1e-4, (b) => b.t) - 1;
    const palette = bi >= 0 && this.boosts[bi].on ? BOOST : BASE;
    this.glow.setRGB(0, 0, 0);

    for (const type of LIGHT_TYPES) {
      const list = this.events.get(type) ?? [];
      const i = lowerBound(list, now + 1e-4, (e) => e.t) - 1;
      const e = i >= 0 ? list[i] : undefined;
      const level = Math.max(OFF_LEVEL, brightness(e, e ? now - e.t : 0));
      if (e?.color) this.color.copy(e.color);
      else if (!e || e.value === 0) this.color.copy(palette.blue);
      else if (e.value >= 9) this.color.copy(WHITE);
      else this.color.copy(e.value >= 5 ? palette.red : palette.blue);
      this.color.multiplyScalar(level);
      this.materials.get(type)!.color.copy(this.color);
      this.glow.add(this.color.multiplyScalar(OPACITY[type]));
    }

    const speed = (type: number) => {
      const list = this.speeds.get(type) ?? [];
      const i = lowerBound(list, now + 1e-4, (e) => e.t) - 1;
      return i >= 0 ? list[i].value : 0;
    };
    const sway = (lasers: THREE.Mesh[], s: number, dir: number) =>
      lasers.forEach((l, i) => (l.rotation.z = dir * (0.35 + 0.3 * Math.sin(now * (0.4 + s * 0.5) + i * 0.9))));
    sway(this.leftLasers, speed(12), 1);
    sway(this.rightLasers, speed(13), -1);

    const spins = lowerBound(this.spins, now + 1e-4, (t) => t);
    const target = spins * (Math.PI / 4);
    this.ringAngle += (target - this.ringAngle) * Math.min(1, dt * 3);
    this.rings.forEach((r, i) => (r.rotation.z = Math.PI / 4 + this.ringAngle * (1 + i * 0.15)));

    return this.glow.multiplyScalar(1 / LIGHT_TYPES.length);
  }
}
