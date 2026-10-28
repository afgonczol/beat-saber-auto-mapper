// three.js track preview: notes, bombs, walls, arcs, chains, 90/360 rotation and simple lights.
// The scene is driven by a clock function (the <audio> element's currentTime), so it stays
// in sync with playback, seeking and pausing without any state of its own.
import * as THREE from "three";
import type { Beatmap, DifficultyMap, Lightshow } from "../api/client";
import { LightRig } from "./LightRig";
import { TempoMap, lowerBound } from "./timing";

const LANE_W = 0.6;
const ROW_H = 0.55;
const ROW_BASE = 0.55;
const NOTE = 0.42;
const MAX_NOTES = 600;
const MAX_WALLS = 120;
const BEHIND_S = 0.25; // bombs/walls stay visible this long after passing the player
const FAR_Z = 42; // metres of track shown ahead

export const SABER_COLORS = [new THREE.Color("#ff2e4d"), new THREE.Color("#2f8bff")];
// Rotation (degrees, around the note's face normal) that turns a down arrow into each direction.
const DIR_ANGLE = [180, 0, -90, 90, -135, 135, -45, 45, 0];
const DIR_VEC: [number, number][] = [
  [0, 1], [0, -1], [-1, 0], [1, 0], [-0.71, 0.71], [0.71, 0.71], [-0.71, -0.71], [0.71, -0.71], [0, 0],
];

const lane = (x: number) => (x - 1.5) * LANE_W;
const row = (y: number) => ROW_BASE + y * ROW_H;
const rad = THREE.MathUtils.degToRad;

interface TimedNote { t: number; x: number; y: number; color: number; angle: number; dot: boolean; yaw: number }
interface TimedBomb { t: number; x: number; y: number; yaw: number }
interface TimedWall { t: number; t1: number; x: number; y: number; w: number; h: number; yaw: number }
interface TimedArc { t: number; t1: number; yaw: number; mesh: THREE.Mesh }
interface TimedLink { t: number; x: number; y: number; color: number; angle: number; yaw: number }

export class TrackScene {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(55, 1, 0.05, 200);
  private track = new THREE.Group();
  private lights = new LightRig();
  private resize: ResizeObserver;

  private noteMesh: THREE.InstancedMesh;
  private arrowMesh: THREE.InstancedMesh;
  private dotMesh: THREE.InstancedMesh;
  private bombMesh: THREE.InstancedMesh;
  private wallMesh: THREE.InstancedMesh;
  private linkMesh: THREE.InstancedMesh;
  private arcGroup = new THREE.Group();
  private arcMaterials = SABER_COLORS.map(
    (c) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.85 }),
  );

  private notes: TimedNote[] = [];
  private bombs: TimedBomb[] = [];
  private walls: TimedWall[] = [];
  private arcs: TimedArc[] = [];
  private links: TimedLink[] = [];
  private rotations: { t: number; yaw: number }[] = [];
  private njs = 16;
  private spawnS: number | null = null; // like the game: notes appear this long before their time
  private camYaw = 0;
  private lastFrame = performance.now();

  private m = new THREE.Matrix4();
  private tmp = new THREE.Matrix4();
  private v = new THREE.Vector3();

  constructor(
    private container: HTMLElement,
    private clock: () => number,
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.scene.background = new THREE.Color("#05060a");
    this.scene.fog = new THREE.Fog("#05060a", 18, 70);

    this.scene.add(new THREE.HemisphereLight("#b8c4ff", "#202030", 1.4));
    const key = new THREE.DirectionalLight("#ffffff", 1.6);
    key.position.set(1, 3, 4);
    this.scene.add(key);
    this.scene.add(this.track, this.arcGroup, this.lights.group);
    this.buildTrack();

    const box = new THREE.BoxGeometry(NOTE, NOTE, NOTE);
    this.noteMesh = this.instanced(box, new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1 }), MAX_NOTES);
    const arrow = new THREE.ShapeGeometry(
      new THREE.Shape([new THREE.Vector2(-0.15, 0.06), new THREE.Vector2(0.15, 0.06), new THREE.Vector2(0, -0.1)]),
    );
    arrow.translate(0, 0, NOTE / 2 + 0.003);
    this.arrowMesh = this.instanced(arrow, new THREE.MeshBasicMaterial({ color: "#ffffff" }), MAX_NOTES);
    const dot = new THREE.CircleGeometry(0.07, 20);
    dot.translate(0, 0, NOTE / 2 + 0.003);
    this.dotMesh = this.instanced(dot, new THREE.MeshBasicMaterial({ color: "#ffffff" }), MAX_NOTES);
    this.bombMesh = this.instanced(
      new THREE.IcosahedronGeometry(0.2, 1),
      new THREE.MeshStandardMaterial({ color: "#3a3a44", roughness: 0.3, metalness: 0.8, flatShading: true }),
      MAX_NOTES,
    );
    this.wallMesh = this.instanced(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({
        color: "#ff3a5a",
        emissive: "#ff1a3a",
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.32,
        depthWrite: false,
      }),
      MAX_WALLS,
    );
    this.linkMesh = this.instanced(
      new THREE.BoxGeometry(NOTE, NOTE * 0.28, NOTE * 0.8),
      new THREE.MeshStandardMaterial({ roughness: 0.35 }),
      MAX_NOTES,
    );

    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(container);
    this.fit();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  /** Show objects only from their in-game spawn (reaction time, seconds), or as far as the
   * track reaches (null). */
  setSpawn(seconds: number | null): void {
    this.spawnS = seconds;
  }

  setMap(beatmap: Beatmap, dmap: DifficultyMap, lightshow: Lightshow): void {
    const tempo = new TempoMap(beatmap);
    const sec = (b: number) => tempo.beatToSeconds(b);
    this.njs = dmap.njs || 16;

    let yaw = 0;
    this.rotations = [...(dmap.rotations ?? [])]
      .sort((a, b) => a.beat - b.beat)
      .map((r) => ({ t: sec(r.beat), yaw: (yaw += r.degrees) }));
    const yawAt = (t: number) => {
      const i = lowerBound(this.rotations, t + 1e-4, (r) => r.t) - 1;
      return i >= 0 ? rad(-this.rotations[i].yaw) : 0;
    };

    this.notes = (dmap.notes ?? [])
      .map((n) => {
        const t = sec(n.beat);
        return {
          t, x: n.x, y: n.y, color: n.color, dot: n.direction === 8, yaw: yawAt(t),
          angle: rad(DIR_ANGLE[n.direction] ?? 0) + rad(n.angle_offset ?? 0),
        };
      })
      .sort((a, b) => a.t - b.t);
    this.bombs = (dmap.bombs ?? []).map((b) => ({ t: sec(b.beat), x: b.x, y: b.y, yaw: yawAt(sec(b.beat)) }));
    this.bombs.sort((a, b) => a.t - b.t);
    this.walls = (dmap.obstacles ?? [])
      .map((o) => ({
        t: sec(o.beat), t1: sec(o.beat + o.duration), x: o.x, y: o.y, w: o.width, h: o.height,
        yaw: yawAt(sec(o.beat)),
      }))
      .sort((a, b) => a.t - b.t);

    this.links = [];
    for (const c of dmap.chains ?? []) {
      const t0 = sec(c.beat);
      const t1 = sec(c.tail_beat);
      const n = Math.max(2, c.slice_count);
      const angle = rad(DIR_ANGLE[c.direction] ?? 0);
      for (let i = 1; i < n; i++) {
        const f = (i / (n - 1)) * (c.squish || 1);
        this.links.push({
          t: t0 + (t1 - t0) * (i / (n - 1)),
          x: c.x + (c.tail_x - c.x) * f, y: c.y + (c.tail_y - c.y) * f,
          color: c.color, angle, yaw: yawAt(t0),
        });
      }
    }
    this.links.sort((a, b) => a.t - b.t);

    for (const a of this.arcs) a.mesh.geometry.dispose();
    this.arcGroup.clear();
    this.arcs = (dmap.arcs ?? []).map((a) => {
      const t = sec(a.beat);
      const t1 = sec(a.tail_beat);
      const len = (t1 - t) * this.njs;
      const [hx, hy] = DIR_VEC[a.direction] ?? [0, 0];
      const [tx, ty] = DIR_VEC[a.tail_direction] ?? [0, 0];
      const p0 = new THREE.Vector3(lane(a.x), row(a.y), 0);
      const p3 = new THREE.Vector3(lane(a.tail_x), row(a.tail_y), -len);
      const p1 = p0.clone().add(new THREE.Vector3(hx, hy, 0).multiplyScalar(0.9 * (a.head_multiplier ?? 1))).setZ(-len / 3);
      const p2 = p3.clone().sub(new THREE.Vector3(tx, ty, 0).multiplyScalar(0.9 * (a.tail_multiplier ?? 1))).setZ((-2 * len) / 3);
      const geo = new THREE.TubeGeometry(new THREE.CubicBezierCurve3(p0, p1, p2, p3), 32, 0.035, 6);
      const mesh = new THREE.Mesh(geo, this.arcMaterials[a.color] ?? this.arcMaterials[0]);
      mesh.matrixAutoUpdate = false;
      this.arcGroup.add(mesh);
      return { t, t1, yaw: yawAt(t), mesh };
    });

    this.lights.setLightshow(lightshow, sec);
  }

  dispose(): void {
    this.renderer.setAnimationLoop(null);
    this.resize.disconnect();
    for (const a of this.arcs) a.mesh.geometry.dispose();
    this.scene.traverse((o) => {
      if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) o.geometry.dispose();
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // --- per frame -------------------------------------------------------------------------

  private frame(): void {
    const now = this.clock();
    const wall = performance.now();
    const dt = Math.min(0.1, (wall - this.lastFrame) / 1000);
    this.lastFrame = wall;
    const ahead = Math.min(FAR_Z / this.njs, this.spawnS ?? Infinity);
    const z = (t: number) => -(t - now) * this.njs;

    // Notes are "cut" at the player; bombs and walls fly past.
    let n = 0;
    let arrows = 0;
    let dots = 0;
    for (let i = lowerBound(this.notes, now - 0.02, (x) => x.t); i < this.notes.length && n < MAX_NOTES; i++) {
      const note = this.notes[i];
      if (note.t > now + ahead) break;
      this.place(note.yaw, lane(note.x), row(note.y), z(note.t), note.angle);
      this.noteMesh.setMatrixAt(n, this.m);
      this.noteMesh.setColorAt(n, SABER_COLORS[note.color] ?? SABER_COLORS[0]);
      if (note.dot) this.dotMesh.setMatrixAt(dots++, this.m);
      else this.arrowMesh.setMatrixAt(arrows++, this.m);
      n++;
    }
    this.commit(this.noteMesh, n, true);
    this.commit(this.arrowMesh, arrows);
    this.commit(this.dotMesh, dots);

    n = 0;
    for (let i = lowerBound(this.links, now - 0.02, (x) => x.t); i < this.links.length && n < MAX_NOTES; i++) {
      const l = this.links[i];
      if (l.t > now + ahead) break;
      this.place(l.yaw, lane(l.x), row(l.y), z(l.t), l.angle);
      this.linkMesh.setMatrixAt(n, this.m);
      this.linkMesh.setColorAt(n++, SABER_COLORS[l.color] ?? SABER_COLORS[0]);
    }
    this.commit(this.linkMesh, n, true);

    n = 0;
    for (let i = lowerBound(this.bombs, now - BEHIND_S, (x) => x.t); i < this.bombs.length && n < MAX_NOTES; i++) {
      const b = this.bombs[i];
      if (b.t > now + ahead) break;
      this.place(b.yaw, lane(b.x), row(b.y), z(b.t), now * 2 + i);
      this.bombMesh.setMatrixAt(n++, this.m);
    }
    this.commit(this.bombMesh, n);

    n = 0;
    for (const w of this.walls) {
      if (w.t > now + ahead) break;
      if (w.t1 < now - BEHIND_S || n >= MAX_WALLS) continue;
      const bottom = w.y === 0 ? 0 : 0.1 + w.y * 0.45;
      const height = Math.max(0.1, w.h * 0.5);
      const zNear = Math.min(z(w.t), 0.6);
      const zFar = Math.max(z(w.t1), -FAR_Z - 10);
      this.place(w.yaw, lane(w.x) - LANE_W / 2 + (w.w * LANE_W) / 2, bottom + height / 2, (zNear + zFar) / 2, 0);
      this.m.scale(this.v.set(Math.max(0.05, w.w * LANE_W - 0.04), height, Math.max(0.02, zNear - zFar)));
      this.wallMesh.setMatrixAt(n++, this.m);
    }
    this.commit(this.wallMesh, n);

    for (const a of this.arcs) {
      a.mesh.visible = a.t1 >= now && a.t <= now + ahead;
      if (a.mesh.visible) {
        this.place(a.yaw, 0, 0, z(a.t), 0);
        a.mesh.matrix.copy(this.m);
        a.mesh.matrixWorldNeedsUpdate = true;
      }
    }

    // Camera follows the 90/360 rotation, eased.
    const i = lowerBound(this.rotations, now + 0.4, (r) => r.t) - 1;
    const target = i >= 0 ? rad(-this.rotations[i].yaw) : 0;
    this.camYaw += (target - this.camYaw) * Math.min(1, dt * 4);
    this.camera.position.set(0, 1.6, 2.4).applyAxisAngle(THREE.Object3D.DEFAULT_UP, this.camYaw);
    this.camera.lookAt(this.v.set(0, 0.9, -10).applyAxisAngle(THREE.Object3D.DEFAULT_UP, this.camYaw));
    this.track.rotation.y = this.camYaw;
    this.lights.group.rotation.y = this.camYaw;

    const glow = this.lights.update(now, dt);
    const tint = (c: number) => Math.min(0.06, c * 0.025);
    (this.scene.background as THREE.Color).setRGB(0.02 + tint(glow.r), 0.024 + tint(glow.g), 0.04 + tint(glow.b));
    this.renderer.render(this.scene, this.camera);
  }

  /** this.m = rotateY(yaw) * translate(x, y, z) * rotateZ(spin) */
  private place(yaw: number, x: number, y: number, z: number, spin: number): void {
    this.m.makeRotationY(yaw);
    this.m.multiply(this.tmp.makeTranslation(x, y, z));
    if (spin) this.m.multiply(this.tmp.makeRotationZ(spin));
  }

  private commit(mesh: THREE.InstancedMesh, count: number, colors = false): void {
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (colors && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  private instanced(geo: THREE.BufferGeometry, mat: THREE.Material, max: number): THREE.InstancedMesh {
    const mesh = new THREE.InstancedMesh(geo, mat, max);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.count = 0;
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return mesh;
  }

  private buildTrack(): void {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(4 * LANE_W + 0.4, FAR_Z + 6),
      new THREE.MeshStandardMaterial({ color: "#11131c", roughness: 0.9 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0, -0.01, -(FAR_Z + 6) / 2 + 3);
    this.track.add(floor);
    const pts: number[] = [];
    for (let i = 0; i <= 4; i++) {
      const x = (i - 2) * LANE_W;
      pts.push(x, 0, 2, x, 0, -FAR_Z);
    }
    pts.push(-2 * LANE_W, 0, 0, 2 * LANE_W, 0, 0); // the player's line
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    this.track.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: "#2a3048" })));
  }

  private fit(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
