# Getting started

bsmapper turns a song file into a Beat Saber map folder: Easy through Expert+, notes from the
trained model (note_v1) and a lightshow, ready to drop into `CustomLevels`.

> **State of things:** notes are good. Lighting is still being reworked (it can jitter), and
> One Saber/90°/360° are lightly tested. See [Known limitations](#known-limitations).

## 1. Open a terminal in the repo

All commands run from the repo folder, because the model checkpoint is looked up at
`checkpoints/note_v1/best.pt` relative to it.

```powershell
cd G:\ClaudeProjects\beat-saber-auto-mapper
.\.venv\Scripts\Activate.ps1
bsmapper --help
```

If PowerShell refuses to run `Activate.ps1`, either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call
`.\.venv\Scripts\bsmapper.exe` directly.

## 2. Make a map from the command line

```powershell
bsmapper generate "C:\Music\My Song.mp3" --arcs --install
```

- This takes about a minute on the RTX 3090. It writes `out\<Song> - <Artist>\` and a zip.
- `--install` copies the folder into Beat Saber's `CustomLevels`, so it shows up under
  Custom Songs (restart the game or refresh the song list).
- Re-running with the same song replaces the earlier bsmapper copy. It never overwrites a
  map someone else made.
- Any format ffmpeg reads works: mp3, ogg, wav, flac or m4a.

### Useful options

| Option | What it does | Example |
|---|---|---|
| `--intensity 0..1` | Challenge within each difficulty. 0.5 is a typical human map; 1 is about the hardest 10%. Your default is **0.8** (in the config file). | `--intensity 0.9` |
| `--style 0..1` | 0 is flowy, 1 is technical. Leave it out for no preference. | `--style 0.3` |
| `--repetition 0..1` | How much repeated music (e.g. choruses) reuses earlier patterns. 0.5 is the amount learned from human maps. | `--repetition 0.7` |
| `--arcs` / `--chains` | Turn arcs or chains on. **Arcs are off by default for now**; pass `--arcs`. | `--arcs --chains` |
| `--difficulties` | Which difficulties to make. | `--difficulties Hard,Expert,Expert+` |
| `--modes` | Characteristics to make. | `--modes Standard,OneSaber,360Degree` |
| `--reaction-time` | Seconds from spawn to strike (current: Expert 0.80, Expert+ 0.70). | `--reaction-time Expert+=0.75` |
| `--njs` | Note jump speed. | `--njs Expert+=17` |
| `--lighting` | `v3` (default; works unmodded), `vanilla`, or `chroma` (needs the Chroma mod). | `--lighting vanilla` |
| `--lighting-source` | `model` or `rules`. If the model's lights are too jittery for a song, `rules` gives a calmer, simpler show. | `--lighting-source rules` |
| `--seed` | A different seed gives a different map for the same song. | `--seed 7` |
| `--environment` | The Info.dat environment. | `--environment EDMEnvironment` |

Run `bsmapper generate --help` for everything.

### Typical recipes

```powershell
# Your usual: all difficulties, arcs on, installed to the game
bsmapper generate "song.mp3" --arcs --install

# Harder and more technical Expert / Expert+ only
bsmapper generate "song.mp3" --difficulties Expert,Expert+ --intensity 0.95 --style 0.7 --arcs --install

# Don't like this version? Re-roll it
bsmapper generate "song.mp3" --arcs --seed 2 --install

# Calmer lights
bsmapper generate "song.mp3" --arcs --lighting-source rules --install
```

## 3. Or use the web UI

```powershell
bsmapper serve
```

Then open <http://127.0.0.1:8765>:
1. Drop in a song.
2. Pick difficulties, modes, lighting, intensity, style and repetition.
3. Watch the progress.
4. Preview the map in 3D, synced to the music.
5. Click **Download** or **Install to game**. You can regenerate a single difficulty with
   a new seed.

The UI is already built (`web\dist`). After pulling web changes, rebuild it with
`cd web; npm install; npm run build`.

## 4. Your settings file

`C:\Users\<you>\.bsmapper\config.toml` holds your defaults. Command-line options override
it. For example:

```toml
[generate]
intensity = 0.8
# repetition = 0.5
# custom_levels = 'C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels'
```

## 5. Playtest and give feedback

[PLAYTEST.md](PLAYTEST.md) has a checklist: sync, flow, vision blocks, difficulty,
repetition, reaction time and lights. Notes like "Expert felt one step too easy" or "the
lights jitter around 1:40" turn directly into fixes.

## Known limitations

- **Lighting:** being redesigned. The current model lights can change too often in busy
  sections. `--lighting-source rules` is a calmer fallback.
- **Arcs** are off by default. Pass `--arcs`; a default change is coming.
- **One Saber / 90° / 360°** are generated but barely playtested.
- **Tempo:** songs with tempo changes or a half/double-time ambiguity can get a wrong beat
  grid. If notes feel off-beat, report the song.
- **Easy** is the weakest difficulty (on purpose, it's low priority).
- **GPU:** while a model is training (`checkpoints\note_*\train.pid` alive), generation is
  slower, and VR gameplay on the same GPU can stutter.
- The model weights live only on this PC (`checkpoints\`, not in git). Back them up if you
  care about them.
