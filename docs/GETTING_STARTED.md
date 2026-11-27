# Getting started

bsmapper turns a song file into a Beat Saber map folder: Easy through Expert+, notes from the
trained model (note_v1) and a lightshow, ready to drop into `CustomLevels`.

> **State of things:** notes are good, and arcs are on from Hard up. Lighting is still being
> reworked (it can jitter). One Saber, 90° and 360° pass every rule check but still need a
> playtest. See [Known limitations](#known-limitations).

## 1. Open a terminal in the repo

All commands run from the repo folder, because the model checkpoint is looked up at
`checkpoints/note_v1/best.pt` relative to it.

```powershell
cd G:\ClaudeProjects\beat-saber-auto-mapper
.\.venv\Scripts\Activate.ps1
bsmapper --help
```

Lights in group-lighting environments re-use human lighting effects. Their library lives at
`checkpoints\light_effects` and is built once from the dataset, which takes about 5 minutes.
It is already built on this PC. To rebuild it:

```powershell
bsmapper lights effects data\processed_v1
```

If PowerShell refuses to run `Activate.ps1`, either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call
`.\.venv\Scripts\bsmapper.exe` directly.

## 2. Make a map from the command line

```powershell
bsmapper generate "C:\Music\My Song.mp3" --install
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
| `--arcs` / `--chains` | Arcs are on from Hard up and chains from Expert up. `--no-arcs` / `--no-chains` turn them off; `--arcs` / `--chains` turn them on in every difficulty. | `--no-chains` |
| `--difficulties` | Which difficulties to make. | `--difficulties Hard,Expert,Expert+` |
| `--modes` | Characteristics to make: Standard, OneSaber, 90Degree, 360Degree (NoArrows works too). | `--modes Standard,OneSaber,360Degree` |
| `--reaction-time` | Seconds from spawn to strike (current: Expert 0.80, Expert+ 0.70). | `--reaction-time Expert+=0.75` |
| `--njs` | Note jump speed. | `--njs Expert+=17` |
| `--lighting` | `v3` (default; works unmodded), `vanilla`, or `chroma` (needs the Chroma mod). | `--lighting vanilla` |
| `--light-style` | The look of the model's lights, with a `light_v2` checkpoint: `auto` (picked for the song; another seed may pick another), `pulse`, `calm`, `scenes`, `flash`, `accent`, or `none` (the model's own choice). | `--light-style calm` |
| `--lighting-source` | `model` or `rules`. If the model's lights are too jittery for a song, `rules` gives a calmer, simpler show. | `--lighting-source rules` |
| `--seed` | A different seed gives a different map for the same song. | `--seed 7` |
| `--title` / `--artist` | The song name the game shows, instead of the file's tags or name (one song at a time). | `--title "Stronger" --artist "Kanye West"` |
| `--environment` | The Info.dat environment. With the human effect library (below) the default is one of its environments, picked for the song and the seed. | `--environment LizzoEnvironment` |
| `--no-light-effects` | Use the old fixed group-lighting template instead of human effects. | `--no-light-effects` |

Run `bsmapper generate --help` for everything.

### Many songs at once

Give `generate` several files, or a folder. It loads the model once and makes the songs one
after another with the same options:

```powershell
# Every song in a folder (mp3, ogg, egg, wav, flac, m4a, aac, opus; subfolders are skipped)
bsmapper generate "C:\Music\To map" --install

# A few files, Expert and Expert+ only
bsmapper generate "song one.mp3" "song two.ogg" --difficulties Expert,Expert+ --install
```

- If a song fails, the rest still run. A summary table at the end lists each song, whether it
  worked, how long it took, and where the map went (or the error).
- With `--install`, each map goes into the game as soon as it is done.
- The command exits with code 1 if any song failed. Ctrl+C stops the batch, and the summary
  still lists the songs that finished.

### Typical recipes

```powershell
# Your usual: all difficulties (arcs from Hard up), installed to the game
bsmapper generate "song.mp3" --install

# Harder and more technical Expert / Expert+ only
bsmapper generate "song.mp3" --difficulties Expert,Expert+ --intensity 0.95 --style 0.7 --install

# Standard plus One Saber and 360°, Hard to Expert+
bsmapper generate "song.mp3" --modes Standard,OneSaber,360Degree --difficulties Hard,Expert,Expert+ --install

# Don't like this version? Re-roll it
bsmapper generate "song.mp3" --seed 2 --install

# Calmer lights
bsmapper generate "song.mp3" --lighting-source rules --install
```

## 3. Or use the web UI

The easiest way: **double-click `bsmapper-web.cmd`** in the repo folder (or pin a shortcut to
it). It starts the server and opens <http://127.0.0.1:8765> in your browser after a few
seconds. Close its window to stop the server. From a terminal, the same thing is:

```powershell
bsmapper serve
```

In the page:
1. Drop in a song. For a single song you can type the **Title** and **Artist** the game
   should show; left empty, they come from the file's tags (or an "Artist - Title" name).
2. Pick difficulties, modes, lighting, intensity, style, repetition, arcs and chains.
   With v3 lighting, **Environment** picks the game environment (its look and colours).
   **Auto** picks one for each song; "few maps" marks environments whose light
   effects come from fewer than 30 human maps. The page remembers your options for next
   time (**Reset** goes back to the standard ones), and "default" shows the value from your
   config file, e.g. *default (0.80)* for intensity.
3. Watch the progress. The browser tab's title shows it too, so you can switch away.
4. Preview the map in 3D, synced to the music:
   - **Space** plays and pauses, **← / →** jump 5 seconds, **Shift + ← / →** one beat,
     **Home** goes back to the start. **0.5× / 0.75×** slow the song down to look at fast
     patterns.
   - **Spawn like the game** shows notes only from when they appear in the game (the
     reaction time), so you can judge whether it is comfortable. Untick it to see further
     down the track.
   - The table under the track compares every difficulty: notes, NPS, NJS, reaction time and
     warnings. Click a row to switch to it. The difficulty tabs show their warning counts.
   - **Settings used** lists the options that made this map, so a good one can be made again.
5. Click **Download** or **Install to game**. You can regenerate a single difficulty with
   a new seed.
6. **Edit** next to the song title fixes its title and artist (for example to drop
   "(Official Video)"). The zip is renamed and, if the map was installed, the game copy is
   replaced. Nothing is generated again.
7. **Make again with other options…** queues the whole song again, without uploading it,
   starting from this map's options: change the seed, intensity, difficulties or anything
   else. The earlier map stays in the queue to compare.

### Queue lots of songs

1. Drop several songs (or a whole folder) on the drop area, or click it and multi-select.
   **Choose a folder…** picks a folder. Files that are not audio are skipped, and the ✕
   next to a song takes it out of the batch.
2. Set the options once. Every song in the batch gets the same ones.
3. Tick **Install to game when done** if you want to leave it running. Each map is copied
   into `CustomLevels` as soon as it is finished.
4. Click **Queue N songs**. The songs upload and then run one at a time on the GPU.

The **Queue** panel at the top of the start page shows each song: waiting, running (with its
stage and progress), done, failed (with the error) or cancelled.
- **Cancel** stops the running song at its next step. The ✕ takes a waiting song off the
  queue, or a finished one off the list. **Cancel waiting** and **Clear finished** do the
  same for all of them. Clearing deletes the server's copy, but installed maps stay in the
  game.
- Click a song's name or **Open** to get its normal result page: preview, **Download** and
  **Install to game**. The **.zip** button downloads it straight from the queue, and
  **Install** copies it into the game. **Install all** installs every finished map that is
  not in the game yet. Songs show their title and artist once their map exists.
- **Again…** makes a finished song again with other options (see step 7 above). A failed
  song has **Retry** (same options) and **Edit…** (other options); **Retry failed** retries
  all of them.
- The header line says how long the rest of the queue should take, from how long the songs
  so far took.
- While you are on a song's page, the header button (**Queue (3 left) · new songs**) goes
  back to the queue.
- The queue is saved on disk (`out\server`), so reloading the page, closing it, or even
  restarting the server loses nothing: finished maps come back, and songs that were waiting
  or running when the server stopped are queued again. **Clear finished** (or ✕) is how a
  song leaves the list for good.

A full song with all five Standard difficulties takes about a minute and a half on the 3090.
The UI is already built (`web\dist`). After pulling web changes, rebuild it with
`cd web; npm install; npm run build`.

The beat tracker `beat_this` gives a better beat grid than the librosa fallback. It is not
in the pip extras (it installs from GitHub). If generation logs say it fell back to librosa:

```powershell
pip install "beat-this @ https://github.com/CPJKU/beat_this/archive/main.zip"
```

## 4. Your settings file

`C:\Users\<you>\.bsmapper\config.toml` holds your defaults. Command-line options override
it. For example:

```toml
[generate]
intensity = 0.8
# repetition = 0.5
# light_style = "auto"   # or pulse, calm, scenes, flash, accent, none (light_v2 checkpoints)
# light_effects = 'G:\elsewhere\light_effects'   # default: checkpoints/light_effects
# arcs = false     # true: arcs in every difficulty; false: none; unset: from Hard up
# chains = false   # the same for chains (unset: from Expert up)
# custom_levels = 'C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels'
```

## 5. Playtest and give feedback

[PLAYTEST.md](PLAYTEST.md) has a checklist: sync, flow, vision blocks, difficulty,
repetition, reaction time and lights. Notes like "Expert felt one step too easy" or "the
lights jitter around 1:40" turn directly into fixes.

After playing a map made in the web UI, open its page and use the **Playtest** box:
**👍 Good** or **👎 Needs work**, plus a note. It saves on its own and stays with the map
(the queue shows the thumb). To read every note later, with the options that made each map:

```powershell
bsmapper feedback
```

## Known limitations

- **Lighting:** being redesigned. The current model lights can change too often in busy
  sections. `--lighting-source rules` is a calmer fallback.
- **One Saber / 90° / 360°** pass every rule check (single colour, no rotation whiplash or
  90° snaps, 90° stays within ±45°), but haven't been playtested yet. On some songs
  (Stronger) they come out denser than human maps of the mode, like Standard does.
- **Tempo:** songs with tempo changes or a half/double-time ambiguity can get a wrong beat
  grid. If notes feel off-beat, report the song.
- **Easy** is the weakest difficulty (on purpose, it's low priority).
- **GPU:** while a model is training (`checkpoints\note_*\train.pid` alive), generation is
  slower, and VR gameplay on the same GPU can stutter.
- The model weights live only on this PC (`checkpoints\`, not in git). Back them up if you
  care about them.
