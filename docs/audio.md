# Audio analysis (WS2)

`bsmapper.audio.Analyzer` implements `interfaces.AudioAnalyzer`:

```python
from bsmapper.audio import Analyzer

result = Analyzer().analyze_full("song.egg")  # .analysis is the AudioAnalysis contract
result.grid  # BeatGrid: bpm, offset_s, bpm_changes, beats_per_bar
result.fit.coverage  # share of tracked beats on the grid (confidence)
```

## Pipeline

1. **Decode** (`decode.py`): ffmpeg to mono float32 at 22.05 kHz, with soundfile + soxr as
   the fallback. ffmpeg comes first because WS1 uses it to write `song.egg`, so we
   analyze the same timeline the game plays.
2. **Track beats** (`beats.py`): beat_this (default, needs `torch` + `beat-this`) or
   librosa. beat_this predicts at 50 fps, so its peaks are refined with parabolic
   interpolation of the logits.
3. **Fit a grid** (`grid.py`): a brute-force "comb" search over period and phase finds the
   constant tempo that puts the most tracked beats within 25 ms of the grid, and least squares
   then refines it. BPM is snapped to the roundest value (integer, .5, .1, .01) that stays
   within 5 ms of the free fit over the whole song. BPM changes are introduced only when fewer
   than 80% of tracked beats fit one tempo and a continuous piecewise fit covers at least 15
   percentage points more. Beat 0 is the first downbeat at or after 0 s, with the bar phase
   voted by the tracker's downbeats.
4. **Refine the phase** (`grid.refine_offset`): shift the whole grid (±40 ms, 1 ms
   steps) to where the onset envelope summed over all beats peaks, minus a calibrated
   envelope latency (`ONSET_LATENCY_S`, 12 ms). Per-beat tracker timing has a
   style-dependent bias of ±10 ms; the envelope averaged over hundreds of beats does not.
5. **Features** (`features.py`, hop 220 = 100.2 fps): `mel` (80 bins, dB relative to the
   song's max, clipped at -80), `onset_strength`, `spectral_flux` and `rms` (each scaled to
   its song max), plus optional HPSS `onset_percussive` / `rms_harmonic`.
   **Beat-synchronous** copies at 24 ticks/beat use linear interpolation on the grid's
   tick times: row `j` is beat `j / 24`.
6. **Sections** (`sections.py`): Foote novelty on a beat-level self-similarity
   matrix (mel + loudness + onset activity), with boundaries snapped to bar lines and at
   least 4 bars apart. Energy is 0.7·loudness + 0.3·onset activity, relative to the song's
   loudest section. Labels (intro/verse/chorus/drop/break/outro) are an energy and position
   heuristic, so treat them as hints.

### Invariants of `AudioAnalysis`
- `beat_times[i]` is grid beat `i`, `beat_times[0]` is the offset, and beat 0 is a
  downbeat. `downbeat_times = beat_times[::beats_per_bar]`.
- `bpm` is the tempo at beat 0. With BPM changes, `beat_times` follows them.
- `beat_features[name].shape[0] == len(beat_times) * 24`.
- `sections` tile `[0, duration_s]` with no gaps.

### Beat 0 vs. the audio file
Beat Saber puts beat 0 at the start of the audio. `BeatGrid.lead_in_s()` is the
silence to prepend to `song.egg` so that the bar before our beat 0 lands at 0 s. After
padding, map beat = grid beat + `beats_per_bar`, and the map stays on an integer grid.
This is a decision for WS1 and WS7.

## Tracker comparison

`python -m bsmapper.audio.benchmark <CustomLevels> [--no-refine]` scores both trackers against
human-mapped maps. The test set is 60 single-BPM maps from a local CustomLevels folder
(2018–2025, many mappers). The reference is the map's BPM grid with its phase taken from the
notes, because many older maps leave beat 0 off the music and offset every note instead.
The metric is where our grid beats land on that reference, in ticks (1/24 beat). A map
"passes" when ≥ 90% of our beats are within ±1 tick. Tempo-octave answers (½× or 2× the
mapper's BPM) are scored on the finer grid. "Trusted" maps (47) are those whose notes sit on
their own BPM grid.

| tracker | phase refinement | pass (all 60) | pass (47 trusted) | BPM or octave right (trusted) | median \|err\| (trusted) | time / song |
|---|---|---|---|---|---|---|
| **beat_this** | **yes (default)** | **43 (72%)** | **36 (77%)** | **44 (94%)** | **0.20 tick** | 0.8 s |
| beat_this | no | 40 (67%) | 35 (74%) | 44 (94%) | 0.56 tick | 0.8 s |
| librosa | yes | 36 (60%) | 29 (62%) | 38 (81%) | 0.22 tick | 1.5 s |
| librosa | no | 3 (5%) | 3 (6%) | 38 (81%) | 1.93 tick | 1.5 s |

(RTX 3090, including decoding and features. beat_this also runs on CPU in a few seconds per song.)

**Decision: beat_this**, with librosa as the fallback when torch or beat-this is not installed.
- beat_this gets the tempo (or its octave) right much more often (94% vs 81%). librosa
  more often locks onto 2/3 or 3/2 of the tempo (for example 113 instead of 170) or onto the
  off-beat, and it falls back to BPM changes on clean songs.
- Raw librosa beats are about 30 ms late, which is more than one tick. Raw beat_this beats
  have a style-dependent bias of about ±10 ms. The onset-envelope phase refinement removes
  both, so the remaining gap between trackers is getting the tempo and beat phase right.
- Envelope latency was calibrated on these maps: 12 ms on human maps vs. 8 ms on an ideal
  synthetic click. Human mappers place the beat within a few ms of the true onset.

**Where beat_this still fails** (17 maps):
- Six maps have tempo drift or a tempo the tracker misreads: Rasputin, Bulls on Parade,
  Brujeria, La Vida Es Un Carnaval, Meant to Live, and Angst (read at 3/2 tempo). Most are
  live or unquantized recordings.
- Two maps have notes free-timed against a nominal BPM: Pump It (< 6% of notes on its own
  1/4 grid) and It's Tricky. Here the reference is the problem, and WS3 should reject such
  maps.
- Six older hand-timed maps are off by a constant 1–1.5 ticks (500 Miles, Bye Bye Bye, Power
  Rangers, Sk8er Boi, Gangsta's Paradise, Better Off Alone). Both trackers agree to within
  0.05 tick on these, so the map's timing is the likely culprit.
- Two are real tracker errors: on Scatman, beat_this is a quarter beat off (librosa gets it),
  and on HEYA both trackers sit on the mapper's off-beat at half tempo.

`coverage < 0.7` flags 5 of the 6 low-confidence beat_this failures. A manual BPM and offset
override (CLI or UI) is the escape hatch, as PLAN §8 says.
