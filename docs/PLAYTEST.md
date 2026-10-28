# Playtest checklist

Human playtesting is the real quality bar (PLAN §5 WS10). The offline metrics in
[EVAL.md](EVAL.md) say whether a change moved the numbers; this checklist says whether the
map is fun. Play every map on the target setup: PC Beat Saber 1.45.3 over Air Link,
unmodded unless the test needs Chroma.

## Before playing

- [ ] Generate with the build under test and note the command, commit and checkpoint:
  `bsmapper generate song.mp3 --install` (add `--model`, `--style`, `--seed` as used).
- [ ] Note the CLI's per-difficulty line: NPS, NJS/offset, rule errors and warnings. In the
  web UI, the table under the track has the same numbers, and **Settings used** the options.
- [ ] Pick at least one song you know well, so you can tell when the map ignores the music.
- [ ] Play Hard, Expert and Expert+ in that order. Easy and Normal are low priority: a quick
  check that they load and are calmer is enough.

## Per map

### 1. Load and sync
- [ ] The map shows up in Custom Levels with the right title, artist, cover and difficulties.
- [ ] The preview plays the energetic part of the song.
- [ ] The first notes land on the music; nothing drifts by the end of the song (listen at the
  last chorus).
- [ ] No notes in the first second or after the music ends.

### 2. Parity and flow
- [ ] Each hand alternates forehand and backhand without thinking about it.
- [ ] No forced resets (two downswings in a row with no time or bomb to reset).
- [ ] No wrist-breaking angles or hand crossovers you didn't expect.
- [ ] Stacks, sliders and windows read as one swing.

### 3. Vision blocks
- [ ] No note hides behind a centre note or bomb right in front of you.
- [ ] Walls never cover notes you have to hit.
- [ ] There is always somewhere to stand: walls never close both centre lanes (lean left
      and lean right) at once or in quick succession (`wall_boxin`).

### 4. Density and challenge, per difficulty
- [ ] Each difficulty is clearly harder than the one below it.
- [ ] It feels like that label on a typical human map. Write down "about right", "one step
  too easy" or "one step too hard" per difficulty (compare with `step offset` in the eval).
- [ ] Density follows the music: busy in drops and choruses, calmer in verses and breaks.
- [ ] No long empty stretches over loud music (on Expert and Expert+, nothing over ~2 beats).

### 5. Repetition on choruses
- [ ] The second chorus reuses the first chorus's patterns (same, mirrored or the same rhythm),
  like a human mapper would.
- [ ] It is not copy-paste everywhere: a verse that sounds different gets different patterns.

### 6. Reaction time
- [ ] Notes appear early enough to read the pattern (the 2026-10-22 playtest found Expert+ at
  0.6 s rushed; the default is now 0.70 s at NJS 17).
- [ ] Not so early that the track looks cluttered or floaty.
- [ ] Note the NJS / offset that felt right if the default didn't.

### 7. Lighting readability
- [ ] Lights follow the beat and the big hits; section changes are visible.
- [ ] The colours stay readable against the red/blue notes; no long blackouts in loud parts.
- [ ] Flashes are not so frequent or bright that they tire the eyes.

### 8. Laser blocking
- [ ] Lasers and ring lights never sweep across the note lanes so they hide notes (PR #28's
  rest pose and 30° sweep cap should hold).
- [ ] Nothing stays stuck in a blocking position after a drop.

### 9. Arcs and chains (Hard and up)
- [ ] Arcs join two swings of the same hand and follow the cut you make anyway; none starts
  or ends in empty air.
- [ ] There are enough to notice, but not on every note (human maps: about 15–25 a minute).
- [ ] Chains (Expert and up) start on a note and their links follow the cut.

### 10. One Saber, 90° and 360°
- [ ] One Saber: only blue notes, and the patterns fit one hand (no rushed back-and-forth).
- [ ] 90°/360°: turns come with the music, never snap the track around in one go, and give
  time to look before the next notes.
- [ ] 90° stays in front of you (within about 45° either way).
- [ ] Walls don't appear in a lane right as the track turns into it.

## Afterwards

Add one row to the playtest log in [PLAN.md](PLAN.md) (§6, "Playtest log"):

| Date | Map | Findings | Follow-up |
|---|---|---|---|
| YYYY-MM-DD | Song (checkpoint, lights, style) | what felt wrong, per difficulty | issue / PR |

Link the eval summary of the same build if there is one, so the numbers and the feel can be
compared over time.
