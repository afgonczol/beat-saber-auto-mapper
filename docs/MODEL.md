# Note and lighting model (WS6)

Code: `src/bsmapper/models/` (network, checkpoints, sampling API) and
`src/bsmapper/training/` (dataset reader, examples, train loop). Configs: `configs/*.yaml`.
Needs the `ml` extra (`torch`, `pyarrow`, `pyyaml`, `tensorboard`).

## Architecture

One `MapperModel`, an encoder-decoder transformer in the style of Whisper / osuT5, with
two decoders that share the audio encoder:

```
 features (B, 384, 83) ─► normalize ─► conv stem (2× Conv1d k=5) ─► 6 encoder layers ─► audio memory
                                                                                          │
 BOS CHAR DIFF NPS STYLE BOMBS WALLS ARCS CHAINS <prev tail> SEP <body> EOS              │
      ─► 8 decoder layers (self-attn, cross-attn to audio memory) ─► note logits (655) ◄─┘

 BOS <prev light tail> SEP <light body> EOS
      ─► 6 decoder layers (cross-attn to [audio memory ; NoteMemory]) ─► light logits (500)
 BOS <note prefix> SEP <note body> EOS ─► NoteMemory (2 encoder layers) ──────────┘
```

- **Input**: one 16-beat window. Features are beat-synchronous (one frame per 1/24 beat,
  so 384 frames): 80 log-mel bins, onset strength, spectral flux and RMS (the WS3b
  layout). Per-feature mean/std are measured on the training set and stored in the
  model's buffers, so a checkpoint normalizes its own input.
- **Sequences** are exactly `TokenWindow.input_ids()` from the tokenizer (WS4). The loss is
  on `body EOS` only. The prefix and the previous window's tail (16 events) give the
  decoder its conditioning and its continuity across windows.
- **Size** (default config, d=512, 8 heads, FF 2048, pre-norm, learned positions, tied
  input/output embeddings):

  | part | params |
  |---|---|
  | audio encoder | 20.6M |
  | note decoder | 34.8M |
  | note memory | 7.2M |
  | light decoder | 26.3M |
  | **total** | **88.8M** (55.4M without lighting, `light_layers: 0`) |

- Teacher forcing uses `scaled_dot_product_attention(is_causal=True)` (flash attention
  in bf16). Sequences are right-padded, so this is exact for every real token.
  Incremental decoding uses a KV cache with explicit masks, which allows left-padded
  batches. A unit test checks that it matches teacher forcing.

## Training

`bsmapper train configs/note_v0.yaml [-o section.key=value ...] [--max-steps N] [--no-resume]`

- **Examples** (`training/data.py`): a note example is one (difficulty, window) pair, and
  a light example is one (map, window) pair. The light example also carries the
  note track of the map's hardest difficulty (50%) or a random difficulty.
- **Conditioning dropout**: STYLE becomes `STYLE_NONE` 20% of the time (and always when
  the map has no style label). NPS becomes `NPS_NONE` 10% of the time. The previous
  tail is dropped 5% of the time, which matches the first window at inference. This
  is what makes classifier-free guidance on STYLE work: the model also learns the
  unconditional distribution.
- **Sampling weights**: a window's weight is `difficulty_weights[diff] ×
  characteristic_weights[char] (not for Easy) × gold_weight (gold maps) ×
  empty_window_weight (no objects)`. The defaults oversample OneSaber ×2, 360° ×3 and
  90° ×4, and leave Easy at ×1 in every mode (Easy is low priority, PLAN §1).
- **Lights** train on gold maps only (`lights_gold_only`). Windows with more than 1024
  light tokens (lightID/Chroma spam, about 5% of the windows) are skipped.
- **Optimizer**: AdamW (β 0.9/0.98, wd 0.05, none on biases/norms/positions), linear
  warmup and then cosine decay to 10% of the peak, gradient clip 1.0, label smoothing 0.1,
  and bf16 autocast on CUDA. Loss = `note_loss + 0.5 × light_loss` per step.
- **Outputs** in `run.out_dir`:
  - `last.pt`, resumed automatically
  - `best.pt`, the lowest validation note loss
  - `step_*.pt`, the newest 3 kept
  - `config.yaml`
  - `tb/`, TensorBoard: `tensorboard --logdir checkpoints`
- `run.cuda_memory_fraction` (default 0.92) caps PyTorch's VRAM use. If a batch does not
  fit, you get an OOM error instead of Windows silently spilling into shared system
  memory, which crawls.
- `data.num_workers > 0` uses spawned DataLoader workers on Windows. A script that
  builds a `Trainer` itself must put that code under `if __name__ == "__main__":`,
  or the workers hang. `bsmapper train` already handles this.
- Gradient checkpointing (`model.grad_checkpointing`) and `torch.compile`
  (`run.compile`) are available. The v0 config needs neither.

### Configs

| config | what | time |
|---|---|---|
| `configs/debug.yaml` | 0.7M params on a generated toy dataset (`data.root: synthetic`), CPU | ~1.5 min |
| `configs/note_v0.yaml` | full model on `data/processed` (dataset v0), 3090, bf16 | see below |

The debug config also runs against the real data:
`-o data.root=data/processed -o run.out_dir=checkpoints/debug_real`.

### Dataset reader

`training/store.py` reads the WS3b on-disk layout directly: `manifest.parquet`,
`tokens/*.parquet`, `lights/*.parquet` and `features/*.npy`. It does not import
`bsmapper.data`. `training/synthetic.py` writes a toy dataset in the same layout.
The tests and the debug config use it.

## Sampling API (for WS7)

```python
from bsmapper.models import NoteSampler, LightSampler, SamplingOptions, load_model
from bsmapper.tokenizer import Conditioning, MapTokenizer, LightingTokenizer

model = load_model("checkpoints/note_v0/best.pt", device="cuda")
opts = SamplingOptions(temperature=0.9, top_p=0.95, seed=1, style_guidance=1.5)
conds = [Conditioning.from_request(r) for r in requests]  # decoded as one batch
windows = NoteSampler(model).generate(features, conds, opts, mask_fn=rule_mask)
dmap = MapTokenizer().decode(windows[0], meta=template)
lights = LightSampler(model).generate(features, windows[0], opts)
```

- `features`: `(frames, 83)` beat-synchronous features of the whole song, 24 frames per
  beat. They are padded with silence to whole windows (`n_windows=` overrides the
  window count).
- Every step samples from the logits restricted to `Parser.allowed_mask()` (the
  grammar), intersected with `mask_fn(StepState)` if one is given. `StepState` has
  `window`, `sequence` (index into `conds`), `body` (tokens so far) and `parser`
  (`.time`, `.events`, `.partial`). If the intersection is empty, the grammar mask
  alone is used, so a rule mask can never stall decoding.
- `temperature=0` is greedy. `top_p` is nucleus sampling. Sampling runs in NumPy with
  `seed`, so the output does not depend on the device's RNG.
- `style_guidance > 1` turns on classifier-free guidance for conditionings that have a
  style: `uncond + s·(cond − uncond)` with `STYLE_NONE` as the unconditional
  prefix. It costs one extra batch row per guided sequence.
- `max_body_tokens` caps a window's body. Past the cap only EOS may start, though an
  open object may still finish. `on_window(k, bodies)` is called after each window,
  before the bodies become the next window's context.

## Planned first full run (dataset v0)

Dataset v0 train split: 2207 maps, 6824 difficulties, 197,786 note windows and
38,632 gold light windows (≤ 1024 tokens). The p99 note body is 294 tokens and
the p99.9 is 499.

- **Config**: `configs/note_v0.yaml`. 89M parameters, batch 32 note windows + 16
  light windows, lr 3e-4 with 2k warmup steps, 60k steps.
  - That is ≈ 9.7 passes over the note windows and ≈ 25 over the light windows.
  - Evaluation and a checkpoint every 2k steps.
- **VRAM** (measured, bf16, no gradient checkpointing):
  - worst-case batch (32 × 1024 note tokens + 16 × 1024 light tokens): 14.4 GB peak
  - typical batch (32 × 512 + 16 × 1024): 10.8 GB
  - real v0 batches over 200 steps: 14.1 GB peak allocated, about 18 GB in
    nvidia-smi with the allocator cache and the desktop
- **Time** (measured):
  - real v0 batches with 4 workers: 0.46 s/step (2.17 steps/s, ≈ 17.7k target
    tokens/s). The GPU stays at 99%, so the run is not data-bound.
  - synthetic shapes: 0.37 s/step at 32 × 512 + 16 × 1024 tokens, 0.51 s/step at
    the worst case
  - 60k steps ≈ 7.7 h, plus about 15 min of evaluation (30 × 50 batches). Plan
    for an overnight run of about 8 h.
- **Sanity check** from the 200-step benchmark: val note loss 3.38 (acc 0.34) and val
  light loss 3.54, falling steadily from ln(655) ≈ 6.5 still in warmup.
- **Watch**:
  - `val/note_loss` vs `train/note_loss`. 2k songs are few, so expect overfitting.
    If validation turns up early, stop at `best.pt` and consider dropout 0.2.
  - `train/gpu_mem_gb`
  - the per-mode loss once WS10's metrics exist
- **Experiments after v0** (PLAN §5): frozen MERT/Whisper features instead of the conv
  stem, context length (8/16/32 events), audio context beyond the window, CFG on the
  difficulty token, and token time-resolution.

## note_v1: intensity and structure memory

Two findings from the note_v0 playtests drive v1 (PLAN, playtest log): every difficulty
feels one step too easy, and human maps reuse patterns when the music repeats (often
mirrored) while v0 does not. v1 adds two conditionings and trains on dataset v1 (about 4×
v0). It needs tokenizer v2 (`docs/tokenizer.md`); note_v0 keeps working unchanged.

### What changed

- **`INTENSITY` prefix token** (`INTENSITY_0..4`, `_NONE`): the map's challenge within its
  label, 0..1 (`docs/DATASET.md`, "Intensity"). 0.5 is a typical human map of that label and
  tempo; 1.0 is about the hardest 10 %. It is dropped to `INTENSITY_NONE` 15 % of the time.
  NPS is dropped more often than in v0 (25 %), so the model also learns to set density from
  the intensity alone.
- **Reference window** (structure memory): for each window, the most similar earlier,
  bar-shifted 16-beat span of the same song (chroma, mel and onset features per beat,
  `bsmapper.data.structure`). When its similarity is at least `ref_threshold` (0.85), the
  human notes of that span go into the prompt as `REF <events> REF_END`, 70 % of the time
  (`ref_dropout` 0.3), so the model learns both with and without one. The model learns from
  human maps how much to copy, mirror or vary; there is no rule for it.
- **Size**: d=512 like v0, but a 10-layer note decoder (v0: 8), 97M parameters. Sequences
  are cropped at 1024 tokens (prefix 9 + reference ≤ 256 + tail + body); a too-long prompt
  drops the reference first, then the tail.

### Sampling with v1 (for WS7)

```python
from bsmapper.data.structure import beat_chroma, find_references, structure_features
from bsmapper.models import NoteSampler, SamplingOptions, WindowRefs, repetition_guidance

# per song: per-beat structure features on the map grid, then the reference per window
chroma = beat_chroma(y, sr, beat_seconds, n_windows * 16)  # beat_seconds: map beat -> s
refs = find_references(structure_features(features, chroma), n_windows)
ref_fn = WindowRefs(refs.kept())  # threshold 0.85, as in training

opts = SamplingOptions(..., repetition_guidance=repetition_guidance(request.repetition))
conds = [Conditioning.from_request(r) for r in requests]  # carries request.intensity
windows = NoteSampler(model).generate(features, conds, opts, ref_fn=ref_fn)
```

- `NoteSampler` picks the prompt layout from the checkpoint's vocabulary size. With a
  note_v0 checkpoint it uses the 8-token prefix, ignores `intensity` and `ref_fn`, and
  decodes token for token as before (`tests/models/test_models_v0_compat.py` checks a
  recorded output of `checkpoints/note_v0/best.pt`).
- `ref_fn(window, sequence, bodies)` returns the reference events for a window from that
  sequence's already generated `bodies`, or None. `WindowRefs` does this from start ticks.
- `repetition_guidance` is classifier-free guidance between the prompt with and without
  the reference: `noref + g·(ref − noref)`. `repetition_guidance(None)` is 1.0, the learned
  behaviour, with no extra batch rows. `repetition` 0..1 maps to g = 2·repetition: 0
  ignores the reference, 0.5 is the learned amount and 1.0 doubles it. Each guided window
  costs one extra batch row, as style guidance does; both can be on together.
- `Conditioning.from_request` fills `intensity` from `GenerationRequest.intensity`.
- `bsmapper.data.structure.repetition_profile(dmap, refs.kept())` measures how a map's
  matched windows relate to their references (copy, mirror, same rhythm, different). WS10
  can compare it with the human shares in `docs/DATASET.md`.

### Run (dataset v1)

- **Config**: `configs/note_v1.yaml`. 97M parameters, batch 32 note windows + 16 light
  windows, lr 3e-4 with 2k warmup, 80k steps (about 3 passes over the v1 note windows).
- **Benchmark** on real v1 batches (200 steps, while the dataset build was still using
  the CPU and the GPU):

  | model | params | steps/s | peak VRAM |
  |---|---|---|---|
  | d=640, 10 heads, 10 decoder layers, crop 1408 | 150M | — | OOM at 22 GB |
  | d=640, 10 heads, 10 decoder layers, crop 1024 | 150M | 1.3–1.6 | 19.8 GB |
  | **d=512, 8 heads, 10 decoder layers, crop 1024** | **97M** | **2.0–2.2** | **15.5 GB** |

  d=640 would only reach about 50k steps overnight and leaves no VRAM for inference by
  other sessions, so v1 uses d=512.
