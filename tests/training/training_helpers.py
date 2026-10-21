from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def tiny_model() -> dict:
    return dict(
        d_model=32,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        encoder_layers=1,
        decoder_layers=1,
        light_layers=1,
        note_memory_layers=1,
        max_len=512,
        light_max_len=512,
        note_memory_len=256,
    )
