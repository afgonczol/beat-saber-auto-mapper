"""Calibration check: human-made maps in the local game install come out nearly clean.

Needs a local Beat Saber install, so it is marked slow. Override the folder with
BSMAPPER_CUSTOM_LEVELS. Only BeatSaver maps with keys >= 0x10000 (2020 and later)
count: older maps predate today's parity conventions and are often badly broken.
"""

import os
import re
from collections import Counter
from pathlib import Path

import pytest

from bsmapper.formats import MapFormatError, read_map_report
from bsmapper.rules import RuleChecker

CUSTOM_LEVELS = Path(
    os.environ.get(
        "BSMAPPER_CUSTOM_LEVELS",
        r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels",
    )
)
MODERN_KEY = re.compile(r"^([0-9a-f]{5,}) \(")


@pytest.mark.slow
def test_modern_human_maps_are_nearly_clean():
    if not CUSTOM_LEVELS.is_dir():
        pytest.skip(f"no CustomLevels folder at {CUSTOM_LEVELS}")
    checker = RuleChecker()
    errors: Counter[str] = Counter()
    notes = difficulties = 0
    for folder in sorted(CUSTOM_LEVELS.iterdir()):
        m = MODERN_KEY.match(folder.name)
        if not m or int(m.group(1), 16) < 0x10000:
            continue
        try:
            beatmap, report = read_map_report(folder)
        except MapFormatError:
            continue
        if report.needs_mods:
            continue
        for dmap in beatmap.difficulties:
            if not dmap.notes:
                continue
            difficulties += 1
            notes += len(dmap.notes)
            errors.update(i.rule for i in checker.check(dmap, beatmap) if i.severity == "error")
    if not notes:
        pytest.skip("no modern maps found")
    rate = 100 * sum(errors.values()) / notes
    print(f"{difficulties} difficulties, {notes} notes, {rate:.2f} errors/100 notes: {errors}")
    assert rate < 0.5, errors
