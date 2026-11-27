"""``GET /api/settings``: the defaults a job gets for options it leaves out, and where maps
install. The web UI shows these next to its "default" choices."""

from __future__ import annotations

import logging

from bsmapper.generation.install import config_path, load_config
from bsmapper.schema import Difficulty
from bsmapper.server.models import InstallTarget, JumpDefaults, ServerSettings

log = logging.getLogger(__name__)


def install_target() -> InstallTarget:
    try:
        config = load_config()
    except (OSError, ValueError) as exc:
        return InstallTarget(
            path="", found=False, chroma=False, config=str(config_path()), error=str(exc)
        )
    levels = config.levels_dir
    return InstallTarget(
        path=str(levels),
        found=levels.is_dir(),
        chroma=config.chroma_installed(),
        config=str(config_path()),
    )


def _configured(fn, fallback):
    """A config value, or ``fallback`` if the config file cannot be read (the job then
    fails with the config error, which says what to fix)."""
    try:
        return fn()
    except (OSError, ValueError) as exc:
        log.warning("config: %s", exc)
        return fallback


def server_settings(generator: str, generator_detail: str = "") -> ServerSettings:
    from bsmapper.generation import configured_jumps
    from bsmapper.generation.density import configured_objects
    from bsmapper.generation.intensity import configured_intensity, configured_repetition
    from bsmapper.generation.lighting import configured_light_style
    from bsmapper.generation.njs import JUMP_TABLE

    objects = _configured(configured_objects, {})
    overrides = _configured(configured_jumps, {})
    jump = {}
    for diff in Difficulty:
        table, override = JUMP_TABLE[diff], overrides.get(diff)
        jump[diff] = JumpDefaults(
            njs=override.njs if override and override.njs is not None else table.njs,
            reaction_time=(
                override.reaction_s
                if override and override.reaction_s is not None
                else table.reaction_s
            ),
        )
    intensity = _configured(configured_intensity, None)
    return ServerSettings(
        generator=generator,
        generator_detail=generator_detail,
        intensity=0.5 if intensity is None else intensity,
        repetition=_configured(configured_repetition, None),
        light_style=_configured(configured_light_style, "auto"),
        arcs=objects.get("arcs"),
        chains=objects.get("chains"),
        jump=jump,
        install=install_target(),
    )
