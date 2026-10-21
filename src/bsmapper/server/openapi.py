"""Write the jobs API's OpenAPI spec: ``python -m bsmapper.server.openapi [path]``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parents[3] / "web" / "openapi.json"


def spec() -> dict[str, Any]:
    from bsmapper.server.app import create_app

    class _NoRunner:  # the spec does not depend on the runner; skip loading the mock map
        def run(self, job, progress): ...
        def regenerate(self, job, previous, request, progress): ...

    return create_app(runner=_NoRunner(), web_dist=Path("/nonexistent")).openapi()


def dumps() -> str:
    return json.dumps(spec(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> None:
    out = Path(argv[0]) if argv else SPEC_PATH
    out.write_text(dumps(), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
