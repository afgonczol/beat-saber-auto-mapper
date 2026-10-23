"""Start, stop and watch a detached training run on Windows (the one RTX 3090).

Copy this file next to the run's ``run_train.py`` (``checkpoints/<run>/code/``)::

    python control.py status        # is it running, last step, guard events
    python control.py stop          # pause: stop trainer and watchdog (keeps last.pt)
    python control.py start         # start or resume from last.pt, with a watchdog
    python control.py watch         # (started by ``start``) the watchdog loop

Files in the run directory (``checkpoints/<run>/``): ``train.pid`` (the trainer's
cmd.exe; ``taskkill /T`` stops its whole tree), ``watchdog.pid``, ``train.log``,
``train.err.log``, ``guard.log`` (written by the trainer's divergence guard),
``watchdog.log`` and ``STOP`` (present after ``stop``: the watchdog never restarts a
run that was stopped on purpose).

Processes are created through WMI (``Win32_Process.Create``), so they are not children
of the shell that ran ``start`` and survive it closing. ``stop`` loses at most the steps
since the last snapshot (1k early on, 2k later).

The watchdog checks every 60 s:

* trainer gone: done if train.log has the final ``step <max_steps> val`` line, or if the
  guard gave up (``GAVE UP`` in guard.log). Otherwise it restarts from last.pt, at most
  3 times.
* trainer alive but the last 5 logged steps have a NaN loss or a mean gradient norm
  above 1e4: the in-process guard should have rolled back long before, so the
  watchdog stops the run and writes ALERT to watchdog.log instead of letting it burn.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = OUT.parent.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
MAX_RESTARTS = 3
GRAD_ALARM = 1e4


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(OUT / "watchdog.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_pid(name: str) -> int | None:
    try:
        return int((OUT / name).read_text().strip())
    except (OSError, ValueError):
        return None


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
    ).stdout
    return str(pid) in out


def kill_tree(pid: int | None) -> None:
    if alive(pid):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def detached(command: str) -> int:
    """Run ``command`` outside this process tree (WMI); returns its pid."""
    ps = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
        f"@{{CommandLine='{command}'; CurrentDirectory='{REPO}'}}; $r.ProcessId"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True
    ).stdout
    return int(out.strip().splitlines()[-1])


def text(name: str) -> str:
    try:
        return (OUT / name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def max_steps() -> int:
    cfg = text("config.yaml")
    m = re.search(r"max_steps:\s*(\d+)", cfg)
    return int(m.group(1)) if m else 80000


def finished() -> bool:
    return f"step {max_steps()} val" in text("train.log")


def gave_up() -> bool:
    return "GAVE UP" in text("guard.log")


def recent_steps(n: int = 5) -> list[tuple[int, float, float]]:
    """(step, note_loss, grad_norm) of the last ``n`` logged training steps."""
    flat = " ".join(text("train.log").split())
    rows = re.findall(r"step (\d+) note_loss=(\S+) .*?grad_norm=(\S+)", flat)
    return [(int(s), float(loss), float(g)) for s, loss, g in rows[-n:]]


def start_trainer() -> int:
    run = HERE / "run_train.py"
    cmd = f'cmd.exe /c ""{PY}" -u "{run}" >> "{OUT / "train.log"}" 2>> "{OUT / "train.err.log"}""'
    pid = detached(cmd)
    (OUT / "train.pid").write_text(str(pid))
    return pid


def cmd_start() -> None:
    (OUT / "STOP").unlink(missing_ok=True)
    if alive(read_pid("train.pid")):
        print(f"already running (train.pid {read_pid('train.pid')})")
    else:
        print(f"trainer started: pid {start_trainer()} (resumes from last.pt if present)")
    if not alive(read_pid("watchdog.pid")):
        pid = detached(f'"{PY}" -u "{Path(__file__).resolve()}" watch')
        (OUT / "watchdog.pid").write_text(str(pid))
        print(f"watchdog started: pid {pid}")


def cmd_stop() -> None:
    (OUT / "STOP").write_text(f"stopped {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    kill_tree(read_pid("watchdog.pid"))
    kill_tree(read_pid("train.pid"))
    print("stopped; resume with: python control.py start")


def cmd_status() -> None:
    steps = recent_steps(1)
    print(f"trainer  {'running' if alive(read_pid('train.pid')) else 'not running'}"
          f" (train.pid {read_pid('train.pid')})")  # fmt: skip
    print(f"watchdog {'running' if alive(read_pid('watchdog.pid')) else 'not running'}")
    print(f"STOP file: {(OUT / 'STOP').exists()}, finished: {finished()}")
    if steps:
        s, loss, g = steps[-1]
        print(f"last logged step {s} of {max_steps()}: note_loss {loss:.4g} grad_norm {g:.3g}")
    guard = text("guard.log").strip().splitlines()
    print("guard events:", *(guard[-3:] or ["none"]), sep="\n  ")


def cmd_watch() -> None:
    log(f"watching {OUT.name} (train.pid {read_pid('train.pid')})")
    restarts = 0
    while True:
        time.sleep(60)
        if (OUT / "STOP").exists():
            log("STOP file present; watchdog exits")
            return
        pid = read_pid("train.pid")
        if alive(pid):
            steps = recent_steps()
            bad = [s for s in steps if s[1] != s[1] or s[2] != s[2]]
            if bad or (len(steps) >= 5 and min(g for _, _, g in steps) > GRAD_ALARM):
                log(f"ALERT: diverging despite the guard ({steps[-1]}); stopping the run")
                (OUT / "STOP").write_text("stopped by the watchdog: divergence\n")
                kill_tree(pid)
                return
            continue
        if finished():
            log("training finished; watchdog exits")
            return
        if gave_up():
            log("the divergence guard gave up (guard.log); not restarting")
            return
        if restarts >= MAX_RESTARTS:
            log(f"trainer died again after {MAX_RESTARTS} restarts; giving up")
            return
        restarts += 1
        log(f"trainer {pid} exited early; restart {restarts}/{MAX_RESTARTS} from last.pt")
        log(f"restarted as pid {start_trainer()}")


if __name__ == "__main__":
    {"start": cmd_start, "stop": cmd_stop, "status": cmd_status, "watch": cmd_watch}[sys.argv[1]]()
