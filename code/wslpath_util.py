# Path and runner helpers for the PARMA driver compiled in WSL.
# Distro name can be set with the environment variable SEEFLUX_WSL_DISTRO (default "Ubuntu").
import os, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
DISTRO = os.environ.get("SEEFLUX_WSL_DISTRO", "Ubuntu")


def wsl(p):
    p = pathlib.Path(p).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


BUILD = wsl(ROOT / "parma" / "build")


def run_driver(lines):
    """Run seeflux_driver on input lines; return split output lines, asserting one output line per input."""
    r = subprocess.run(["wsl", "-d", DISTRO, "--", "bash", "-lc", f'cd "{BUILD}" && ./seeflux_driver'],
                       input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    out = [l.split() for l in r.stdout.split("\n") if l.strip()]
    assert len(out) == len(lines), f"driver returned {len(out)} lines for {len(lines)} inputs"
    return out
