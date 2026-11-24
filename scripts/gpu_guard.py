"""Keep benchmark model servers from interfering with PROD workloads on the shared GPU.

LOCKED OUT (2026-10-22): this machine's GPU serves production work, so `launch` refuses to start anything
while LOCKOUT is set. Lifting it is a deliberate edit here, on a GPU without PROD, not a flag.

  check   preflight only: is there room for a model of --need-gb while leaving --reserve-gb free?
  launch  preflight, then run a module or script in this same Python process with
          - a hard VRAM cap (torch per-process memory fraction), so the server can't grow into PROD's memory
          - a watchdog that exits immediately if free VRAM falls below --min-free-gb (PROD needs it back)

Examples (run with the venv that has the server's torch); any arguments gpu_guard doesn't know go to the target:
  python scripts/gpu_guard.py check --need-gb 11
  servers/kev/.venv/Scripts/python.exe scripts/gpu_guard.py launch --need-gb 11 --module kev.serve --run jaredpalmer/kev-4b --port 8009
  servers/intern-venv/Scripts/python.exe scripts/gpu_guard.py launch --need-gb 11 --script scripts/intern_server.py servers/intern-decision-4b --port 8010

Exit codes: 0 ok, 2 preflight refused, 3 watchdog stopped the server, 4 locked out.
"""

import argparse
import os
import runpy
import subprocess
import sys
import threading
import time

GB = 1024  # nvidia-smi reports MiB

LOCKOUT = (
    "local model servers are locked out on this machine: its GPU serves production work. The router runs on "
    "DeepSeek via API. See README, 'Model servers: local GPU locked out'."
)


def gpu_state(index: int = 0) -> dict:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout.strip()
    name, total, used, free, util = [p.strip() for p in out.split(",")]
    return {"name": name, "total": int(total), "used": int(used), "free": int(free), "util": int(util)}


def preflight(need_gb: float, reserve_gb: float, max_util: int, index: int = 0) -> tuple[bool, str]:
    s = gpu_state(index)
    after = s["free"] - need_gb * GB
    report = (f"{s['name']}: {s['used'] / GB:.1f} GB used, {s['free'] / GB:.1f} GB free, {s['util']}% busy; "
              f"after loading {need_gb:g} GB: {after / GB:.1f} GB free (reserve {reserve_gb:g} GB)")
    if after < reserve_gb * GB:
        return False, f"REFUSED (not enough VRAM): {report}"
    if s["util"] > max_util:
        return False, f"REFUSED (GPU busy above {max_util}%): {report}"
    return True, f"OK: {report}"


def cap_torch(cap_gb: float, index: int = 0) -> None:
    import torch

    total = torch.cuda.get_device_properties(index).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * GB * 1024 * 1024 / total), index)


def watchdog(min_free_gb: float, interval: float, index: int = 0) -> None:
    def loop():
        while True:
            time.sleep(interval)
            try:
                free = gpu_state(index)["free"]
            except (subprocess.SubprocessError, OSError, ValueError):
                continue
            if free < min_free_gb * GB:
                print(f"gpu_guard: free VRAM {free / GB:.1f} GB < {min_free_gb:g} GB floor; stopping to give it back",
                      file=sys.stderr, flush=True)
                os._exit(3)

    threading.Thread(target=loop, daemon=True, name="gpu-guard-watchdog").start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--need-gb", type=float, required=True, help="VRAM the model will take")
        p.add_argument("--reserve-gb", type=float, default=4, help="VRAM that must stay free for PROD")
        p.add_argument("--max-util", type=int, default=60, help="refuse if the GPU is busier than this (%%)")
        p.add_argument("--gpu", type=int, default=0)
        if name == "launch":
            p.add_argument("--cap-gb", type=float, help="hard VRAM cap for this process (default: need + 1)")
            p.add_argument("--min-free-gb", type=float, default=2, help="watchdog floor: exit if free VRAM drops below")
            p.add_argument("--interval", type=float, default=5, help="watchdog poll seconds")
            target = p.add_mutually_exclusive_group(required=True)
            target.add_argument("--module", help="run this module as __main__ (like python -m)")
            target.add_argument("--script", help="run this script as __main__")
    args, rest = parser.parse_known_args()
    if args.cmd == "launch" and LOCKOUT:
        print(f"gpu_guard: REFUSED: {LOCKOUT}", file=sys.stderr, flush=True)
        return 4
    if args.cmd == "check" and rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    rest = rest[1:] if rest[:1] == ["--"] else rest

    ok, report = preflight(args.need_gb, args.reserve_gb, args.max_util, args.gpu)
    print(f"gpu_guard: {report}", flush=True)
    if not ok:
        return 2
    if args.cmd == "check":
        return 0

    cap = args.cap_gb or args.need_gb + 1
    cap_torch(cap, args.gpu)
    watchdog(args.min_free_gb, args.interval, args.gpu)
    print(f"gpu_guard: capped at {cap:g} GB, watchdog floor {args.min_free_gb:g} GB", flush=True)
    if args.module:
        sys.argv = [args.module, *rest]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    else:
        sys.argv = [args.script, *rest]
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))
        runpy.run_path(args.script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
