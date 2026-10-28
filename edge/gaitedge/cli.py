"""Command line for the headless device.

    python -m gaitedge bench   --clip walk.mp4 [--accuracy]
    python -m gaitedge run                       # the service: watch, record, process, upload
    python -m gaitedge record  --seconds 30      # one manual recording
    python -m gaitedge process walk.mp4          # analyse a file now, print the result
    python -m gaitedge status                    # queue, disk, last benchmark
    python -m gaitedge doctor                    # check the install end to end

Every command takes ``--config /etc/gaitedge/edge.yaml``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import settings as settings_module

DEFAULT_CONFIG = Path("/etc/gaitedge/edge.yaml")


def _settings(args):
    path = args.config
    if path is None and DEFAULT_CONFIG.exists():
        path = DEFAULT_CONFIG
    return settings_module.load(path)


# --------------------------------------------------------------------------
def cmd_bench(args) -> int:
    from . import bench

    s = _settings(args)
    models = tuple(args.models.split(",")) if args.models else settings_module.MODEL_VARIANTS
    report = bench.run(s, Path(args.clip), frames=args.frames, models=models,
                       accuracy=args.accuracy)
    path = bench.save(report, s)
    print(f"\nsaved {path}")
    return 0


def cmd_process(args) -> int:
    from .processor import analyse_file

    s = _settings(args)
    out = Path(args.out) if args.out else Path(args.video).with_suffix("")
    result = analyse_file(s, Path(args.video), out, make_overlay=not args.no_overlay)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwritten to {out}")
    return 0


def cmd_run(args) -> int:
    from .agent import Agent

    agent = Agent(_settings(args))
    return agent.run(max_clips=args.max_clips)


def cmd_record(args) -> int:
    from .agent import record_once

    job = record_once(_settings(args), seconds=args.seconds)
    if job is None:
        print("nothing recorded")
        return 1
    print(f"recorded {job.id} ({job.data['duration_s']:.1f} s); queued for processing")
    return 0


def cmd_status(args) -> int:
    from . import bench
    from .spool import Spool

    s = _settings(args)
    spool = Spool(s.root / "spool")
    print(json.dumps({
        "device": s.describe(),
        "queue": spool.counts(),
        "disk_used_gb": round(spool.disk_used_gb(), 2),
        "last_benchmark": (bench.load_latest(s) or {}).get("recommendation"),
    }, indent=2))
    return 0


def cmd_doctor(args) -> int:
    from .doctor import run_checks

    return 0 if run_checks(_settings(args)) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaitedge", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help=f"settings file (default {DEFAULT_CONFIG} if present)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bench", help="measure whether this device can analyse walks")
    p.add_argument("--clip", required=True, help="a recorded walk to measure with")
    p.add_argument("--frames", type=int, default=300, help="frames to time per model")
    p.add_argument("--models", default=None, help="comma-separated subset, e.g. full,lite")
    p.add_argument("--accuracy", action="store_true",
                   help="also analyse the whole clip with each model and compare")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("run", help="run the device service")
    p.add_argument("--max-clips", type=int, default=None, help="stop after N clips (testing)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("record", help="record one walk now")
    p.add_argument("--seconds", type=float, default=30.0)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("process", help="analyse a video file now")
    p.add_argument("video")
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--no-overlay", action="store_true")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("status", help="queue, disk and last benchmark")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="check the install end to end")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
