"""Offline baseline CLI; doctor/report do not import native BLT."""

import argparse
import json
from pathlib import Path

from .evidence import Config, doctor, write_json
from .runner import execute, render_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        print(render_report(json.loads(args.result.read_text())))
        return 0
    cfg = Config.load(args.config.resolve())
    if args.command == "doctor":
        args.out.mkdir(parents=True, exist_ok=False)
        state = doctor(cfg)
        write_json(args.out / "readiness.json", state)
        result = {"status": state["status"], "readiness": state}
        write_json(args.out / "result.json", result)
        (args.out / "report.md").write_text(render_report(result))
        print(f"{state['status']}: {args.out / 'report.md'}")
        return 2 if state["blockers"] else 0
    success = execute(cfg, args.out)
    print(f"{'COMPLETED' if success else 'BLOCKED'}: {args.out / 'report.md'}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
