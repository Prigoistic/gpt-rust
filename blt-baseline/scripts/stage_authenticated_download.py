"""Explicitly authorized token use on a user-owned host, only over SSH stdin.

Do not run without specific user authorization for the credential/destination.
The token is never put in arguments, logged, or saved on the remote host.
"""

import argparse
import json
import shlex
import subprocess

from huggingface_hub import get_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    root = args.remote_root
    command = [f"{root}/.venv/bin/python", f"{root}/blt-baseline/scripts/fetch_checkpoints.py",
               "--cache", args.cache, "--config-out", f"{root}/blt-baseline/configs/t4-smoke.json",
               "--upstream", f"{root}/vendor/blt", "--workload", f"{root}/blt-baseline/data/diagnostics.jsonl",
               "--token-stdin"]
    remote = "env HF_HUB_DISABLE_XET=1 " + shlex.quote(f"PYTHONPATH={root}/blt-baseline/src") + " " + shlex.join(command)
    ssh = ["ssh", "-i", args.key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
           "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15", args.host, remote]
    return subprocess.run(ssh, input=json.dumps({"token": get_token()}), text=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
