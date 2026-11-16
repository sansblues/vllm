# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprob drift against a reference build.

Record the reference build's runs, then the candidate's, comparing them:
    pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
        --drift-out=runs/main
    pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
        --drift-out=runs/pr --drift-reference=runs/main
"""

import asyncio
import json
import secrets
import shlex
import tempfile
from pathlib import Path

import yaml

from tests.utils import RemoteOpenAIServer

from .logprob_drift_eval import (
    compare,
    fresh_caches,
    jsonable,
    load,
    record,
    render,
    save,
)

DEFAULT_RUNS = 5
DEFAULT_STARTUP_MAX_WAIT_SECONDS = 1200


def test_logprob_drift(config_filename: Path, pytestconfig):
    cfg = yaml.safe_load(config_filename.read_text())
    out = Path(pytestconfig.getoption("--drift-out")) / config_filename.stem
    runs = pytestconfig.getoption("--drift-runs") or cfg.get("runs", DEFAULT_RUNS)
    server_args = shlex.split(cfg.get("server_args", ""))
    env = {k: str(v) for k, v in cfg.get("env", {}).items()}
    setup = {"server_args": server_args, "env": env}
    assert not any(out.glob("*.npz")), f"{out} already holds runs: pick a new one"

    # A server per run, each compiling and autotuning from empty caches: a
    # run's numerics can depend on its launch, and the comparison needs runs
    # that are independent of each other.
    for i in range(runs):
        seed = secrets.randbelow(2**31)
        with (
            tempfile.TemporaryDirectory() as cache,
            RemoteOpenAIServer(
                cfg["model_name"],
                [*server_args, "--disable-uvicorn-access-log"],
                env_dict=env | fresh_caches(cache),
                max_wait_seconds=cfg.get(
                    "startup_max_wait_seconds", DEFAULT_STARTUP_MAX_WAIT_SECONDS
                ),
            ) as server,
        ):
            run = asyncio.run(record(server.url_root, seed, setup))
        save(run, out / f"{seed}.npz")
        print(f"Run {i + 1} of {runs}: {out / f'{seed}.npz'}")

    reference = pytestconfig.getoption("--drift-reference")
    if reference is None:
        print(f"Recorded {runs} runs; pass --drift-reference to compare them")
        return
    refs = [
        load(p) for p in sorted((Path(reference) / config_filename.stem).glob("*.npz"))
    ]
    cands = [load(p) for p in sorted(out.glob("*.npz"))]
    result = compare(refs, cands)
    print(render(result))
    (out / "result.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    assert not result["changed"], "the candidate's outputs changed beyond rerun noise"
