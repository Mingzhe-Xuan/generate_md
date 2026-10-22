"""Small Bash process tests for durable Slurm exit evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

HELPER = Path(__file__).with_name("job_status.sh").resolve()
BASH = os.environ.get("SMOACS_TEST_BASH") or shutil.which("bash")


@pytest.mark.parametrize("command, expected", [("true", 0), ("exit 7", 7), ("kill -TERM $$", 143)])
def test_exit_status_is_preserved_and_recorded(tmp_path, command, expected):
    env = os.environ.copy()
    env["SLURM_JOB_ID"] = "123"
    result = subprocess.run(
        [BASH, "-c", 'set -euo pipefail; source "$1"; ' + command, "--", HELPER.as_posix()],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == expected, result.stderr
    status = json.loads((tmp_path / "results/smoacs/status/job-123.json").read_text())
    assert status["job_id"] == "123"
    assert status["exit_code"] == expected
    assert status["ended_at"].endswith("Z")


def test_refuses_missing_job_id(tmp_path):
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [BASH, "-c", 'set -euo pipefail; source "$1"', "--", HELPER.as_posix()],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert not (tmp_path / "results").exists()
