import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import MakeGguf, plain
from typer.testing import CliRunner

from bonsai import job_cli
from bonsai.cli import app
from bonsai.jobs import (
    JobError,
    JobState,
    PotConnection,
    local_path,
    new_job_id,
    run_command,
)

HOST = "pi@pot"
HOME = "/home/pi/bonsai"
NOW = datetime(2026, 11, 16, 20, 35, 31, tzinfo=UTC)

cli = CliRunner()


class FakeRunner:
    """Stands in for ssh and scp: records commands, replays canned replies.

    ``replies`` maps a fragment of a remote shell command to its output; the
    first fragment found in the command wins. Unmatched commands print nothing.
    """

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = {'printf %s "$HOME"': "/home/pi", **(replies or {})}
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        self.commands.append(list(command))
        if command[0] == "ssh":
            for fragment, reply in self.replies.items():
                if fragment in command[-1]:
                    return reply
        return ""

    @property
    def remote(self) -> list[str]:
        """The shell commands sent over ssh, without the home lookup."""
        return [c[-1] for c in self.commands if c[0] == "ssh" and "$HOME" not in c[-1]]

    @property
    def copies(self) -> list[list[str]]:
        return [c for c in self.commands if c[0] == "scp"]


def make(replies: dict[str, str] | None = None) -> tuple[PotConnection, FakeRunner]:
    runner = FakeRunner(replies)
    return PotConnection(HOST, runner=runner), runner


# Job ids


def test_job_id_is_name_and_utc_timestamp() -> None:
    assert new_job_id("bench", NOW) == "bench-20261116T203531Z"


@pytest.mark.parametrize(
    "name", ["", "two words", "../escape", "semi;colon", "qwen2.5"]
)
def test_job_names_are_restricted(name: str) -> None:
    with pytest.raises(JobError, match="invalid job name"):
        new_job_id(name, NOW)


def test_job_dirs_reject_unsafe_ids() -> None:
    pot, _ = make()

    with pytest.raises(JobError, match="invalid job id"):
        pot.job_dir("../../etc")


# Paths


def test_home_is_resolved_once() -> None:
    pot, runner = make()

    assert pot.home == HOME
    assert pot.scripts == f"{HOME}/scripts/pi"
    assert len(runner.commands) == 1


def test_home_must_be_absolute() -> None:
    pot, _ = make({'printf %s "$HOME"': "oops"})

    with pytest.raises(JobError, match="unexpected home directory"):
        _ = pot.home


def test_ssh_never_prompts() -> None:
    pot, runner = make()

    pot.ssh("true")

    assert runner.commands[0][:3] == ["ssh", "-o", "BatchMode=yes"]


def test_local_paths_never_look_remote(tmp_path: Path) -> None:
    path = tmp_path / "tree.gguf"

    converted = local_path(path)

    assert Path(converted).resolve() == path.resolve()
    assert ":" not in converted.split("/")[0].split("\\")[0]


def test_local_paths_on_another_drive_stay_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_relative_path(path: object) -> str:
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, "relpath", no_relative_path)

    assert local_path(tmp_path) == str(tmp_path)


# Submitting


def test_submit_uploads_inputs_and_starts_detached(tmp_path: Path) -> None:
    pot, runner = make()
    tree = tmp_path / "tree.gguf"
    tree.write_bytes(b"gguf")

    job_id = pot.submit(
        "bench", ["run.sh", "inputs", "results"], inputs=[tree], now=NOW
    )

    job_dir = f"{HOME}/jobs/bench-20261116T203531Z"
    assert job_id == "bench-20261116T203531Z"
    assert f"mkdir -p {job_dir}/inputs" in runner.remote
    (copy,) = runner.copies
    assert copy[-1] == f"{HOST}:{job_dir}/inputs/"
    assert Path(copy[-2]).resolve() == tree.resolve()
    assert runner.remote[-1] == (
        f"{HOME}/scripts/pi/run-detached.sh {job_id} {job_dir} -- run.sh inputs results"
    )


def test_submit_quotes_the_command() -> None:
    pot, runner = make()

    pot.submit("job", ["echo", "two words; rm -rf ~"], now=NOW)

    assert runner.remote[-1].endswith("-- echo 'two words; rm -rf ~'")


def test_submit_refuses_while_another_job_runs() -> None:
    pot, runner = make({"tmux list-sessions": "admin\nbonsai-bench-1\n"})

    with pytest.raises(JobError, match="busy with bench-1"):
        pot.submit("job", ["true"], now=NOW)
    assert not any("run-detached" in c for c in runner.remote)


def test_other_tmux_sessions_are_ignored() -> None:
    pot, _ = make({"tmux list-sessions": "admin\nclaude\n"})

    assert pot.running_jobs() == []


def test_submit_needs_a_command() -> None:
    pot, _ = make()

    with pytest.raises(JobError, match="needs a command"):
        pot.submit("job", [], now=NOW)


def test_sync_scripts_copies_runnable_files(tmp_path: Path) -> None:
    for name in ["a.sh", "b.py", "c.c", "README.md"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    pot, runner = make()

    pot.sync_scripts(tmp_path)

    (copy,) = runner.copies
    assert sorted(Path(arg).name for arg in copy[3:-1]) == ["a.sh", "b.py", "c.c"]
    assert copy[-1] == f"{HOST}:{HOME}/scripts/pi/"
    assert runner.remote[-1] == f"cd {HOME}/scripts/pi && chmod +x ./*.sh ./*.py"


def test_sync_scripts_needs_scripts(tmp_path: Path) -> None:
    pot, _ = make()

    with pytest.raises(JobError, match="no scripts"):
        pot.sync_scripts(tmp_path)


def test_submit_checks_free_memory() -> None:
    pot, runner = make({"MemAvailable": "1048576\n"})  # KiB: 1 GiB

    with pytest.raises(JobError, match=r"1.00 GiB of memory free"):
        pot.submit("job", ["true"], needs_free_bytes=2 * 1024**3, now=NOW)
    assert not any("run-detached" in c for c in runner.remote)


def test_submit_starts_with_enough_free_memory() -> None:
    pot, runner = make({"MemAvailable": "2097152\n"})  # KiB: 2 GiB

    pot.submit("job", ["true"], needs_free_bytes=1024**3, now=NOW)

    assert "run-detached" in runner.remote[-1]


def test_unreadable_free_memory_is_reported() -> None:
    pot, _ = make({"MemAvailable": "\n"})

    with pytest.raises(JobError, match="cannot read free memory"):
        pot.free_memory_bytes()


# Status


@pytest.mark.parametrize(
    ("reply", "state", "exit_code"),
    [
        ("started_at=t0\nrunning=yes\n", JobState.RUNNING, None),
        ("started_at=t0\nfinished_at=t1\nexit_code=0\n", JobState.SUCCEEDED, 0),
        ("started_at=t0\nfinished_at=t1\nexit_code=3\n", JobState.FAILED, 3),
        ("started_at=t0\n", JobState.LOST, None),
    ],
)
def test_status_reads_job_records(
    reply: str, state: JobState, exit_code: int | None
) -> None:
    pot, _ = make({"for f in started_at": reply})

    status = pot.status("job-1")

    assert status.state is state
    assert status.exit_code == exit_code
    assert status.started_at == "t0"


def test_status_finds_sessions_tmux_renamed() -> None:
    pot, runner = make({"for f in started_at": "running=yes\n"})

    pot.status("bench-qwen2.5-1")

    assert "has-session -t =bonsai-bench-qwen2_5-1 " in runner.remote[-1]


def test_status_of_unknown_job() -> None:
    pot, _ = make({"for f in started_at": "missing=yes\n"})

    with pytest.raises(JobError, match="no job job-1"):
        pot.status("job-1")


def test_log_tails_the_job_log() -> None:
    pot, runner = make({"tail -n 5": "last lines\n"})

    assert pot.log("job-1", 5) == "last lines\n"
    assert runner.remote[-1] == f"tail -n 5 {HOME}/jobs/job-1/log.txt"


def test_list_jobs_ignores_stray_names() -> None:
    pot, _ = make({"ls -1": "b-2\na-1\nnot a job\n"})

    assert pot.list_jobs() == ["a-1", "b-2"]


# Collecting

FINISHED = "started_at=t0\nfinished_at=t1\nexit_code=0\n"


def test_collect_copies_records_and_results(tmp_path: Path) -> None:
    pot, runner = make(
        {"for f in started_at": FINISHED, "ls -1d": "exit_code\nlog.txt\nresults\n"}
    )

    target = pot.collect("job-1", tmp_path)

    assert target == tmp_path / "job-1"
    assert target.is_dir()
    (copy,) = runner.copies
    assert copy[:4] == ["scp", "-q", "-B", "-r"]
    assert copy[4:-1] == [
        f"{HOST}:{HOME}/jobs/job-1/exit_code",
        f"{HOST}:{HOME}/jobs/job-1/log.txt",
        f"{HOST}:{HOME}/jobs/job-1/results",
    ]
    assert not any("rm -rf" in c for c in runner.remote)


def test_collect_can_clean_inputs(tmp_path: Path) -> None:
    pot, runner = make({"for f in started_at": FINISHED, "ls -1d": "log.txt\n"})

    pot.collect("job-1", tmp_path, clean=True)

    assert runner.remote[-1] == f"rm -rf {HOME}/jobs/job-1/inputs"


def test_collect_waits_for_the_job(tmp_path: Path) -> None:
    pot, _ = make({"for f in started_at": "running=yes\n"})

    with pytest.raises(JobError, match="still running"):
        pot.collect("job-1", tmp_path)


def test_collect_with_nothing_to_copy(tmp_path: Path) -> None:
    pot, runner = make({"for f in started_at": FINISHED})

    pot.collect("job-1", tmp_path)

    assert runner.copies == []


# Running commands


def test_run_command_returns_stdout() -> None:
    assert run_command([sys.executable, "-c", "print('hi')"]).strip() == "hi"


def test_run_command_reports_failures() -> None:
    script = "import sys; sys.stderr.write('bad thing'); sys.exit(2)"

    with pytest.raises(JobError, match="bad thing"):
        run_command([sys.executable, "-c", script])


def test_run_command_reports_missing_programs() -> None:
    with pytest.raises(JobError, match="not found"):
        run_command(["bonsai-no-such-program"])


def test_run_command_falls_back_to_exit_code() -> None:
    with pytest.raises(JobError, match="exit code 4"):
        run_command([sys.executable, "-c", "import sys; sys.exit(4)"])


# The CLI


@pytest.fixture
def fake_pot(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    runner = FakeRunner(
        {
            "for f in started_at": FINISHED,
            "ls -1 ": "bench-1\n",
            "ls -1d": "log.txt\n",
            "tail -n": "done\n",
            "MemAvailable": "2000000\n",  # KiB: about 1.9 GiB free
        }
    )
    monkeypatch.setattr(
        job_cli, "make_pot", lambda host: PotConnection(host, runner=runner)
    )
    monkeypatch.setenv("BONSAI_PI", HOST)
    return runner


@pytest.fixture
def scripts(tmp_path: Path) -> Path:
    directory = tmp_path / "scripts"
    directory.mkdir()
    (directory / "run.sh").write_text("", encoding="utf-8")
    return directory


def write_profile(directory: Path, memory_gib: float = 1.0) -> Path:
    profile = directory / "pot.toml"
    profile.write_text(
        'name = "test"\ndescription = "a test pot"\n'
        "[bandwidth]\nread_gbs = 10.0\n"
        f"[budget]\nmemory_gib = {memory_gib}\nstorage_gib = 0.001\nthreads = 3\n"
        "[runtime]\nrepack = false\n",
        encoding="utf-8",
    )
    return profile


def test_cli_bench_runs_trees_with_the_pots_settings(
    fake_pot: FakeRunner, scripts: Path, make_gguf: MakeGguf, tmp_path: Path
) -> None:
    tree = make_gguf("tree-q4_0.gguf", **{"output.weight": 1000})

    result = cli.invoke(
        app,
        [
            "job", "bench", str(tree), "--reference", "*q4_0*",
            "--pot", str(write_profile(tmp_path)), "--scripts", str(scripts),
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("bench-")
    started = fake_pot.remote[-1]
    assert "-- env THREADS=3 REPACK=0 'REFERENCE=*q4_0*' " in started
    assert started.endswith("/scripts/pi/bench-trees.sh inputs results")
    assert len(fake_pot.copies) == 2  # scripts, then the tree


def test_cli_bench_waits_for_free_memory(
    fake_pot: FakeRunner, scripts: Path, make_gguf: MakeGguf, tmp_path: Path
) -> None:
    tree = make_gguf("tree-q4_0.gguf", **{"output.weight": 1000})
    profile = write_profile(tmp_path, memory_gib=1.95)

    result = cli.invoke(
        app,
        ["job", "bench", str(tree), "--pot", str(profile), "--scripts", str(scripts)],
    )

    assert result.exit_code == 1
    assert "1.91 GiB of memory free; this job needs 1.95 GiB" in result.output
    assert not any("run-detached" in c for c in fake_pot.remote)


def test_cli_bench_checks_trees_before_contacting_the_pot(
    fake_pot: FakeRunner, scripts: Path, make_gguf: MakeGguf, tmp_path: Path
) -> None:
    # 300,000 float32 values: 1.2 MB, over a 0.001 GiB (1.07 MB) budget.
    tree = make_gguf("big.gguf", **{"output.weight": 300_000})
    profile = write_profile(tmp_path, memory_gib=0.001)

    result = cli.invoke(
        app,
        ["job", "bench", str(tree), "--pot", str(profile), "--scripts", str(scripts)],
    )

    assert result.exit_code == 1
    assert "big.gguf: weights of 0.00 GiB exceed" in result.output
    assert "the storage budget is 0.00 GiB" in result.output
    assert "must match exactly one tree; it matches 0" in result.output
    assert fake_pot.commands == []


def test_cli_bench_reports_a_bad_profile(
    fake_pot: FakeRunner, scripts: Path, make_gguf: MakeGguf, tmp_path: Path
) -> None:
    tree = make_gguf("tree-q4_0.gguf", **{"output.weight": 10})
    profile = tmp_path / "pot.toml"
    profile.write_text('name = "test"\n', encoding="utf-8")

    result = cli.invoke(
        app,
        ["job", "bench", str(tree), "--pot", str(profile), "--scripts", str(scripts)],
    )

    assert result.exit_code == 1
    assert "missing 'description'" in result.output


def test_cli_profile_submits_profile_job(fake_pot: FakeRunner, scripts: Path) -> None:
    result = cli.invoke(app, ["job", "profile", "--scripts", str(scripts)])

    assert result.exit_code == 0, result.output
    assert fake_pot.remote[-1].endswith("/scripts/pi/profile-pot.sh results")


def test_cli_sync(fake_pot: FakeRunner, scripts: Path) -> None:
    result = cli.invoke(app, ["job", "sync", "--scripts", str(scripts)])

    assert result.exit_code == 0, result.output
    assert len(fake_pot.copies) == 1


def test_cli_status_lists_jobs(fake_pot: FakeRunner) -> None:
    result = cli.invoke(app, ["job", "status"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "bench-1  succeeded (exit 0)\n"


def test_cli_log(fake_pot: FakeRunner) -> None:
    result = cli.invoke(app, ["job", "log", "bench-1", "-n", "3"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "done\n"


def test_cli_collect(fake_pot: FakeRunner, tmp_path: Path) -> None:
    result = cli.invoke(
        app, ["job", "collect", "bench-1", "--to", str(tmp_path), "--clean"]
    )

    assert result.exit_code == 0, result.output
    assert Path(result.stdout.strip()) == tmp_path / "bench-1"
    assert fake_pot.remote[-1].startswith("rm -rf ")


@pytest.mark.parametrize(
    "args",
    [
        ["job", "status", "bad/id"],
        ["job", "log", "bad/id"],
        ["job", "collect", "bad/id"],
    ],
)
def test_cli_reports_errors(fake_pot: FakeRunner, args: list[str]) -> None:
    result = cli.invoke(app, args)

    assert result.exit_code == 1
    assert "error: invalid job id" in result.output


def test_cli_reports_submit_errors(
    fake_pot: FakeRunner, scripts: Path, make_gguf: MakeGguf, tmp_path: Path
) -> None:
    fake_pot.replies["tmux list-sessions"] = "bonsai-other-1\n"
    tree = make_gguf("t-q4_0.gguf", **{"output.weight": 10})
    profile = write_profile(tmp_path)

    for args in (
        ["job", "bench", str(tree), "--pot", str(profile), "--scripts", str(scripts)],
        ["job", "profile", "--scripts", str(scripts)],
    ):
        result = cli.invoke(app, args)
        assert result.exit_code == 1
        assert "busy with other-1" in result.output


def test_cli_sync_reports_errors(fake_pot: FakeRunner, tmp_path: Path) -> None:
    result = cli.invoke(app, ["job", "sync", "--scripts", str(tmp_path)])

    assert result.exit_code == 1
    assert "no scripts" in result.output


def test_cli_needs_a_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BONSAI_PI", raising=False)

    result = cli.invoke(app, ["job", "status"])

    assert result.exit_code != 0
    assert "--host" in plain(result.output)


# A real round trip through run-detached.sh, where a POSIX shell is available.


@pytest.mark.skipif(
    sys.platform == "win32" or not Path("/bin/bash").exists(),
    reason="needs a POSIX shell",
)
def test_run_detached_records_exit_code(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # Stand-in for tmux: run the job synchronously; report no sessions.
    tmux = fake_bin / "tmux"
    tmux.write_text(
        '#!/bin/sh\ncase "$1" in new-session) shift 4; "$@";; *) exit 1;; esac\n',
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/pi/run-detached.sh"
    job_dir = tmp_path / "job"

    subprocess.run(
        [str(script), "t", str(job_dir), "--", "sh", "-c", "echo hi; exit 3"],
        check=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert (job_dir / "log.txt").read_text(encoding="utf-8") == "hi\n"
    assert (job_dir / "exit_code").read_text(encoding="utf-8").strip() == "3"
