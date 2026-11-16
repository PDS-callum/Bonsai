"""Run measuring jobs in the pot without keeping a connection open.

Every job runs detached on the pot inside tmux (through
``scripts/pi/run-detached.sh``), so a dropped SSH connection never stops it.
Each job gets its own directory on the pot::

    ~/bonsai/jobs/<job-id>/
        job.sh          the command, as run
        log.txt         everything it printed
        started_at      UTC timestamp
        finished_at     UTC timestamp, once it has finished
        exit_code       its exit status, once it has finished
        inputs/         files uploaded for it (e.g. trees), removable after
        results/        what it measured

The PC drives everything over ``ssh`` and ``scp``: submit a job, check on it
later, then collect its results. Only one job runs in the pot at a time, so
measurements never compete with each other.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

Runner = Callable[[Sequence[str]], str]
"""Runs a local command (ssh or scp) and returns its standard output."""

SESSION_PREFIX = "bonsai-"
# tmux turns "." in session names into "_", so new names leave it out. Ids of
# older jobs may still contain one, so lookups accept it.
JOB_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
JOB_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def session_name(job_id: str) -> str:
    """The tmux session a job runs in, as tmux itself would name it."""
    return SESSION_PREFIX + job_id.replace(".", "_")


class JobError(RuntimeError):
    """A job could not be submitted, inspected or collected."""


def run_command(command: Sequence[str]) -> str:
    """Run a command, raising JobError with its stderr if it fails."""
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=True
        )
    except FileNotFoundError as error:
        msg = f"{command[0]} not found"
        raise JobError(msg) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or f"exit code {error.returncode}"
        msg = f"{shlex.join(command)} failed: {detail}"
        raise JobError(msg) from error
    return result.stdout


def local_path(path: Path) -> str:
    """A local path in a form scp cannot mistake for a remote one.

    scp reads anything with a colon before its first slash as ``host:path``,
    so a Windows path such as ``C:/x`` would be taken for host ``C``. Relative
    paths never start that way.
    """
    try:
        return os.path.relpath(path)
    except ValueError:  # On another Windows drive; no relative path exists.
        return str(path)


class JobState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LOST = "lost"
    """Not running, but it never recorded an exit code (e.g. the pot rebooted)."""


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: JobState
    exit_code: int | None
    started_at: str | None
    finished_at: str | None


def new_job_id(name: str, now: datetime | None = None) -> str:
    """A unique, sortable job id such as ``bench-20261116T203531Z``."""
    if not JOB_NAME.match(name):
        msg = f"invalid job name {name!r}: use letters, digits, '_' or '-'"
        raise JobError(msg)
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}-{stamp}"


class PotConnection:
    """A pot reached over SSH, where jobs are submitted and collected."""

    def __init__(
        self,
        host: str,
        *,
        remote_home: str = "bonsai",
        runner: Runner = run_command,
    ) -> None:
        """Reach the pot at ``host`` (``user@address``).

        Bonsai's files on the pot live in ``~/<remote_home>``.
        """
        self.host = host
        self._runner = runner
        self._remote_home = remote_home
        self._home: str | None = None

    # Paths on the pot are absolute, because a job's command runs from inside
    # its own directory.

    @property
    def home(self) -> str:
        """Absolute path of Bonsai's directory on the pot."""
        if self._home is None:
            user_home = self.ssh('printf %s "$HOME"').strip()
            if not user_home.startswith("/"):
                msg = f"unexpected home directory on {self.host}: {user_home!r}"
                raise JobError(msg)
            self._home = f"{user_home}/{self._remote_home}"
        return self._home

    @property
    def scripts(self) -> str:
        return f"{self.home}/scripts/pi"

    def job_dir(self, job_id: str) -> str:
        if not JOB_ID.match(job_id):
            msg = f"invalid job id {job_id!r}"
            raise JobError(msg)
        return f"{self.home}/jobs/{job_id}"

    def ssh(self, command: str) -> str:
        """Run a shell command on the pot."""
        return self._runner(["ssh", "-o", "BatchMode=yes", self.host, command])

    def upload(self, files: Sequence[Path], remote_dir: str) -> None:
        """Copy local files into a directory on the pot, creating it."""
        if not files:
            return
        self.ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        self._runner(
            ["scp", "-q", "-B", *map(local_path, files), f"{self.host}:{remote_dir}/"]
        )

    def sync_scripts(self, local_scripts: Path) -> None:
        """Copy the pot's scripts over, so jobs run the current versions."""
        files = sorted(
            path
            for path in local_scripts.iterdir()
            if path.suffix in {".sh", ".py", ".c"}
        )
        if not files:
            msg = f"no scripts in {local_scripts}"
            raise JobError(msg)
        self.upload(files, self.scripts)
        self.ssh(f"cd {shlex.quote(self.scripts)} && chmod +x ./*.sh ./*.py")

    def running_jobs(self) -> list[str]:
        """Ids of jobs whose tmux sessions are still alive."""
        sessions = self.ssh(
            "tmux list-sessions -F '#{session_name}' 2>/dev/null || true"
        )
        return sorted(
            line.removeprefix(SESSION_PREFIX)
            for line in sessions.splitlines()
            if line.startswith(SESSION_PREFIX)
        )

    def free_memory_bytes(self) -> int:
        """Memory the pot could give a new process now (MemAvailable)."""
        reply = self.ssh("awk '/^MemAvailable:/ {print $2}' /proc/meminfo")
        try:
            return int(reply.strip()) * 1024
        except ValueError as error:
            msg = f"cannot read free memory on {self.host}: {reply!r}"
            raise JobError(msg) from error

    def submit(
        self,
        name: str,
        command: Sequence[str],
        *,
        inputs: Sequence[Path] = (),
        needs_free_bytes: int = 0,
        now: datetime | None = None,
    ) -> str:
        """Start a detached job and return its id.

        ``inputs`` are uploaded to the job's ``inputs/`` directory first. The
        command runs from the job's directory, so relative paths such as
        ``results`` or ``inputs`` land inside it. Refuses to start while
        another job is running, so measurements never overlap, or while the
        pot has less than ``needs_free_bytes`` of memory free, so a job never
        squeezes the pot's other work.
        """
        if not command:
            msg = "a job needs a command"
            raise JobError(msg)
        busy = self.running_jobs()
        if busy:
            msg = f"the pot is busy with {', '.join(busy)}; one job at a time"
            raise JobError(msg)
        if needs_free_bytes:
            free = self.free_memory_bytes()
            if free < needs_free_bytes:
                msg = (
                    f"the pot has {free / 1024**3:.2f} GiB of memory free; "
                    f"this job needs {needs_free_bytes / 1024**3:.2f} GiB"
                )
                raise JobError(msg)

        job_id = new_job_id(name, now)
        job_dir = self.job_dir(job_id)
        self.upload(inputs, f"{job_dir}/inputs")
        self.ssh(
            shlex.join(
                [f"{self.scripts}/run-detached.sh", job_id, job_dir, "--", *command]
            )
        )
        return job_id

    def status(self, job_id: str) -> JobStatus:
        job_dir = self.job_dir(job_id)
        reply = self.ssh(
            f"cd {shlex.quote(job_dir)} 2>/dev/null || {{ echo missing=yes; exit; }}; "
            "for f in started_at finished_at exit_code; do "
            '[ -f "$f" ] && printf "%s=%s\\n" "$f" "$(cat "$f")"; done; '
            f"tmux has-session -t ={session_name(job_id)} 2>/dev/null "
            "&& echo running=yes || true"
        )
        fields = dict(line.split("=", 1) for line in reply.splitlines() if "=" in line)
        if fields.get("missing") == "yes":
            msg = f"no job {job_id} on {self.host}"
            raise JobError(msg)
        exit_code = int(fields["exit_code"]) if "exit_code" in fields else None
        if fields.get("running") == "yes":
            state = JobState.RUNNING
        elif exit_code is None:
            state = JobState.LOST
        else:
            state = JobState.SUCCEEDED if exit_code == 0 else JobState.FAILED
        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            started_at=fields.get("started_at"),
            finished_at=fields.get("finished_at"),
        )

    def log(self, job_id: str, lines: int = 20) -> str:
        return self.ssh(
            f"tail -n {int(lines)} {shlex.quote(self.job_dir(job_id))}/log.txt"
        )

    def list_jobs(self) -> list[str]:
        """Ids of every job on the pot, oldest first."""
        reply = self.ssh(f"ls -1 {shlex.quote(self.home)}/jobs 2>/dev/null || true")
        return sorted(line for line in reply.splitlines() if JOB_ID.match(line))

    def collect(self, job_id: str, local_dir: Path, *, clean: bool = False) -> Path:
        """Copy a finished job's results and records to ``local_dir/<job-id>``.

        With ``clean``, the job's uploaded inputs are then deleted from the pot,
        freeing its storage for the next tree; the records stay.
        """
        status = self.status(job_id)
        if status.state is JobState.RUNNING:
            msg = f"{job_id} is still running"
            raise JobError(msg)

        target = local_dir / job_id
        target.mkdir(parents=True, exist_ok=True)
        job_dir = self.job_dir(job_id)
        records = ["log.txt", "job.sh", "started_at", "finished_at", "exit_code"]
        present = self.ssh(
            f"cd {shlex.quote(job_dir)} && ls -1d {' '.join(records)} results "
            "2>/dev/null || true"
        ).split()
        if present:
            self._runner(
                [
                    "scp",
                    "-q",
                    "-B",
                    "-r",
                    *(f"{self.host}:{job_dir}/{name}" for name in present),
                    local_path(target),
                ]
            )
        if clean:
            self.ssh(f"rm -rf {shlex.quote(job_dir)}/inputs")
        return target
