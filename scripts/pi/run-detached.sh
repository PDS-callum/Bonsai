#!/usr/bin/env bash
# Run a command as a detached job that survives SSH disconnects.
#
# The command runs inside a named tmux session. Everything it prints goes to
# <job-dir>/log.txt, and when it finishes its exit code is written to
# <job-dir>/exit_code. Check on a job later with:
#
#   tail -f <job-dir>/log.txt        # follow progress
#   cat <job-dir>/exit_code          # present only once the job has finished
#
# Usage: run-detached.sh <job-name> <job-dir> -- <command> [args...]

set -euo pipefail

usage() {
    echo "usage: $(basename "$0") <job-name> <job-dir> -- <command> [args...]" >&2
    exit 2
}

[[ $# -ge 4 && $3 == "--" ]] || usage
name=$1
job_dir=$2
shift 3

[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid job name: $name" >&2; exit 2; }
session="bonsai-$name"

if tmux has-session -t "=$session" 2>/dev/null; then
    echo "job already running: tmux session $session" >&2
    exit 1
fi

mkdir -p "$job_dir"
job_dir=$(cd "$job_dir" && pwd)
rm -f "$job_dir/exit_code" "$job_dir/finished_at"

# Write the job as a script, quoting each argument, so tmux doesn't need to
# re-parse the command line.
{
    echo '#!/usr/bin/env bash'
    printf 'cd %q\n' "$job_dir"
    echo 'date -u +%Y-%m-%dT%H:%M:%SZ > started_at'
    printf '%q ' "$@"
    echo '> log.txt 2>&1'
    echo 'status=$?'
    echo 'date -u +%Y-%m-%dT%H:%M:%SZ > finished_at'
    echo 'echo "$status" > exit_code'
} > "$job_dir/job.sh"
chmod +x "$job_dir/job.sh"

tmux new-session -d -s "$session" "$job_dir/job.sh"
echo "started $session; logs in $job_dir/log.txt"
