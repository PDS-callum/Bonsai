"""Run ``llama-server`` on the potting bench and ask it questions.

Generative tasks such as GSM8K need a tree to write its own answers. The
server applies the tree's chat template (``--jinja``) and decodes several
requests at once, so a batch of problems runs in parallel. Only the standard
library is used to talk to it.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from bonsai.llama import LlamaError

# GGUFs can carry their makers' suggested sampling settings (llama.cpp's
# converter copies them from generation_config.json), which llama-server then
# applies by default: Qwen2.5's include a repetition penalty of 1.1. Measuring
# must not depend on what a file suggests, so every request sets plain greedy
# decoding explicitly.
GREEDY: dict[str, Any] = {
    "temperature": 0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


@dataclass(frozen=True)
class Reply:
    text: str
    completion_tokens: int


class LlamaServer:
    """A ``llama-server`` process serving one tree, used as a context manager."""

    def __init__(
        self,
        server: Path,
        model: Path,
        *,
        parallel: int = 4,
        context: int = 8192,
        threads: int | None = None,
        log: Path | None = None,
        port: int | None = None,
        extra: Sequence[str] = (),
    ) -> None:
        self.port = port or free_port()
        self.command = [
            str(server),
            "--model", str(model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--parallel", str(parallel),
            "--ctx-size", str(context),
            "--jinja",
            *(["--threads", str(threads)] if threads else []),
            *extra,
        ]  # fmt: skip
        self.log = log
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: IO[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LlamaServer:
        self._log_file = self.log.open("wb") if self.log else None
        output = self._log_file or subprocess.DEVNULL
        self._process = subprocess.Popen(
            self.command, stdout=output, stderr=subprocess.STDOUT
        )
        self.wait_until_ready()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._log_file:
            self._log_file.close()

    def wait_until_ready(self, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                msg = f"llama-server exited with {self._process.returncode}"
                raise LlamaError(msg)
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2):
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        msg = f"llama-server was not ready after {timeout:.0f}s"
        raise LlamaError(msg)

    def chat(self, prompt: str, *, max_tokens: int = 512, seed: int = 0) -> Reply:
        """Ask one question with greedy decoding, so answers are repeatable."""
        return chat(self.url, prompt, max_tokens=max_tokens, seed=seed)

    def complete(self, prompt_ids: Sequence[int], tokens: int) -> list[int]:
        """Continue raw token ids greedily, with no chat template."""
        return complete(self.url, prompt_ids, tokens)


def _post(url: str, body: dict[str, Any], what: str) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as error:
        msg = f"{what} request failed: {error}"
        raise LlamaError(msg) from error


def chat(url: str, prompt: str, *, max_tokens: int = 512, seed: int = 0) -> Reply:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": seed,
        **GREEDY,
    }
    reply = _post(f"{url}/v1/chat/completions", body, "chat")
    try:
        return Reply(
            text=reply["choices"][0]["message"]["content"],
            completion_tokens=int(reply["usage"]["completion_tokens"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        msg = f"chat request failed: unexpected reply {error!r}"
        raise LlamaError(msg) from error


def complete(url: str, prompt_ids: Sequence[int], tokens: int) -> list[int]:
    """Greedy continuation of a prompt given as token ids."""
    body = {
        "prompt": list(prompt_ids),
        "n_predict": tokens,
        "cache_prompt": False,
        "return_tokens": True,
        **GREEDY,
    }
    reply = _post(f"{url}/completion", body, "completion")
    try:
        return [int(token) for token in reply["tokens"]]
    except (KeyError, TypeError, ValueError) as error:
        msg = f"completion request failed: unexpected reply {error!r}"
        raise LlamaError(msg) from error
