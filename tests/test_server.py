import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bonsai import server as server_module
from bonsai.llama import LlamaError
from bonsai.server import LlamaServer, chat, complete, free_port

FAKE = Path(__file__).parent / "fake_llama_server.py"


class PythonServer(LlamaServer):
    """Runs the fake server script through the current Python interpreter."""

    def __init__(self, tmp_path: Path, *extra: str) -> None:
        super().__init__(
            FAKE, tmp_path / "tree.gguf", log=tmp_path / "server.log", extra=extra
        )
        self.command.insert(0, sys.executable)


def test_command_line_carries_the_settings(tmp_path: Path) -> None:
    server = LlamaServer(
        Path("llama-server"), tmp_path / "tree.gguf", parallel=2, threads=3, port=9
    )

    assert server.command == [
        "llama-server",
        "--model", str(tmp_path / "tree.gguf"),
        "--host", "127.0.0.1",
        "--port", "9",
        "--parallel", "2",
        "--ctx-size", "8192",
        "--jinja",
        "--threads", "3",
    ]  # fmt: skip
    assert server.url == "http://127.0.0.1:9"


def test_free_port_is_usable() -> None:
    assert 0 < free_port() < 65536


def test_server_answers_and_stops(tmp_path: Path) -> None:
    with PythonServer(tmp_path) as server:
        reply = server.chat("What is six times seven?")
        process = server._process

    assert reply.text == "What is six times seven?\nThe answer is: 42"
    assert reply.completion_tokens == 7
    assert process is not None
    assert process.poll() is not None


def test_server_that_exits_is_reported(tmp_path: Path) -> None:
    with (
        pytest.raises(LlamaError, match="exited with 5"),
        PythonServer(tmp_path, "--fail"),
    ):
        pass


def test_server_that_never_starts_is_reported(tmp_path: Path) -> None:
    server = LlamaServer(Path("unused"), tmp_path / "tree.gguf", port=free_port())

    with pytest.raises(LlamaError, match="not ready after 1s"):
        server.wait_until_ready(timeout=1)


def test_chat_failures_are_reported() -> None:
    with pytest.raises(LlamaError, match="chat request failed"):
        chat(f"http://127.0.0.1:{free_port()}", "hello")


def test_server_continues_token_ids(tmp_path: Path) -> None:
    with PythonServer(tmp_path) as server:
        assert server.complete([5, 9], 3) == [10, 11, 12]


@pytest.mark.parametrize(
    ("call", "what"),
    [
        (lambda url: chat(url, "hello"), "chat"),
        (lambda url: complete(url, [1], 2), "completion"),
    ],
)
def test_unexpected_replies_are_reported(
    call: Any, what: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_module, "_post", lambda url, body, label: {"x": 1})

    with pytest.raises(LlamaError, match=f"{what} request failed: unexpected"):
        call("http://unused")


class StubbornProcess:
    """A process that ignores the first request to stop."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        if self.calls.count("wait") == 1:
            raise subprocess.TimeoutExpired("llama-server", timeout or 0)
        return 0


def test_server_that_will_not_stop_is_killed(tmp_path: Path) -> None:
    server = LlamaServer(Path("llama-server"), tmp_path / "tree.gguf", port=9)
    process = StubbornProcess()
    server._process = process  # type: ignore[assignment]

    server.__exit__(None, None, None)

    assert process.calls == ["terminate", "wait", "kill", "wait"]
