import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

MakeGguf = Callable[..., Path]

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip terminal styling, which Rich adds when it detects colour (as in CI)."""
    return ANSI_ESCAPE.sub("", text)


@pytest.fixture
def make_gguf(tmp_path: Path) -> MakeGguf:
    """Write a tiny GGUF whose tensors are float32 arrays of the given lengths."""

    def make(name: str = "tree.gguf", **tensors: int) -> Path:
        path = tmp_path / name
        writer = GGUFWriter(path, arch="llama")
        for tensor, length in tensors.items():
            writer.add_tensor(tensor, np.zeros(length, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path

    return make
