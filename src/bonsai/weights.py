"""How many bytes a tree reads to generate one token.

The ceiling ``read bandwidth / bytes per token`` is only as good as its byte
count. A GGUF's size overstates it: the input embedding table
(``token_embd.weight``) is a lookup, and generating a token reads one row of
it, not the whole table. For a small model with a large vocabulary that table
is a big share of the file (about a fifth of Qwen2.5-0.5B).

The exception is a model with *tied* embeddings stored once: with no separate
``output.weight``, the output projection reuses the embedding table and reads
all of it for every token.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gguf import GGUFReader

INPUT_EMBEDDING = "token_embd.weight"
OUTPUT = "output.weight"


@dataclass(frozen=True)
class WeightBytes:
    """Tensor bytes in a GGUF, and how many are read per generated token."""

    total: int
    """Every tensor's data."""
    input_embedding: int
    """The token embedding table."""
    tied: bool
    """True if there is no separate output layer, so the table is reused."""

    @property
    def per_token(self) -> int:
        """Bytes read to generate one token (ignoring the single embedding row)."""
        return self.total if self.tied else self.total - self.input_embedding


def weight_bytes(path: Path) -> WeightBytes:
    """Read a GGUF's tensor table (not its data) and count its bytes."""
    reader = GGUFReader(path)
    sizes = {tensor.name: int(tensor.n_bytes) for tensor in reader.tensors}
    return WeightBytes(
        total=sum(sizes.values()),
        input_embedding=sizes.get(INPUT_EMBEDDING, 0),
        tied=OUTPUT not in sizes,
    )
