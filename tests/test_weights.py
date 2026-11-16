from conftest import MakeGguf

from bonsai.weights import weight_bytes

# Tensor names as llama.cpp writes them; float32 is 4 bytes per value.
EMBED = "token_embd.weight"
OUTPUT = "output.weight"
LAYER = "blk.0.ffn_up.weight"


def test_untied_model_skips_the_embedding_table(make_gguf: MakeGguf) -> None:
    path = make_gguf(**{EMBED: 1000, OUTPUT: 1000, LAYER: 500})

    counted = weight_bytes(path)

    assert counted.total == 4 * 2500
    assert counted.input_embedding == 4 * 1000
    assert not counted.tied
    assert counted.per_token == 4 * 1500


def test_tied_model_reads_the_embedding_table(make_gguf: MakeGguf) -> None:
    path = make_gguf(**{EMBED: 1000, LAYER: 500})

    counted = weight_bytes(path)

    assert counted.tied
    assert counted.per_token == counted.total == 4 * 1500


def test_model_without_embeddings(make_gguf: MakeGguf) -> None:
    counted = weight_bytes(make_gguf(**{LAYER: 500}))

    assert counted.input_embedding == 0
    assert counted.per_token == 4 * 500
