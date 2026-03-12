"""
Tests to verify compatibility with torch 2.10.0.

These tests use mock data / minimal in-memory models to validate that
key API paths (torch.load, torch.compile, model encode/decode) work
correctly under torch 2.10.0.
"""

import inspect
import os
import tempfile
import warnings

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. torch version
# ---------------------------------------------------------------------------


def test_torch_version():
    """Verify that torch >= 2.10 is installed."""
    major, minor = torch.__version__.split(".")[:2]
    version_tuple = (int(major), int(minor.split("+")[0]))
    assert version_tuple >= (2, 10), (
        f"Expected torch >= 2.10, got {torch.__version__}"
    )


# ---------------------------------------------------------------------------
# 2. torch.load API compatibility
# ---------------------------------------------------------------------------


def _save_and_reload(obj, **load_kwargs):
    """Helper: save *obj* to a temp file, reload and return."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(obj, path)
        return torch.load(path, **load_kwargs)
    finally:
        os.unlink(path)


def test_torch_load_weights_only_true():
    """torch.load with weights_only=True must work for plain tensor dicts."""
    data = {"weight": torch.randn(4, 4), "bias": torch.zeros(4)}
    result = _save_and_reload(data, map_location="cpu", weights_only=True)
    assert set(result.keys()) == {"weight", "bias"}
    assert result["weight"].shape == (4, 4)


def test_torch_load_mmap_and_weights_only():
    """torch.load with mmap=True and weights_only=True must work."""
    data = {"a": torch.randn(8)}
    result = _save_and_reload(data, map_location="cpu", mmap=True, weights_only=True)
    assert result["a"].shape == (8,)


def test_torch_load_weights_only_false():
    """torch.load with weights_only=False must work for arbitrary objects."""
    data = {"a": torch.randn(3), "meta": {"info": "test"}}
    result = _save_and_reload(data, map_location="cpu", weights_only=False)
    assert result["meta"]["info"] == "test"


def test_torch_load_default_no_warning_for_tensor_dict():
    """torch.load without weights_only should not raise for tensor-only dicts."""
    data = {"w": torch.ones(2)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _save_and_reload(data, map_location="cpu")
    # Should not have raised a FutureWarning about weights_only
    future_warns = [
        w for w in caught if issubclass(w.category, FutureWarning) and "weights_only" in str(w.message)
    ]
    assert len(future_warns) == 0, f"Unexpected FutureWarning: {future_warns}"
    assert result["w"].allclose(torch.ones(2))


def test_torch_load_state_dict_with_assign():
    """load_state_dict(assign=True) must work (used in dac/inference.py)."""
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    model = TinyModel()
    original_sd = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(original_sd, path)
        loaded_sd = torch.load(path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(loaded_sd, strict=True, assign=True)
        assert len(result.missing_keys) == 0
        assert len(result.unexpected_keys) == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. torch.compile API compatibility
# ---------------------------------------------------------------------------


def test_torch_compile_signature_has_mode():
    """torch.compile must accept a 'mode' keyword argument."""
    sig = inspect.signature(torch.compile)
    assert "mode" in sig.parameters, "torch.compile must have a 'mode' parameter"


def test_torch_compile_mode_none():
    """torch.compile(backend='aot_eager', mode=None) must not crash."""
    def fn(x):
        return x * 2 + 1

    compiled = torch.compile(fn, backend="aot_eager", mode=None)
    x = torch.randn(4)
    result = compiled(x)
    assert result.shape == (4,)
    torch.testing.assert_close(result, fn(x))


def test_torch_compile_backend_inductor_like():
    """torch.compile with inductor-like settings matches eager output."""
    def fn(x):
        return torch.relu(x) + x.mean()

    compiled = torch.compile(fn, backend="aot_eager", mode="default", fullgraph=True)
    x = torch.randn(8)
    eager_out = fn(x)
    compiled_out = compiled(x)
    torch.testing.assert_close(eager_out, compiled_out)


# ---------------------------------------------------------------------------
# 4. Minimal mock-model encode / decode path (mirrors dac/modded_dac DAC usage)
# ---------------------------------------------------------------------------


class _TinyEncoder(torch.nn.Module):
    """Minimal stand-in for the DAC Encoder."""
    def forward(self, x):
        # x: [B, 1, T] → [B, C, T//hop]
        return x[:, :, ::4]  # stride-4 mock


class _TinyDecoder(torch.nn.Module):
    """Minimal stand-in for the DAC Decoder."""
    def forward(self, z):
        return torch.nn.functional.interpolate(z, scale_factor=4, mode="linear", align_corners=False)


class _MockCodec(torch.nn.Module):
    """Minimal mock codec that exercises the same encode→decode pattern as DAC."""
    sample_rate = 44100

    def __init__(self):
        super().__init__()
        self.encoder = _TinyEncoder()
        self.decoder = _TinyDecoder()

    def encode(self, audio: torch.Tensor, audio_lengths=None):
        z = self.encoder(audio)
        # Mock codebook indices: quantise to 0..7
        indices = (z.abs() * 7).long().clamp(0, 7)
        return indices, None

    def decode(self, indices: torch.Tensor):
        z = indices.float() / 7.0
        return self.decoder(z)


def test_mock_codec_encode_decode_roundtrip():
    """Encode then decode must return audio with the same batch/channel dims."""
    codec = _MockCodec().eval()
    B, T = 2, 4096
    audio = torch.randn(B, 1, T)
    with torch.no_grad():
        indices, _ = codec.encode(audio)
        reconstructed = codec.decode(indices)
    assert reconstructed.shape[0] == B
    assert reconstructed.shape[1] == 1


def test_mock_codec_state_dict_save_load():
    """Save & reload codec state-dict (mirrors dac/inference.py pattern)."""
    codec = _MockCodec()
    sd = codec.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(sd, path)
        loaded_sd = torch.load(path, map_location="cpu", weights_only=True)
        codec2 = _MockCodec()
        result = codec2.load_state_dict(loaded_sd, strict=True)
        assert len(result.missing_keys) == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 5. Minimal mock LLaMA-like model (mirrors text2semantic usage)
# ---------------------------------------------------------------------------


class _TinyTransformerBlock(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(self.norm(x))


class _TinyLLaMA(torch.nn.Module):
    """Mock of text2semantic LLaMA model for load-path testing."""
    def __init__(self, vocab=256, dim=16, n_layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.layers = torch.nn.ModuleList([_TinyTransformerBlock(dim) for _ in range(n_layers)])
        self.head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def test_mock_llama_load_with_mmap_and_weights_only():
    """Mock LLaMA load using mmap=True, weights_only=True (matches llama.py)."""
    model = _TinyLLaMA()
    sd = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(sd, path)
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        model2 = _TinyLLaMA()
        model2.load_state_dict(loaded, strict=True)
    finally:
        os.unlink(path)


def test_mock_llama_inference():
    """Mock LLaMA forward pass (sanity-check shape & no crash)."""
    model = _TinyLLaMA().eval()
    tokens = torch.randint(0, 256, (1, 8))
    with torch.inference_mode():
        logits = model(tokens)
    assert logits.shape == (1, 8, 256)


def test_mock_llama_compile_inference():
    """torch.compile wrapping a mock decode step must work."""
    model = _TinyLLaMA().eval()

    def decode_one(x):
        return model(x)

    compiled = torch.compile(decode_one, backend="aot_eager", mode=None, fullgraph=False)
    tokens = torch.randint(0, 256, (1, 4))
    with torch.no_grad():
        out = compiled(tokens)
    assert out.shape == (1, 4, 256)
