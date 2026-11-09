"""The installed product must support safe, exact continuation in both stages."""
from __future__ import annotations

import pytest
import torch

from conceptmod.ops import OpDefaults
from conceptmod.plan import compile_phrase
from conceptmod.training import TrainingConfig, TrainingSession
from supra_conceptmod.provider import PROVIDER


def _snapshot(backend):
    return {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
            for name, module in backend.training_modules().items()}


def _assert_weights(actual, expected):
    assert actual.keys() == expected.keys()
    for name, state in expected.items():
        assert actual[name].keys() == state.keys()
        for key, value in state.items():
            assert torch.equal(actual[name][key], value), (name, key)


@pytest.mark.parametrize("stage", ["model", "encoder"])
def test_dummy_session_checkpoint_resumes_exactly(stage, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(131)
    source = PROVIDER.create(device="cpu", model_id="dummy", lora_rank=4)
    plan = compile_phrase("red=blue", stage=stage, capabilities=PROVIDER.capabilities)
    config = TrainingConfig(iterations=2, lr=1e-3, seed=47, encoder_rank=2,
                            op_defaults=OpDefaults(**source.training_defaults()))
    checkpoint = tmp_path / "committed.pt"
    with TrainingSession(source, plan, config) as session:
        before = _snapshot(source)
        first = session.step()
        assert first.optimizer_steps == first.microsteps == 1
        after_one = _snapshot(source)
        assert any(not torch.equal(value, after_one[name][key])
                   for name, state in before.items() for key, value in state.items())
        session.save_checkpoint(checkpoint)
        expected_report = session.step()
        expected_weights = _snapshot(source)
    # A different initialization ensures the checkpoint restores frozen base
    # weights and encoders as well as the currently optimized adapter.
    torch.manual_seed(997)
    restored = PROVIDER.create(device="cpu", model_id="dummy", lora_rank=4)
    with TrainingSession(restored, plan, config) as resumed:
        resumed.load_checkpoint(checkpoint)
        assert resumed.completed_microsteps == 1
        _assert_weights(_snapshot(restored), after_one)
        assert resumed.step() == expected_report
        assert resumed.done
        _assert_weights(_snapshot(restored), expected_weights)
