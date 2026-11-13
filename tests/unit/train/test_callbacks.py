import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from wfspread.train.callbacks import TrainingDiagnostics


class _Recorder:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append((step, dict(metrics)))


def _one_step(callback, global_step):
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(global_step=global_step, logger=_Recorder())
    model(torch.ones(1, 3)).sum().backward()
    callback.on_before_optimizer_step(trainer, model, optimizer)
    optimizer.step()
    trainer.global_step += 1
    callback.on_train_batch_end(trainer, model, None, None, 0)
    return trainer.logger.calls


def test_logs_lr_grad_norm_and_weight_change():
    calls = _one_step(TrainingDiagnostics(every_n_steps=1), global_step=0)

    assert len(calls) == 1
    step, metrics = calls[0]
    assert step == 0
    assert metrics["diag/lr"] == pytest.approx(0.1)
    # Weight gradient is all ones (2 x 3), bias gradient all ones (2).
    assert metrics["diag/grad_norm"] == pytest.approx(math.sqrt(8))
    # SGD moves every weight from 1.0 by 0.1, so the change is 10% of the weight norm.
    assert metrics["diag/update/weight"] == pytest.approx(0.1)
    assert "diag/update/bias" not in metrics


def test_skips_steps_between_measurements():
    assert _one_step(TrainingDiagnostics(every_n_steps=2), global_step=1) == []


def test_rejects_a_non_positive_interval():
    with pytest.raises(ValueError, match="at least 1"):
        TrainingDiagnostics(every_n_steps=0)
