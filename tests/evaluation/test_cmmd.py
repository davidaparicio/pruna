from typing import Any

import pytest

from pruna.config.smash_config import SmashConfig
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.engine.pruna_model import PrunaModel
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics.metric_cmmd import CMMD
from pruna.evaluation.task import Task

@pytest.mark.parametrize(
    "model_fixture, device, clip_model",
    [
        pytest.param("ddpm-cifar10", "cuda", "openai/clip-vit-large-patch14-336", marks=pytest.mark.cuda),
        pytest.param("flux_tiny_random", "cpu", "openai/clip-vit-large-patch14-336", marks=pytest.mark.cpu),
    ],
    indirect=["model_fixture"],
)
def test_cmmd(model_fixture: tuple[Any, SmashConfig], device: str, clip_model: str) -> None:
    """Test CMMD."""
    model, smash_config = model_fixture
    smash_config.device = device
    pruna_model = PrunaModel(model, smash_config=smash_config)

    metric = CMMD(clip_model_name=clip_model, device=device)

    batch = next(iter(smash_config.test_dataloader()))
    x, gt = batch
    outputs = pruna_model.run_inference(batch, device)

    # Calculate CMMD between model outputs and ground truth
    metric.update(x, gt, outputs)
    comparison_results = metric.compute().result

    metric.reset()

    # Calculate CMMD between ground truth and itself
    metric.update(x, gt, gt)
    self_comparison_results = metric.compute().result

    assert self_comparison_results == pytest.approx(0.0, abs=1e-2)
    assert comparison_results > self_comparison_results


@pytest.mark.parametrize(
    "model_fixture, device, clip_model",
    [
        pytest.param("sd_tiny_random", "cuda", "openai/clip-vit-large-patch14-336", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_task_cmmd_pairwise(model_fixture: tuple[Any, SmashConfig], device: str, clip_model: str):
    """Test CMMD pairwise."""
    model, _ = model_fixture
    data_module = PrunaDataModule.from_string("LAION256")
    data_module.limit_datasets(10)

    task = Task(
        request=[CMMD(call_type="pairwise", clip_model_name=clip_model, device=device)],
        datamodule=data_module,
        device=device,
    )
    eval_agent = EvaluationAgent(task=task)

    eval_agent.evaluate(model)
    result = eval_agent.evaluate(model)

    assert result[0].result == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize(
    "model_fixture, device, clip_model",
    [
        pytest.param("sd_tiny_random", "cuda", "openai/clip-vit-large-patch14-336", marks=pytest.mark.cuda),
    ],
    indirect=["model_fixture"],
)
def test_cmmd_pairwise_direct_params(model_fixture: tuple[Any, SmashConfig], device: str, clip_model: str):
    """Test CMMD pairwise using direct parameters to EvaluationAgent."""
    model, _ = model_fixture
    data_module = PrunaDataModule.from_string("LAION256")
    data_module.limit_datasets(10)

    eval_agent = EvaluationAgent(
        request=[CMMD(call_type="pairwise", clip_model_name=clip_model, device=device)],
        datamodule=data_module,
        device=device,
    )

    eval_agent.evaluate(model)
    result = eval_agent.evaluate(model)

    assert result[0].result == pytest.approx(0.0, abs=1e-2)


def test_evaluation_agent_parameter_validation():
    """Test parameter validation for EvaluationAgent constructor."""
    data_module = PrunaDataModule.from_string("LAION256")

    device = "cpu"

    with pytest.raises(ValueError, match=r"Cannot specify both 'task' parameter and direct parameters"):
        task = Task(request="image_generation_quality", datamodule=data_module, device=device)
        EvaluationAgent(task=task, request="image_generation_quality")

    with pytest.raises(ValueError, match=r"both 'request' and 'datamodule' must be provided"):
        EvaluationAgent(request="image_generation_quality")

    with pytest.raises(ValueError, match=r"both 'request' and 'datamodule' must be provided"):
        EvaluationAgent(datamodule=data_module)

    task = Task(request="image_generation_quality", datamodule=data_module, device=device)
    agent = EvaluationAgent(task=task)
    assert agent is not None
