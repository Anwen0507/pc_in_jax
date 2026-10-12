"""Check the optional activity-optimizer experiment using notebook source cells."""

import copy
import json
from unittest.mock import Mock

import equinox as eqx
import jax
import numpy as np
import optax
import pytest

from task45_test_support import cell_with, execute_cell
from test_pc_in_jax_task45 import (
    isolated_outputs, notebook, sample_result, tiny_config, tiny_loaders,
)
from test_pc_in_jax_tasks import array_snapshot, assert_arrays_unchanged, make_model


@pytest.fixture
def comparison(notebook):
    execute_cell(cell_with("def activity_optimizer_configs("), notebook)
    execute_cell(cell_with("def activity_settling_trace("), notebook,
                 definitions_only=True)
    execute_cell(cell_with("def summarize_activity_optimizers("), notebook,
                 definitions_only=True)
    return notebook


def complete_results(comparison):
    return [sample_result(config) for config in comparison["activity_optimizer_configs"]()]


def test_default_grid_only_changes_activities_and_pairs_seeds(comparison):
    comparison["OPTIMIZER_SEEDS"] = [42, 43]
    baseline = copy.deepcopy(comparison["TASK4_BASE_CONFIG"])
    configs = list(comparison["activity_optimizer_configs"]())
    assert len(configs) == 12
    assert comparison["OPTIMIZER_USE_WANDB"] is False
    assert comparison["OPTIMIZER_LRS"] == [0.001, 0.01, 0.1]
    for sgd, adam in zip(configs[::2], configs[1::2]):
        assert sgd["activity_optimizer"] == "sgd"
        assert adam == {**sgd, "activity_optimizer": "adam"}
        assert sgd["n_train_iters"] == 1000
        assert sgd["inference_steps"] == 20
    assert comparison["TASK4_BASE_CONFIG"] == baseline


@pytest.mark.parametrize("name,values", [
    ("OPTIMIZER_LRS", []), ("OPTIMIZER_LRS", [0]),
    ("OPTIMIZER_LRS", [float("nan")]), ("OPTIMIZER_LRS", [-0.1]),
    ("OPTIMIZER_LRS", [0.1, 0.1]), ("OPTIMIZER_SEEDS", []),
    ("OPTIMIZER_SEEDS", [42, 42]),
])
def test_invalid_comparison_settings(comparison, name, values):
    comparison[name] = values
    with pytest.raises(ValueError):
        list(comparison["activity_optimizer_configs"]())


def test_local_sweep_saves_completed_runs_and_failures(comparison, tmp_path):
    configs = list(comparison["activity_optimizer_configs"]())
    comparison["run_task4_experiment"] = Mock(side_effect=[
        FloatingPointError("diverged"), *[sample_result(c) for c in configs[1:]]])
    comparison["run_task4_wandb"] = Mock(side_effect=AssertionError("Unexpected logging"))
    # Local use must also work when all WandB configuration names are absent.
    for name in ("wandb", "WANDB_MODE", "TASK4_GROUP"):
        comparison.pop(name)
    path = tmp_path / "comparison.json"
    results, failures = comparison["run_activity_optimizer_comparison"](results_path=path)
    assert len(results) == 5
    assert failures == [{"config": configs[0], "error": "diverged"}]
    assert json.loads(path.read_text()) == {
        "planned_configs": configs, "results": results, "failures": failures}
    assert [call.args[0] for call in comparison["run_task4_experiment"].call_args_list] == configs
    comparison["run_task4_wandb"].assert_not_called()


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_optional_wandb_uses_separate_group(comparison, tmp_path, mode):
    comparison["WANDB_MODE"] = mode
    comparison["wandb"] = Mock()
    comparison["run_task4_wandb"] = Mock(side_effect=lambda config, **_: sample_result(config))
    results, failures = comparison["run_activity_optimizer_comparison"](
        use_wandb=True, results_path=tmp_path / "logged.json")
    assert len(results) == 6 and not failures
    assert comparison["wandb"].login.call_count == int(mode == "online")
    for call in comparison["run_task4_wandb"].call_args_list:
        assert call.kwargs["group"] == comparison["TASK4_GROUP"] + "-optimizer-comparison"


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_settling_trace_includes_initial_and_final_energy_and_resets_state(comparison, optimizer):
    model = make_model(comparison)
    original = array_snapshot(model)
    batches, _ = tiny_loaders()
    inputs, targets = (array.numpy() for array in batches[0])
    trace_fn = comparison["activity_settling_trace"]
    trace = np.asarray(trace_fn(model, inputs, targets, optimizer=optimizer, lr=0.01, steps=5))
    p_optim = optax.adam(0.001)
    out = comparison["train_step"](
        model, inputs, targets, p_optim, p_optim.init(eqx.filter(model, eqx.is_array)),
        getattr(optax, optimizer)(0.01), T=5, record_aux_details=True)
    expected_preupdate = sum(np.asarray(values) for values in out["aux"].values()) / len(inputs)
    assert trace.shape == (6,)
    np.testing.assert_allclose(trace[:-1], expected_preupdate, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(trace[-1], out["energy"], rtol=2e-6, atol=1e-7)
    repeated = trace_fn(model, inputs, targets, optimizer=optimizer, lr=0.01, steps=5)
    np.testing.assert_array_equal(trace, repeated)
    assert_arrays_unchanged(model, original)


@pytest.mark.parametrize("optimizer,steps", [("sgd", 0), ("invalid", 5)])
def test_settling_trace_rejects_invalid_controls(comparison, optimizer, steps):
    batches, _ = tiny_loaders()
    inputs, targets = (array.numpy() for array in batches[0])
    with pytest.raises(ValueError):
        comparison["activity_settling_trace"](
            make_model(comparison), inputs, targets,
            optimizer=optimizer, lr=0.1, steps=steps)


def test_fixed_batch_diagnostic_cell_executes(comparison, tmp_path):
    comparison["OPTIMIZER_BASE_CONFIG"] = tiny_config(comparison)
    comparison["get_mnist_loaders"] = Mock(return_value=tiny_loaders())
    execute_cell(cell_with("def activity_settling_trace("), comparison)
    traces = comparison["optimizer_energy_traces"]
    assert len(traces) == 6
    assert all(trace.shape == (2,) and np.isfinite(trace).all() for trace in traces.values())
    assert len({float(trace[0]) for trace in traces.values()}) == 1
    assert (tmp_path / "task4_activity_optimizer_settling.png").exists()


def test_summary_pairs_seeds_and_excludes_incomplete_best_settings(comparison, capsys):
    comparison["OPTIMIZER_SEEDS"] = [42, 43]
    results = complete_results(comparison)
    for result in results:
        config = result["config"]
        accuracy = 90 + config["activity_lr"] + (config["seed"] - 42)
        if config["activity_optimizer"] == "adam":
            accuracy += 2
        result["final_test_accuracy"] = accuracy
    rows, best = comparison["summarize_activity_optimizers"](results)
    assert len(rows) == 6
    assert best["sgd"]["lr"] == best["adam"]["lr"] == 0.1
    assert best["adam"]["accuracy"] - best["sgd"]["accuracy"] == pytest.approx(2)
    assert best["adam"]["accuracy_std"] == pytest.approx(np.sqrt(0.5))
    assert capsys.readouterr().out.count("+2.00 +/- 0.00 pp") == 3
    missing = [result for result in results if not (
        result["config"]["activity_optimizer"] == "adam"
        and result["config"]["activity_lr"] == 0.1
        and result["config"]["seed"] == 43)]
    rows, best = comparison["summarize_activity_optimizers"](missing)
    assert best["adam"]["lr"] == 0.01
    assert "incomplete pairs" in capsys.readouterr().out
    fig = comparison["plot_activity_optimizer_results"](rows, best)
    assert len(fig.axes) == 4


@pytest.mark.parametrize("problem", ["config", "key", "duplicate", "history", "empty", "nan"])
def test_summary_rejects_incompatible_results(comparison, problem):
    results = complete_results(comparison)
    if problem == "config":
        results[0]["config"]["inference_steps"] += 1
    elif problem == "key":
        results[0]["config"]["seed"] = 123
    elif problem == "duplicate":
        results.append(copy.deepcopy(results[0]))
    elif problem == "history":
        results[0]["history"][-1]["train/iteration"] -= 1
    elif problem == "empty":
        results[0]["history"] = []
    else:
        results[0]["final_test_accuracy"] = float("nan")
    with pytest.raises(ValueError):
        comparison["summarize_activity_optimizers"](results)


@pytest.mark.parametrize("delta,verdict", [(-1, "lower"), (0, "equal"), (1, "higher")])
def test_summary_reports_measured_direction_without_assuming_adam_wins(comparison, capsys, delta, verdict):
    results = complete_results(comparison)
    for result in results:
        if result["config"]["activity_optimizer"] == "adam":
            result["final_test_accuracy"] += delta
    comparison["summarize_activity_optimizers"](results)
    assert f"Adam's best tested mean accuracy is {verdict}" in capsys.readouterr().out


def test_all_failed_runs_still_have_a_report(comparison, capsys):
    rows, best = comparison["summarize_activity_optimizers"]([])
    assert rows == [] and best == {}
    output = capsys.readouterr().out
    assert output.count("missing/failed") == 6
    assert "Not enough complete settings" in output
    fig = comparison["plot_activity_optimizer_results"](rows, best)
    assert len(fig.axes) == 4


def test_real_local_pipeline_executes_training_and_summary_cells(comparison, tmp_path):
    comparison["OPTIMIZER_BASE_CONFIG"] = tiny_config(comparison)
    comparison["OPTIMIZER_BASE_CONFIG"].update(n_train_iters=3, test_every=2, inference_steps=3)
    comparison["get_mnist_loaders"] = Mock(return_value=tiny_loaders())
    execute_cell(cell_with("optimizer_results, optimizer_failures ="), comparison)
    results = comparison["optimizer_results"]
    assert len(results) == 6 and not comparison["optimizer_failures"]
    for result in results:
        assert [row["train/iteration"] for row in result["history"]] == [2, 3]
        assert 0 <= result["final_test_accuracy"] <= 100
        assert result["runtime_seconds"] > 0
    execute_cell(cell_with("def summarize_activity_optimizers("), comparison)
    assert len(comparison["optimizer_summary"]) == 6
    assert (tmp_path / "task4_activity_optimizer_results.json").exists()
    assert (tmp_path / "task4_activity_optimizer_comparison.png").exists()
