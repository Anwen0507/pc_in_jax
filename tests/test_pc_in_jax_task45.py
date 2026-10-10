"""Task 4/5 contracts, independent numerical oracles, and offline integrations."""

import copy
import gc
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest
import torch

from notebook_test_support import NOTEBOOK
from task45_test_support import (
    RESULTS, ROOT, cell_with, execute_cell, extract_cells, load_namespace,
    selected_cells,
)
from test_pc_in_jax_tasks import (
    array_snapshot, assert_arrays_unchanged, local_mnist_batch, make_model,
    numpy_feedforward, numpy_settling, stable_model,
)


@pytest.fixture
def notebook():
    return load_namespace()


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", Mock())
    plt.close("all")
    yield
    plt.close("all")
    # Each namespace defines fresh jitted functions and model classes. Retaining
    # their executables across the whole suite can exhaust a notebook's memory.
    jax.clear_caches()
    gc.collect()


def config_for(notebook, **updates):
    config = dict(notebook["TASK4_BASE_CONFIG"])
    config.update(updates)
    return config


def tiny_config(notebook, **updates):
    return config_for(
        notebook, input_dim=2, output_dim=2, hidden_size=4, n_layers=2,
        batch_size=4, n_train_iters=2, test_every=50, inference_steps=1,
        **updates,
    )


def tiny_loaders():
    inputs = torch.tensor([[1., 0.], [0., 1.], [.8, .1], [.1, .8]])
    labels = torch.eye(2)[[0, 1, 0, 1]]
    return ([(inputs, labels)],
            [(inputs[:3], labels[:3]), (inputs[3:], labels[3:])])


def stub_training(notebook, *, energy=0.25, accuracy=75.0):
    train_loader, test_loader = tiny_loaders()
    notebook["get_mnist_loaders"] = Mock(return_value=(train_loader, test_loader))
    notebook["evaluate"] = Mock(return_value=accuracy)
    step = Mock(side_effect=lambda model, inputs, labels, p_optim, p_state,
                x_optim, **kwargs: {
                    "model": model, "p_opt_state": p_state,
                    "energy": jnp.asarray(energy),
                })
    notebook["train_step"] = step
    return step, train_loader, test_loader


def train_args(notebook, **updates):
    args = dict(
        model=make_model(notebook), activity_lr=0.1, inference_steps=1,
        param_lr=0.001, batch_size=4, test_every=2, n_train_iters=5,
    )
    args.update(updates)
    return args


class FakeRun:
    def __init__(self, config=None, *, run_id="unit-run"):
        self.config = {} if config is None else config
        self.id = run_id
        self.entity = "unit-entity"
        self.project = "unit-project"
        self.summary = {}
        self.log = Mock()
        self.define_metric = Mock()
        self.entered = False
        self.exits = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exception):
        self.exits.append(exception)
        return False


def fake_wandb(notebook, *, run=None, runs=(), entity="default-entity"):
    if run is None:
        run = FakeRun()
    api = SimpleNamespace(default_entity=entity, runs=Mock(return_value=list(runs)))
    sdk = SimpleNamespace(init=Mock(return_value=run), login=Mock(),
                          Api=Mock(return_value=api))
    notebook["wandb"] = sdk
    return sdk, api, run


def api_run(notebook, *, parameter="inference_steps", value=1, seed=42,
            accuracy=85.0, completed=1000, run_id="api-run", **config_updates):
    config = config_for(notebook, sweep_parameter=parameter, seed=seed,
                        **config_updates)
    config[parameter] = value
    return SimpleNamespace(
        config=config, id=run_id, state="finished",
        summary={"test/accuracy_at_1000": accuracy,
                 "train/iterations_completed": completed,
                 "runtime/seconds": 12.5},
    )


def sample_result(config, *, completed=None, accuracy=88.0):
    if completed is None:
        completed = config["n_train_iters"]
    return dict(config=dict(config),
                history=[{"train/iteration": completed, "train/energy": 0.25,
                          "test/accuracy": accuracy}],
                final_test_accuracy=accuracy, runtime_seconds=1.25)


def test_all_section_cells_are_extracted_verbatim():
    selected = selected_cells()
    assert [task for task, _ in selected.values()] == [4, 5, 5, 5, 5, 5]
    paths = extract_cells()
    for index, (_, source) in selected.items():
        assert paths[index].read_text() == source
    assert "# for activity_optimizer" in next(
        source for _, source in selected.values() if "# for activity_optimizer" in source)


def test_task4_default_grid_and_baseline_match_instructions(notebook):
    assert notebook["TASK4_SWEEPS"] == {
        "inference_steps": [1, 5, 20, 50, 100],
        "activity_lr": [0.001, 0.01, 0.1, 0.5, 1.0],
    }
    baseline = notebook["TASK4_BASE_CONFIG"]
    assert baseline == {
        "seed": 42, "input_dim": 784, "output_dim": 10,
        "hidden_size": 300, "n_layers": 3, "activation": "tanh",
        "activity_lr": 0.1, "inference_steps": 20, "param_lr": 0.001,
        "batch_size": 64, "test_every": 50, "n_train_iters": 1000,
        "activity_optimizer": "sgd",
    }
    assert notebook["TASK4_SEEDS"] == [42]


@pytest.mark.parametrize("seeds", [[42], [42, 43, 44], []])
def test_task4_grid_varies_only_one_parameter_and_seed(notebook, seeds):
    notebook["TASK4_SEEDS"] = seeds
    original = copy.deepcopy(notebook["TASK4_BASE_CONFIG"])
    configs = list(notebook["task4_configs"]())
    expected = []
    for parameter, values in notebook["TASK4_SWEEPS"].items():
        for value in values:
            for seed in seeds:
                expected.append({**original, parameter: value, "seed": seed,
                                 "sweep_parameter": parameter})
    assert configs == expected
    assert len(configs) == 10 * len(seeds)
    assert notebook["TASK4_BASE_CONFIG"] == original
    assert len({id(config) for config in configs}) == len(configs)
    if configs:
        configs[0]["batch_size"] = -1
        assert configs[1]["batch_size"] == original["batch_size"]


@pytest.mark.parametrize("sweeps", [{}, {"inference_steps": []},
                                      {"activity_lr": [0.0]}])
def test_task4_empty_or_zero_value_grid(notebook, sweeps):
    notebook["TASK4_SWEEPS"] = sweeps
    configs = list(notebook["task4_configs"]())
    assert len(configs) == sum(map(len, sweeps.values()))
    if configs:
        assert configs[0]["activity_lr"] == 0.0


@pytest.mark.parametrize("seed", [0, 42, 43, 2**31 - 1])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_experiment_forwards_configuration_and_uses_fresh_seeded_model(
        notebook, seed, optimizer):
    config = tiny_config(notebook)
    config.update(seed=seed, activity_optimizer=optimizer)
    original = copy.deepcopy(config)
    history = [{"train/iteration": 2, "test/accuracy": 62.5}]
    notebook["train"] = Mock(return_value=(None, history))
    notebook["torch"] = SimpleNamespace(manual_seed=Mock())
    notebook["perf_counter"] = Mock(side_effect=[10.0, 12.5, 20.0, 22.5])
    callback = Mock()
    first = notebook["run_task4_experiment"](config, callback)
    second = notebook["run_task4_experiment"](config)
    assert [call.args for call in notebook["torch"].manual_seed.call_args_list] == [(seed,), (seed,)]
    first_model = notebook["train"].call_args_list[0].kwargs["model"]
    second_model = notebook["train"].call_args_list[1].kwargs["model"]
    assert first_model is not second_model
    for actual, expected in zip(array_snapshot(first_model), array_snapshot(second_model)):
        np.testing.assert_array_equal(actual, expected)
    expected_model = make_model(notebook, input_size=2, hidden_size=4,
                                output_size=2, n_layers=2, seed=seed)
    assert_arrays_unchanged(first_model, array_snapshot(expected_model))
    forwarded = notebook["train"].call_args_list[0].kwargs
    assert set(forwarded) == {
        "model", "activity_lr", "inference_steps", "param_lr", "batch_size",
        "test_every", "n_train_iters", "seed", "activity_optimizer",
        "metrics_callback", "return_history",
    }
    for key in set(forwarded) - {"model", "metrics_callback", "return_history"}:
        assert forwarded[key] == config[key]
    assert forwarded["metrics_callback"] is callback
    assert forwarded["return_history"] is True
    assert first == dict(config=original, history=history,
                         final_test_accuracy=62.5, runtime_seconds=2.5)
    assert second["runtime_seconds"] == 2.5
    assert first["config"] is not config
    assert config == original


@pytest.mark.parametrize("error_type", [FloatingPointError, RuntimeError])
def test_experiment_training_errors_propagate(notebook, error_type):
    error = error_type("training failed")
    notebook["train"] = Mock(side_effect=error)
    with pytest.raises(error_type) as caught:
        notebook["run_task4_experiment"](tiny_config(notebook))
    assert caught.value is error


@pytest.mark.parametrize("history", [[], [{"train/iteration": 1, "test/accuracy": 90.0}]])
def test_experiment_rejects_incomplete_training_history(notebook, history):
    notebook["train"] = Mock(return_value=(None, history))
    with pytest.raises(ValueError, match="requested number of iterations"):
        notebook["run_task4_experiment"](tiny_config(notebook))


def test_experiment_rejects_an_unexpectedly_exhausted_training_stream(notebook):
    step, train_loader, _ = stub_training(notebook)
    notebook["infinite_loader"] = Mock(return_value=iter(train_loader))
    config = tiny_config(notebook)
    config.update(n_train_iters=5, test_every=1)
    with pytest.raises(ValueError, match="requested number of iterations"):
        notebook["run_task4_experiment"](config)
    assert step.call_count == 1


@pytest.mark.parametrize("iterations,test_every,expected", [
    (1, 50, [1]), (5, 2, [2, 4, 5]), (4, 2, [2, 4]),
    (3, 1, [1, 2, 3]), (1000, 50, list(range(50, 1001, 50))),
])
@pytest.mark.parametrize("return_history", [True, False])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_train_exact_updates_final_evaluation_history_and_callbacks(
        notebook, capsys, iterations, test_every, expected, return_history, optimizer):
    step, _, test_loader = stub_training(notebook)
    notebook["optax"] = SimpleNamespace(sgd=Mock(wraps=optax.sgd),
                                         adam=Mock(wraps=optax.adam))
    callback = Mock()
    args = train_args(notebook, n_train_iters=iterations, test_every=test_every,
                      activity_optimizer=optimizer, metrics_callback=callback,
                      return_history=return_history, seed=123)
    initial_snapshot = array_snapshot(args["model"])
    out = notebook["train"](**args)
    assert step.call_count == iterations
    assert [call.args for call in notebook["optax"].sgd.call_args_list] == (
        [(0.1,)] if optimizer == "sgd" else [])
    assert [call.args for call in notebook["optax"].adam.call_args_list] == (
        [(0.001,)] if optimizer == "sgd" else [(0.1,), (0.001,)])
    notebook["get_mnist_loaders"].assert_called_once_with(4, seed=123)
    assert notebook["evaluate"].call_count == len(expected)
    for call in notebook["evaluate"].call_args_list:
        assert call.args == (args["model"], test_loader)
    history = [call.args[0] for call in callback.call_args_list]
    assert [row["train/iteration"] for row in history] == expected
    assert history == [{"train/iteration": iteration, "train/energy": 0.25,
                        "test/accuracy": 75.0} for iteration in expected]
    assert all(type(row["train/energy"]) is float for row in history)
    if return_history:
        assert out == (args["model"], history)
    else:
        assert out is args["model"]
    for call in step.call_args_list:
        assert call.kwargs == {"T": 1, "record_aux_details": False}
        assert call.args[0] is args["model"]
        assert isinstance(call.args[1], np.ndarray)
        assert isinstance(call.args[2], np.ndarray)
    assert_arrays_unchanged(args["model"], initial_snapshot)
    assert capsys.readouterr().out.count("avg test accuracy=75.00%") == len(expected)


def test_train_parameter_state_and_model_are_carried_between_batches(notebook):
    stub_training(notebook)
    models = [make_model(notebook, seed=seed) for seed in range(4)]
    states = [object() for _ in range(3)]
    notebook["train_step"].side_effect = [
        {"model": models[index + 1], "p_opt_state": states[index], "energy": 0.5}
        for index in range(3)
    ]
    notebook["train"](**train_args(notebook, model=models[0], n_train_iters=3, test_every=1))
    calls = notebook["train_step"].call_args_list
    assert [call.args[0] for call in calls] == models[:3]
    assert calls[1].args[4] is states[0]
    assert calls[2].args[4] is states[1]
    assert [call.args[0] for call in notebook["evaluate"].call_args_list] == models[1:]


@pytest.mark.parametrize("field", ["n_train_iters", "test_every", "inference_steps"])
@pytest.mark.parametrize("value", [0, -1, -100])
def test_train_rejects_nonpositive_iteration_counts(notebook, field, value):
    stub_training(notebook)
    with pytest.raises(ValueError, match="must be positive"):
        notebook["train"](**train_args(notebook, **{field: value}))
    notebook["get_mnist_loaders"].assert_not_called()
    notebook["train_step"].assert_not_called()


@pytest.mark.parametrize("optimizer", ["Adam", "", None, "rmsprop"])
def test_train_rejects_unknown_activity_optimizer(notebook, optimizer):
    stub_training(notebook)
    with pytest.raises(ValueError, match="sgd.*adam"):
        notebook["train"](**train_args(notebook, activity_optimizer=optimizer))
    notebook["get_mnist_loaders"].assert_not_called()


@pytest.mark.parametrize("energy", [np.nan, np.inf, -np.inf])
def test_train_divergence_fails_before_logging_or_evaluation(notebook, energy):
    stub_training(notebook, energy=energy)
    callback = Mock()
    with pytest.raises(FloatingPointError, match="training energy"):
        notebook["train"](**train_args(notebook, test_every=1, metrics_callback=callback))
    notebook["evaluate"].assert_not_called()
    callback.assert_not_called()


@pytest.mark.parametrize("source", ["train_step", "evaluate", "callback", "loaders"])
def test_train_unexpected_errors_propagate(notebook, source):
    stub_training(notebook)
    error = RuntimeError(source)
    callback = Mock()
    target = {"callback": callback, "loaders": notebook["get_mnist_loaders"]}.get(
        source, notebook.get(source))
    target.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        notebook["train"](**train_args(notebook, test_every=1, metrics_callback=callback))
    assert caught.value is error


def test_train_without_callback(notebook):
    step, _, _ = stub_training(notebook)
    out = notebook["train"](**train_args(notebook, n_train_iters=1, return_history=True))
    assert step.call_count == 1
    assert out[1] == [{"train/iteration": 1, "train/energy": 0.25, "test/accuracy": 75.0}]


@pytest.mark.parametrize("batch_sizes", [[3], [2, 1], [1, 1, 1]])
@pytest.mark.parametrize("correct", [[0, 0, 0], [1, 0, 1], [1, 1, 1]])
def test_evaluate_weights_examples_and_returns_percent(notebook, batch_sizes, correct):
    predictions = torch.tensor([[1., 0.], [1., 0.], [1., 0.]])
    labels = torch.eye(2)[[0 if value else 1 for value in correct]]
    batches = []
    start = 0
    for size in batch_sizes:
        batches.append((predictions[start:start + size], labels[start:start + size]))
        start += size
    assert notebook["evaluate"](eqx.nn.Identity(), batches) == 100 * sum(correct) / 3


def test_evaluate_uses_feedforward_predictions_with_deterministic_ties(notebook):
    batches = [(torch.tensor([[1., 1., 0.], [0., 2., 2.]]), torch.eye(3)[[0, 1]])]
    assert notebook["evaluate"](eqx.nn.Identity(), batches) == 100.0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_evaluate_rejects_nonfinite_predictions(notebook, value):
    with pytest.raises(FloatingPointError, match="predictions"):
        notebook["evaluate"](eqx.nn.Identity(), [
            (torch.tensor([[value, 0.]]), torch.tensor([[1., 0.]]))])


@pytest.mark.parametrize("loader", [[], [(torch.empty(0, 2), torch.empty(0, 2))]])
def test_evaluate_rejects_no_examples(notebook, loader):
    with pytest.raises(ValueError, match="empty"):
        notebook["evaluate"](eqx.nn.Identity(), loader)


def test_infinite_loader_repeats_complete_epochs(notebook):
    iterator = notebook["infinite_loader"](["first", "last"])
    assert [next(iterator) for _ in range(7)] == ["first", "last", "first", "last", "first", "last", "first"]


def test_empty_training_loader_raises_instead_of_hanging(notebook):
    # Check in a bounded subprocess before risking an infinite loop here.
    script = """
import ast, json
from pathlib import Path
nb = json.loads(Path(sys_notebook).read_text())
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def infinite_loader(' in ''.join(cell['source']):
        tree = ast.parse(''.join(cell['source']))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'infinite_loader']
        ns = {}
        exec(compile(tree, '<notebook-empty-loader>', 'exec'), ns)
        try:
            next(ns['infinite_loader']([]))
        except ValueError:
            raise SystemExit(0)
        raise SystemExit(1)
"""
    script = "sys_notebook = " + repr(str(NOTEBOOK)) + "\n" + script
    try:
        child_env = {key: value for key, value in os.environ.items()
                     if not key.startswith("COV_CORE_")}
        completed = subprocess.run([sys.executable, "-c", script], timeout=2,
                                   capture_output=True, text=True, env=child_env)
    except subprocess.TimeoutExpired:
        pytest.fail("infinite_loader([]) hangs forever instead of rejecting an empty training loader")
    assert completed.returncode == 0, completed.stderr
    with pytest.raises(ValueError, match="empty"):
        next(notebook["infinite_loader"]([]))
    notebook["get_mnist_loaders"] = Mock(return_value=([], tiny_loaders()[1]))
    with pytest.raises(ValueError, match="empty"):
        notebook["train"](**train_args(notebook, n_train_iters=1))


def test_wandb_configuration_group_is_unique_and_has_no_credentials(notebook, capsys):
    assert notebook["WANDB_PROJECT"] == "pc-in-jax-task4"
    assert notebook["WANDB_ENTITY"] is None
    assert notebook["WANDB_MODE"] == "online"
    first_group = notebook["TASK4_GROUP"]
    execute_cell(cell_with("WANDB_PROJECT ="), notebook)
    assert first_group != notebook["TASK4_GROUP"]
    assert notebook["TASK4_GROUP"].startswith("task4-")
    assert len(notebook["TASK4_GROUP"].split("-")[-1]) == 6
    assert "Experiment group:" in capsys.readouterr().out


@pytest.mark.parametrize("iterations", [2, 999, 1000, 1001])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("sweep", [None, "activity_lr"])
def test_wandb_run_arguments_metrics_summary_and_context_exit(
        notebook, iterations, explicit, sweep):
    config = config_for(notebook, n_train_iters=iterations)
    if sweep is not None:
        config["sweep_parameter"] = sweep
    expected_result = sample_result(config)
    notebook["run_task4_experiment"] = Mock(return_value=expected_result)
    sdk, _, run = fake_wandb(notebook)
    options = dict(mode="offline", group="explicit-group") if explicit else {}
    result = notebook["run_task4_wandb"](config, **options)
    sdk.init.assert_called_once_with(
        project=notebook["WANDB_PROJECT"], entity=notebook["WANDB_ENTITY"],
        group="explicit-group" if explicit else notebook["TASK4_GROUP"],
        name=f"{sweep or 'baseline'}-T20-lr0.1-sgd-seed42", job_type="task4",
        tags=["task4", sweep or "baseline"], config=config,
        mode="offline" if explicit else "online", dir=".",
    )
    assert sdk.init.call_args.kwargs["config"] is not config
    assert [call.args for call in run.define_metric.call_args_list] == [
        ("train/iteration",), ("train/energy",), ("test/accuracy",)]
    assert [call.kwargs for call in run.define_metric.call_args_list] == [
        {}, {"step_metric": "train/iteration"}, {"step_metric": "train/iteration"}]
    notebook["run_task4_experiment"].assert_called_once_with(config, metrics_callback=run.log)
    expected_summary = {"test/final_accuracy": 88.0,
                        "train/iterations_completed": iterations, "runtime/seconds": 1.25}
    if iterations == 1000:
        expected_summary["test/accuracy_at_1000"] = 88.0
    assert run.summary == expected_summary
    assert result is expected_result
    assert result["run_id"] == run.id
    assert result["entity"] == run.entity
    assert result["project"] == run.project
    assert run.entered
    assert run.exits == [(None, None, None)]


@pytest.mark.parametrize("error_type", [FloatingPointError, RuntimeError])
def test_wandb_training_failure_finishes_context_without_success_summary(notebook, error_type):
    error = error_type("diverged")
    notebook["run_task4_experiment"] = Mock(side_effect=error)
    _, _, run = fake_wandb(notebook)
    with pytest.raises(error_type) as caught:
        notebook["run_task4_wandb"](tiny_config(notebook))
    assert caught.value is error
    assert run.summary == {}
    assert run.exits[0][0] is error_type
    assert run.exits[0][1] is error


def test_wandb_does_not_label_a_short_run_as_accuracy_at_1000(notebook):
    config = config_for(notebook, n_train_iters=1000)
    notebook["run_task4_experiment"] = Mock(return_value=sample_result(config, completed=999))
    _, _, run = fake_wandb(notebook)
    notebook["run_task4_wandb"](config)
    assert run.summary["train/iterations_completed"] == 999
    assert "test/accuracy_at_1000" not in run.summary


def test_wandb_logging_error_propagates_and_closes_context(notebook):
    sdk, _, run = fake_wandb(notebook)
    error = RuntimeError("logging unavailable")
    run.log.side_effect = error
    notebook["run_task4_experiment"] = Mock(side_effect=lambda config, metrics_callback:
                                           metrics_callback({"train/iteration": 1}))
    with pytest.raises(RuntimeError) as caught:
        notebook["run_task4_wandb"](tiny_config(notebook))
    assert caught.value is error
    assert run.exits[0][1] is error
    assert run.summary == {}


def test_smoke_cell_uses_offline_separate_group_and_two_updates(notebook):
    notebook["run_task4_wandb"] = Mock(return_value="smoke-result")
    original = copy.deepcopy(notebook["TASK4_BASE_CONFIG"])
    execute_cell(cell_with("smoke_config ="), notebook)
    expected = {**original, "sweep_parameter": "smoke", "inference_steps": 1,
                "n_train_iters": 2, "test_every": 2}
    assert notebook["smoke_config"] == expected
    notebook["run_task4_wandb"].assert_called_once_with(
        expected, mode="offline", group=f"{notebook['TASK4_GROUP']}-smoke")
    assert notebook["smoke_result"] == "smoke-result"
    assert notebook["TASK4_BASE_CONFIG"] == original


@pytest.mark.parametrize("mode", ["online", "offline"])
@pytest.mark.parametrize("failures", [[], [0, 2], list(range(10))])
def test_sweep_cell_runs_all_configs_and_continues_after_divergence(
        notebook, capsys, mode, failures):
    notebook["WANDB_MODE"] = mode
    sdk, _, _ = fake_wandb(notebook)
    configs = list(notebook["task4_configs"]())
    outputs = []
    for index, config in enumerate(configs):
        result = sample_result(config)
        result["run_id"] = f"run-{index}"
        outputs.append(FloatingPointError(f"failure-{index}") if index in failures else result)
    notebook["run_task4_wandb"] = Mock(side_effect=outputs)
    execute_cell(cell_with("failed_experiments ="), notebook)
    assert sdk.login.call_count == int(mode == "online")
    assert [call.args[0] for call in notebook["run_task4_wandb"].call_args_list] == configs
    assert notebook["wandb_results"] == [result for index, result in enumerate(outputs) if index not in failures]
    assert notebook["failed_experiments"] == [
        {"config": configs[index], "error": f"failure-{index}"} for index in failures]
    output = capsys.readouterr().out
    assert f"Completed {10 - len(failures)} runs; {len(failures)} failed." in output
    assert output.count("Running ") == 10


def test_sweep_cell_empty_grid_and_unexpected_failure(notebook):
    sdk, _, _ = fake_wandb(notebook)
    notebook["WANDB_MODE"] = "offline"
    notebook["TASK4_SEEDS"] = []
    notebook["run_task4_wandb"] = Mock()
    execute_cell(cell_with("failed_experiments ="), notebook)
    assert notebook["wandb_results"] == notebook["failed_experiments"] == []
    notebook["run_task4_wandb"].assert_not_called()
    notebook["TASK4_SEEDS"] = [42]
    notebook["run_task4_wandb"].side_effect = RuntimeError("SDK error")
    with pytest.raises(RuntimeError, match="SDK error"):
        execute_cell(cell_with("failed_experiments ="), notebook)
    assert notebook["run_task4_wandb"].call_count == 1
    assert notebook["failed_experiments"] == []


@pytest.mark.parametrize("explicit_entity,configured_entity,default_entity,expected", [
    ("explicit", "configured", "default", "explicit"),
    (None, "configured", "default", "configured"),
    (None, None, "default", "default"),
    ("", None, "default", "default"),
])
@pytest.mark.parametrize("explicit_path", [False, True])
def test_api_query_filters_completed_group_and_uses_entity_precedence(
        notebook, explicit_entity, configured_entity, default_entity, expected, explicit_path):
    notebook["WANDB_ENTITY"] = configured_entity
    run = api_run(notebook)
    sdk, api, _ = fake_wandb(notebook, runs=[run], entity=default_entity)
    options = {"entity": explicit_entity}
    if explicit_path:
        options.update(project="other-project", group="other-group")
    results = notebook["fetch_task4_results"](**options)
    sdk.Api.assert_called_once_with()
    api.runs.assert_called_once_with(
        f"{expected}/{'other-project' if explicit_path else notebook['WANDB_PROJECT']}",
        filters={"group": "other-group" if explicit_path else notebook["TASK4_GROUP"],
                 "state": "finished", "config.n_train_iters": 1000},
        order="-created_at",
    )
    assert results == [{"sweep_parameter": "inference_steps", "value": 1,
                        "seed": 42, "test_accuracy": 85.0,
                        "runtime_seconds": 12.5, "run_id": "api-run"}]


def test_api_missing_entity_and_no_runs_have_actionable_errors(notebook):
    _, api, _ = fake_wandb(notebook, entity=None)
    with pytest.raises(ValueError, match="WANDB_ENTITY"):
        notebook["fetch_task4_results"]()
    api.runs.assert_not_called()
    fake_wandb(notebook, runs=[])
    with pytest.raises(ValueError, match="No completed 1,000-iteration"):
        notebook["fetch_task4_results"]()


def test_api_newest_complete_per_seed_is_sorted_and_missing_is_reported(notebook, capsys):
    newest = api_run(notebook, value=5, accuracy=91.0, run_id="newest")
    older = api_run(notebook, value=5, accuracy=70.0, run_id="older")
    second = api_run(notebook, parameter="activity_lr", value=0.01, run_id="lr-run")
    invalid = [
        api_run(notebook, parameter="smoke"),
        api_run(notebook, parameter="activity_optimizer", value="adam"),
        api_run(notebook, accuracy=None),
        api_run(notebook, completed=999),
    ]
    fake_wandb(notebook, runs=[*invalid, newest, second, older])
    results = notebook["fetch_task4_results"]()
    assert [row["run_id"] for row in results] == ["lr-run", "newest"]
    assert results[1]["test_accuracy"] == 91.0
    output = capsys.readouterr().out
    assert "Missing or failed (parameter, value, seed):" in output
    assert "('inference_steps', 1, 42)" in output
    assert "('inference_steps', 5, 42)" not in output


@pytest.mark.parametrize("seeds", [[42], [42, 43, 44], []])
def test_api_complete_grid_and_multiple_seeds_without_missing_warning(notebook, capsys, seeds):
    notebook["TASK4_SEEDS"] = seeds
    runs = [api_run(notebook, parameter=config["sweep_parameter"],
                    value=config[config["sweep_parameter"]], seed=config["seed"],
                    run_id=f"run-{index}")
            for index, config in enumerate(notebook["task4_configs"]())]
    fake_wandb(notebook, runs=runs)
    if not seeds:
        with pytest.raises(ValueError, match="No completed"):
            notebook["fetch_task4_results"]()
    else:
        result = notebook["fetch_task4_results"]()
        assert len(result) == 10 * len(seeds)
        assert "Missing or failed" not in capsys.readouterr().out


@pytest.mark.parametrize("invalid_updates", [
    {"value": 999}, {"seed": 999}, {"activity_lr": 0.5},
    {"hidden_size": 50}, {"activity_optimizer": "adam"}, {"n_train_iters": 2},
])
def test_api_excludes_runs_outside_the_controlled_comparison(notebook, invalid_updates):
    valid = api_run(notebook, value=1, run_id="correct-baseline")
    invalid = api_run(notebook, value=5, run_id="wrong-comparison")
    for key, value in invalid_updates.items():
        invalid.config["inference_steps" if key == "value" else key] = value
    fake_wandb(notebook, runs=[invalid, valid])
    results = notebook["fetch_task4_results"]()
    assert [row["run_id"] for row in results] == ["correct-baseline"]


@pytest.mark.parametrize("bad_accuracy", [np.nan, np.inf, -np.inf, -0.1, 100.1, "invalid", {}, []])
def test_api_invalid_accuracy_does_not_contaminate_plots(notebook, bad_accuracy):
    invalid = api_run(notebook, accuracy=bad_accuracy, run_id="invalid-newest")
    valid = api_run(notebook, accuracy=0.0, run_id="valid-older")
    fake_wandb(notebook, runs=[invalid, valid])
    results = notebook["fetch_task4_results"]()
    assert len(results) == 1
    assert results[0]["run_id"] == "valid-older"
    assert results[0]["test_accuracy"] == 0.0


@pytest.mark.parametrize("missing", ["seed", "inference_steps", "activity_lr"])
def test_api_incomplete_config_is_ignored_without_losing_healthy_runs(notebook, missing):
    invalid = api_run(notebook, value=5)
    del invalid.config[missing]
    healthy = api_run(notebook, value=1, accuracy="100", run_id="healthy")
    healthy.summary.pop("runtime/seconds")
    fake_wandb(notebook, runs=[invalid, healthy])
    result = notebook["fetch_task4_results"]()
    assert len(result) == 1
    assert result[0]["test_accuracy"] == 100.0
    assert result[0]["runtime_seconds"] is None


def test_api_network_errors_propagate(notebook):
    _, api, _ = fake_wandb(notebook)
    api.runs.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        notebook["fetch_task4_results"]()


@pytest.mark.parametrize("population", ["empty", "one-sided", "single-seed", "multiple-seeds"])
def test_plots_sorted_means_sample_standard_deviation_labels_and_saved_image(
        notebook, monkeypatch, population):
    results = []
    if population != "empty":
        for parameter, values in notebook["TASK4_SWEEPS"].items():
            if population == "one-sided" and parameter == "activity_lr":
                continue
            for index, value in enumerate(values[::-1]):
                samples = [80 + index] if population != "multiple-seeds" else [70 + index, 80 + index, 90 + index]
                for seed, accuracy in enumerate(samples):
                    results.append(dict(sweep_parameter=parameter, value=value,
                                        seed=seed, test_accuracy=accuracy))
    results.append(dict(sweep_parameter="smoke", value=99, seed=1, test_accuracy=1))
    original = copy.deepcopy(results)
    real_subplots = plt.subplots
    recorded = []

    def subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        fig.tight_layout = Mock(wraps=fig.tight_layout)
        fig.savefig = Mock(wraps=fig.savefig)
        for ax in axes:
            ax.errorbar = Mock(wraps=ax.errorbar)
        recorded.append((fig, axes))
        return fig, axes

    monkeypatch.setattr(plt, "subplots", Mock(side_effect=subplots))
    figure = notebook["plot_task4_results"](results)
    plt.subplots.assert_called_once_with(1, 2, figsize=(12, 4))
    assert figure is recorded[0][0]
    for ax, parameter in zip(figure.axes, notebook["TASK4_SWEEPS"]):
        by_value = {}
        for row in original:
            if row["sweep_parameter"] == parameter:
                by_value.setdefault(row["value"], []).append(row["test_accuracy"])
        values = sorted(by_value)
        if values:
            args, kwargs = ax.errorbar.call_args
            np.testing.assert_array_equal(args[0], values)
            np.testing.assert_allclose(args[1], [sum(by_value[value]) / len(by_value[value]) for value in values])
            expected_std = []
            for value in values:
                sample = by_value[value]
                mean = sum(sample) / len(sample)
                expected_std.append((sum((item - mean)**2 for item in sample) / (len(sample) - 1))**0.5 if len(sample) > 1 else 0.0)
            np.testing.assert_allclose(kwargs["yerr"], expected_std)
            assert kwargs["marker"] == "o" and kwargs["capsize"] == 4
            assert len(ax.texts) == 0
        else:
            ax.errorbar.assert_not_called()
            assert [text.get_text() for text in ax.texts] == ["No completed runs"]
        assert ax.get_xlabel() == parameter
        assert ax.get_ylabel() == "Test accuracy at 1,000 iterations (%)"
        assert ax.get_xscale() == ("log" if parameter == "activity_lr" else "linear")
        expected_title = "Activity LR fixed at 0.1" if parameter == "inference_steps" else "Inference steps fixed at 20"
        assert ax.get_title() == expected_title
        np.testing.assert_allclose(ax.get_xticks(), notebook["TASK4_SWEEPS"][parameter])
        assert [text.get_text() for text in ax.get_xticklabels()] == [str(value) for value in notebook["TASK4_SWEEPS"][parameter]]
        assert all(line.get_visible() and line.get_alpha() == 0.3 for line in ax.get_xgridlines())
    figure.tight_layout.assert_called_once_with()
    figure.savefig.assert_called_once_with("task4_wandb_accuracy.png", dpi=180, bbox_inches="tight")
    plt.show.assert_called_once_with()
    output = Path("task4_wandb_accuracy.png")
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert results == original
    if population == "multiple-seeds":
        (RESULTS / "task4_test_plot.png").write_bytes(output.read_bytes())


def test_complete_api_and_plot_cell_executes_verbatim(notebook, capsys):
    runs = [api_run(notebook, parameter=config["sweep_parameter"],
                    value=config[config["sweep_parameter"]], seed=config["seed"])
            for config in notebook["task4_configs"]()]
    fake_wandb(notebook, runs=runs)
    execute_cell(cell_with("def fetch_task4_results("), notebook)
    assert len(notebook["task4_api_results"]) == 10
    assert isinstance(notebook["task4_figure"], plt.Figure)
    assert capsys.readouterr().out.count("'test_accuracy': 85.0") == 10
    execute_cell(cell_with("# for activity_optimizer"), notebook)


@pytest.mark.parametrize("steps", [1, 5, 20, 50, 100])
@pytest.mark.parametrize("activity_lr", [0.001, 0.01, 0.1, 0.5, 1.0])
def test_real_sgd_settling_for_all_requested_values_matches_independent_numpy(
        notebook, steps, activity_lr):
    model = stable_model(notebook, activation="tanh")
    inputs = np.array([[1., -.5], [-.3, .7], [.2, .4]], dtype=np.float32)
    labels = np.eye(2, dtype=np.float32)[[0, 1, 0]]
    oracle = numpy_settling(model, inputs, labels, activation="tanh",
                            steps=steps, activity_lr=activity_lr)
    p_optim = optax.adam(0.001)
    p_state = p_optim.init(eqx.filter(model, eqx.is_array))
    result = notebook["train_step"](model, inputs, labels, p_optim, p_state,
                                    optax.sgd(activity_lr), T=steps, record_aux_details=False)
    assert result["aux"] == {}
    for actual, expected in zip(result["activities"], oracle["activities"]):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result["energy"], oracle["energy"], rtol=3e-5, atol=3e-6)
    assert int(result["p_opt_state"][0].count) == 1
    for index, layer in enumerate(result["p_opt_state"][0].mu.layers):
        np.testing.assert_allclose(layer.layers[1].weight, 0.1 * oracle["weight_gradients"][index], rtol=5e-5, atol=3e-6)


def numpy_adam_settling(model, inputs, labels, *, steps, activity_lr):
    """Analytic PC gradients and Adam equations, independent of JAX/Optax."""
    # Initialize and recompute predictions in the same precision. Casting a
    # float32 feedforward result AFTER initialization produces spurious hidden
    # residuals in float64; Adam magnifies these into real activity updates.
    inputs = np.asarray(inputs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    activities = [array.astype(np.float64) for array in numpy_feedforward(model, inputs)]
    weights = [np.asarray(layer.layers[1].weight, dtype=np.float64) for layer in model.layers]
    biases = [np.asarray(layer.layers[1].bias, dtype=np.float64) for layer in model.layers]
    moments = [np.zeros_like(array) for array in activities[:-1]]
    variances = [np.zeros_like(array) for array in activities[:-1]]

    def residuals():
        errors = []
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            previous = inputs if index == 0 else np.tanh(activities[index - 1])
            target = labels if index == len(weights) - 1 else activities[index]
            errors.append(target - (previous @ weight.T + bias))
        return errors

    for step in range(1, steps + 1):
        errors = residuals()
        gradients = [errors[index] - (errors[index + 1] @ weights[index + 1])
                     * (1 - np.tanh(activities[index])**2)
                     for index in range(len(weights) - 1)]
        for index, gradient in enumerate(gradients):
            moments[index] = 0.9 * moments[index] + 0.1 * gradient
            variances[index] = 0.999 * variances[index] + 0.001 * gradient**2
            corrected_mean = moments[index] / (1 - 0.9**step)
            corrected_variance = variances[index] / (1 - 0.999**step)
            activities[index] -= activity_lr * corrected_mean / (np.sqrt(corrected_variance) + 1e-8)
    errors = residuals()
    energy = sum(0.5 * np.sum(error**2) for error in errors) / len(inputs)
    return activities, energy


@pytest.mark.parametrize("steps", [1, 5, 20])
@pytest.mark.parametrize("activity_lr", [0.001, 0.1, 1.0])
def test_real_activity_adam_matches_independent_moment_equations(notebook, steps, activity_lr):
    model = stable_model(notebook, activation="tanh")
    inputs = np.array([[1., -.5], [-.3, .7], [.2, .4]], dtype=np.float32)
    labels = np.eye(2, dtype=np.float32)[[0, 1, 0]]
    expected_activities, expected_energy = numpy_adam_settling(
        model, inputs, labels, steps=steps, activity_lr=activity_lr)
    p_optim = optax.adam(0.001)
    p_state = p_optim.init(eqx.filter(model, eqx.is_array))
    result = notebook["train_step"](model, inputs, labels, p_optim, p_state,
                                    optax.adam(activity_lr), T=steps, record_aux_details=False)
    for actual, expected in zip(result["activities"], expected_activities):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(result["energy"], expected_energy, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("seed", range(10))
def test_randomized_pc_shapes_depths_and_values_match_analytic_oracle(notebook, seed):
    rng = np.random.default_rng(seed)
    n_layers = int(rng.integers(1, 5))
    input_size, hidden_size, output_size = map(int, rng.integers(1, 6, size=3))
    batch_size = int(rng.integers(1, 8))
    model = make_model(notebook, seed=seed, n_layers=n_layers,
                       input_size=input_size, hidden_size=hidden_size,
                       output_size=output_size)
    model = jax.tree_util.tree_map(lambda leaf: leaf * 0.3 if eqx.is_array(leaf) else leaf, model)
    inputs = rng.normal(size=(batch_size, input_size)).astype(np.float32)
    labels = np.eye(output_size, dtype=np.float32)[rng.integers(0, output_size, size=batch_size)]
    steps = int(rng.choice([1, 5, 20]))
    activity_lr = float(rng.choice([0.01, 0.1, 0.5]))
    oracle = numpy_settling(model, inputs, labels, activation="tanh",
                            steps=steps, activity_lr=activity_lr)
    snapshot = array_snapshot(model)
    p_optim = optax.adam(0.001)
    p_state = p_optim.init(eqx.filter(model, eqx.is_array))
    result = notebook["train_step"](model, inputs, labels, p_optim, p_state,
                                    optax.sgd(activity_lr), T=steps, record_aux_details=False)
    for actual, expected in zip(result["activities"], oracle["activities"]):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result["energy"], oracle["energy"], rtol=3e-5, atol=3e-6)
    assert_arrays_unchanged(model, snapshot)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("seed", [0, 42, 43])
def test_real_experiments_are_reproducible_and_log_final_iteration(notebook, optimizer, seed):
    notebook["get_mnist_loaders"] = Mock(side_effect=lambda *args, **kwargs: tiny_loaders())
    config = tiny_config(notebook)
    config.update(seed=seed, activity_optimizer=optimizer, n_train_iters=5, test_every=2)
    original = copy.deepcopy(config)
    callback = Mock()
    first = notebook["run_task4_experiment"](config, callback)
    second = notebook["run_task4_experiment"](config)
    assert first["history"] == second["history"]
    assert [row["train/iteration"] for row in first["history"]] == [2, 4, 5]
    assert [call.args[0] for call in callback.call_args_list] == first["history"]
    assert np.isfinite(first["runtime_seconds"]) and first["runtime_seconds"] >= 0
    assert first["final_test_accuracy"] == first["history"][-1]["test/accuracy"]
    assert 0 <= first["final_test_accuracy"] <= 100
    assert config == original


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_real_thousand_update_experiment_learns_and_finishes_exactly(notebook, optimizer):
    notebook["get_mnist_loaders"] = Mock(side_effect=lambda *args, **kwargs: tiny_loaders())
    config = tiny_config(notebook)
    config.update(activity_optimizer=optimizer, n_train_iters=1000, test_every=333,
                  inference_steps=5)
    result = notebook["run_task4_experiment"](config)
    assert [row["train/iteration"] for row in result["history"]] == [333, 666, 999, 1000]
    assert result["final_test_accuracy"] == 100.0
    initial_model = make_model(notebook, input_size=2, hidden_size=4,
                                output_size=2, n_layers=2, seed=42)
    inputs, labels = tiny_loaders()[0][0]
    initial_energy = numpy_settling(initial_model, inputs.numpy(), labels.numpy(),
                                    activation="tanh", steps=0)["energy"]
    assert result["history"][-1]["train/energy"] < initial_energy * 0.25


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_real_divergent_training_is_rejected_before_metrics_are_logged(notebook, value):
    train_loader, test_loader = tiny_loaders()
    inputs, labels = train_loader[0]
    inputs = inputs.clone()
    inputs[0, 0] = value
    notebook["get_mnist_loaders"] = Mock(return_value=([(inputs, labels)], test_loader))
    callback = Mock()
    with pytest.raises(FloatingPointError):
        notebook["run_task4_experiment"](tiny_config(notebook), callback)
    callback.assert_not_called()


def test_real_mnist_loaders_seed_normalization_and_full_test_set(notebook, monkeypatch):
    folder = ROOT / "data" / "MNIST" / "raw"
    if not (folder / "t10k-images-idx3-ubyte").is_file():
        pytest.skip("Local MNIST data unavailable; tests never download it")
    monkeypatch.chdir(ROOT)
    # Prevent even an accidental download in this unit/integration suite.
    def offline_init(dataset, root, train=True, transform=None, target_transform=None,
                     download=False):
        assert download is True  # Notebook's argument remains covered.
        return original_init(dataset, root, train=train, transform=transform,
                             target_transform=target_transform, download=False)
    original_init = notebook["datasets"].MNIST.__init__
    monkeypatch.setattr(notebook["datasets"].MNIST, "__init__", offline_init)
    train_a, test = notebook["get_mnist_loaders"](64, seed=42)
    train_b, _ = notebook["get_mnist_loaders"](64, seed=42)
    train_c, _ = notebook["get_mnist_loaders"](64, seed=43)
    first_a, first_b, first_c = next(iter(train_a)), next(iter(train_b)), next(iter(train_c))
    assert torch.equal(first_a[0], first_b[0])
    assert torch.equal(first_a[1], first_b[1])
    assert not torch.equal(first_a[0], first_c[0])
    assert first_a[0].shape == (64, 784) and first_a[1].shape == (64, 10)
    assert train_a.drop_last and not test.drop_last
    assert len(test.dataset) == 10000
    sizes = [len(images) for images, _ in test]
    assert sum(sizes) == 10000 and sizes[-1] == 16
    raw = notebook["MNIST"](train=True, normalize=False)
    normalized = notebook["MNIST"](train=True, normalize=True)
    raw_image, raw_label = raw[0]
    norm_image, norm_label = normalized[0]
    np.testing.assert_allclose(norm_image, (raw_image - 0.1307) / 0.3081, atol=1e-6)
    assert torch.equal(raw_label, norm_label)
    assert torch.equal(notebook["one_hot"](torch.tensor([0, 3]), 4), torch.eye(4)[[0, 3]])
    model = make_model(notebook, input_size=784, hidden_size=4, output_size=10, n_layers=2)
    accuracy = notebook["evaluate"](model, test)
    assert 0 <= accuracy <= 100
    assert abs(accuracy * 100 - round(accuracy * 100)) < 1e-9
    # Run the complete experiment wrapper with the notebook's real 784/300/10
    # architecture and all 10,000 test images, using two training updates.
    experiment_config = config_for(notebook, n_train_iters=2, test_every=2,
                                    inference_steps=1)
    callback = Mock()
    result = notebook["run_task4_experiment"](experiment_config, callback)
    assert len(result["history"]) == 1
    assert result["history"][0]["train/iteration"] == 2
    assert 0 <= result["final_test_accuracy"] <= 100
    callback.assert_called_once_with(result["history"][0])


def offline_wandb_records(path):
    # W&B's documented transaction-log wire format; no private SDK imports.
    # https://github.com/wandb/wandb/blob/main/core/pkg/leveldb/record.go
    from wandb.proto import wandb_internal_pb2
    stream = path.read_bytes()
    assert struct.unpack("<4sHB", stream[:7]) == (b":W&B", 0xBEE1, 0)
    offset = 7
    fragments = bytearray()
    while offset < len(stream):
        available = 32768 - offset % 32768
        if available < 7:
            assert stream[offset:offset + available] == bytes(available)
            offset += available
            continue
        checksum, length, kind = struct.unpack("<IHB", stream[offset:offset + 7])
        payload = stream[offset + 7:offset + 7 + length]
        assert len(payload) == length
        assert checksum == zlib.crc32(bytes([kind]) + payload) & 0xffffffff
        offset += 7 + length
        assert kind in (1, 2, 3, 4)
        if kind in (1, 2):
            assert not fragments
        fragments.extend(payload)
        if kind in (1, 4):
            record = wandb_internal_pb2.Record()
            record.ParseFromString(bytes(fragments))
            fragments.clear()
            yield record
    assert not fragments


def test_real_wandb_offline_smoke_flushes_metrics_and_summary(notebook, tmp_path):
    import wandb
    notebook["wandb"] = wandb
    notebook["get_mnist_loaders"] = Mock(side_effect=lambda *args, **kwargs: tiny_loaders())
    notebook["TASK4_BASE_CONFIG"] = tiny_config(notebook)
    notebook["TASK4_BASE_CONFIG"].update(inference_steps=1)
    execute_cell(cell_with("smoke_config ="), notebook)
    result = notebook["smoke_result"]
    assert result["history"][-1]["train/iteration"] == 2
    assert wandb.run is None
    run_files = list(tmp_path.glob(f"wandb/offline-run-*-{result['run_id']}/run-*.wandb"))
    assert len(run_files) == 1 and run_files[0].stat().st_size > 0
    # Read the SDK's real persisted event stream; do not rely on empty/stale
    # wandb-summary.json files in offline mode.
    summary = {}
    history = []
    finished = False
    for record in offline_wandb_records(run_files[0]):
        if record.HasField("summary"):
            for item in record.summary.update:
                summary[item.key] = json.loads(item.value_json)
        if record.HasField("history"):
            row = {item.key or "/".join(item.nested_key): json.loads(item.value_json)
                   for item in record.history.item}
            history.append(row)
        if record.HasField("exit"):
            finished = record.exit.exit_code == 0
    assert finished
    assert summary["train/iterations_completed"] == 2
    assert summary["test/final_accuracy"] == result["final_test_accuracy"]
    assert summary["runtime/seconds"] == result["runtime_seconds"]
    assert "test/accuracy_at_1000" not in summary
    assert len(history) == 1
    for key, value in result["history"][0].items():
        assert history[0][key] == value
    (RESULTS / "offline_smoke_summary.json").write_text(json.dumps(summary, indent=2))
