"""Behavioral and integration tests for the cells immediately after Tasks 2/3."""

import os
import struct
from types import SimpleNamespace
from unittest.mock import Mock

os.environ["MPLBACKEND"] = "Agg"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytest

from notebook_test_support import (
    RESULTS, ROOT, execute_task, extract_task_cells, load_support, task_source,
)


@pytest.fixture(scope="session")
def notebook():
    extract_task_cells()
    return load_support()


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def make_model(notebook, *, input_size=2, hidden_size=3, output_size=2,
               n_layers=3, activation="tanh", use_bias=True, seed=42,
               input_activation="identity"):
    activations = {
        "identity": eqx.nn.Identity(),
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
    }
    return notebook["FeedforwardPCModel"](
        jax.random.PRNGKey(seed), input_size=input_size,
        hidden_size=hidden_size, output_size=output_size,
        activation_fn=activations[activation], n_layers=n_layers,
        input_activation_fn=activations[input_activation], use_bias=use_bias,
    )


def numpy_activation(name, values):
    if name == "identity":
        return values
    if name == "tanh":
        return np.tanh(values)
    if name == "relu":
        return np.maximum(values, 0)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-values))
    raise ValueError(name)


def numpy_feedforward(model, inputs, activation="tanh",
                      input_activation="identity"):
    current = np.asarray(inputs)
    activities = []
    for index, layer in enumerate(model.layers):
        linear = layer.layers[1]
        current = numpy_activation(
            input_activation if index == 0 else activation, current)
        current = current @ np.asarray(linear.weight).T
        if linear.bias is not None:
            current = current + np.asarray(linear.bias)
        activities.append(current.copy())
    return activities


def array_snapshot(tree):
    return [np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(tree)
            if eqx.is_array(leaf)]


def assert_arrays_unchanged(tree, snapshot):
    current = array_snapshot(tree)
    assert len(current) == len(snapshot)
    for actual, expected in zip(current, snapshot):
        np.testing.assert_array_equal(actual, expected)


MODEL_CASES = [
    pytest.param(1, 1, 1, 1, 1, "identity", False, id="scalar-one-layer"),
    pytest.param(1, 2, 3, 2, 2, "tanh", True, id="singleton-batch"),
    pytest.param(5, 4, 2, 3, 3, "relu", False, id="relu-no-bias"),
    pytest.param(7, 5, 3, 4, 4, "sigmoid", True, id="four-layers"),
    pytest.param(3, 2, 1, 2, 5, "tanh", True, id="five-layers"),
    pytest.param(64, 784, 300, 10, 3, "tanh", True, id="mnist-dimensions"),
]


@pytest.mark.parametrize("container", ["numpy", "jax"])
@pytest.mark.parametrize("pattern", ["zeros", "mixed", "large"])
@pytest.mark.parametrize(
    "batch_size,input_size,hidden_size,output_size,n_layers,activation,use_bias",
    MODEL_CASES,
)
def test_task2_shapes_values_prints_and_no_mutation(
        notebook, capsys, container, pattern, batch_size, input_size,
        hidden_size, output_size, n_layers, activation, use_bias):
    model = make_model(
        notebook, input_size=input_size, hidden_size=hidden_size,
        output_size=output_size, n_layers=n_layers,
        activation=activation, use_bias=use_bias,
    )
    inputs = np.linspace(-1, 1, batch_size * input_size, dtype=np.float32)
    inputs = inputs.reshape(batch_size, input_size)
    if pattern == "zeros":
        inputs[:] = 0
    elif pattern == "large":
        inputs *= 100
    if container == "jax":
        inputs = jnp.asarray(inputs)
    original_inputs = np.asarray(inputs).copy()
    original_model = array_snapshot(model)
    namespace = dict(notebook, model=model, img_batch=inputs)
    execute_task(2, namespace)

    expected = numpy_feedforward(model, inputs, activation)
    # Float32 matrix-matrix and matrix-vector reductions can differ by about
    # 1e-4 for 784-term dot products when input magnitudes reach 100.
    numeric_atol = 1e-4 if pattern == "large" else 3e-5
    assert isinstance(namespace["activities"], list)
    assert isinstance(namespace["activities_one"], list)
    assert len(namespace["activities"]) == n_layers
    assert len(namespace["activities_one"]) == n_layers
    expected_prints = []
    for index, (single, batched, oracle) in enumerate(zip(
            namespace["activities_one"], namespace["activities"], expected)):
        width = output_size if index == n_layers - 1 else hidden_size
        assert single.shape == (width,)
        assert batched.shape == (batch_size, width)
        assert isinstance(batched, jax.Array)
        assert np.issubdtype(batched.dtype, np.floating)
        np.testing.assert_allclose(single, oracle[0], rtol=3e-5, atol=numeric_atol)
        np.testing.assert_allclose(batched, oracle, rtol=3e-5, atol=numeric_atol)
        # The batched result must also equal separate calls for every example.
        separately = np.stack([
            np.asarray(notebook["init_activities_with_ffwd"](model, row)[index])
            for row in inputs
        ])
        np.testing.assert_allclose(batched, separately, rtol=3e-5, atol=numeric_atol)
        expected_prints.append(f"single example, activity z_{index}: {(width,)}")
    expected_prints.extend(
        f"batch, activity z_{index}: {activity.shape}"
        for index, activity in enumerate(namespace["activities"])
    )
    assert capsys.readouterr().out.splitlines() == expected_prints
    np.testing.assert_array_equal(inputs, original_inputs)
    assert_arrays_unchanged(model, original_model)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32, np.uint8])
@pytest.mark.parametrize("container", ["numpy", "jax"])
def test_task2_numeric_input_dtypes(notebook, dtype, container):
    model = make_model(notebook, activation="identity")
    inputs = np.arange(6, dtype=dtype).reshape(3, 2)
    if container == "jax":
        inputs = jnp.asarray(inputs)
    result = execute_task(2, dict(notebook, model=model, img_batch=inputs))
    for actual, expected in zip(
            result["activities"], numpy_feedforward(model, inputs, "identity")):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("input_activation", ["tanh", "relu", "sigmoid"])
def test_task2_nonidentity_input_activation(notebook, input_activation):
    model = make_model(notebook, input_activation=input_activation)
    inputs = np.array([[-1, 0], [0, 1], [2, -2]], dtype=np.float32)
    result = execute_task(2, dict(notebook, model=model, img_batch=inputs))
    for actual, expected in zip(result["activities"], numpy_feedforward(
            model, inputs, input_activation=input_activation)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_task2_batch_permutation_duplication_and_reexecution(notebook):
    model = make_model(notebook)
    inputs = np.array([[1, -1], [0, 2], [-3, 1]], dtype=np.float32)
    namespace = dict(notebook, model=model, img_batch=inputs)
    execute_task(2, namespace)
    original = [np.asarray(a).copy() for a in namespace["activities"]]
    indices = [2, 0, 2, 1]
    namespace["img_batch"] = inputs[indices]
    execute_task(2, namespace)
    for actual, expected in zip(namespace["activities"], original):
        np.testing.assert_allclose(actual, expected[indices], atol=1e-6)
    namespace["img_batch"] = inputs
    execute_task(2, namespace)
    for actual, expected in zip(namespace["activities"], original):
        np.testing.assert_array_equal(actual, expected)


def test_task2_zero_width_output(notebook):
    model = make_model(notebook, output_size=0, n_layers=2)
    result = execute_task(2, dict(
        notebook, model=model, img_batch=np.ones((3, 2), dtype=np.float32)))
    assert result["activities_one"][-1].shape == (0,)
    assert result["activities"][-1].shape == (3, 0)


@pytest.mark.parametrize("container", ["numpy", "jax"])
def test_task2_empty_batch_raises_index_error(notebook, container):
    inputs = np.empty((0, 2), dtype=np.float32)
    if container == "jax":
        inputs = jnp.asarray(inputs)
    with pytest.raises(IndexError):
        execute_task(2, dict(notebook, model=make_model(notebook), img_batch=inputs))


@pytest.mark.parametrize("shape", [(3,), (2, 2, 2), (2, 1), (2, 3)])
@pytest.mark.parametrize("container", ["numpy", "jax"])
def test_task2_invalid_rank_or_feature_size(notebook, shape, container):
    inputs = np.ones(shape, dtype=np.float32)
    if container == "jax":
        inputs = jnp.asarray(inputs)
    with pytest.raises((TypeError, ValueError)):
        execute_task(2, dict(notebook, model=make_model(notebook), img_batch=inputs))


@pytest.mark.parametrize("dtype", [object, "U1"])
def test_task2_nonnumeric_inputs_fail(notebook, dtype):
    inputs = np.full((2, 2), "x", dtype=dtype)
    with pytest.raises((TypeError, ValueError)):
        execute_task(2, dict(notebook, model=make_model(notebook), img_batch=inputs))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_task2_nonfinite_values_propagate_without_shape_failure(notebook, value):
    inputs = np.full((2, 2), value, dtype=np.float32)
    model = make_model(notebook, activation="identity")
    result = execute_task(2, dict(notebook, model=model, img_batch=inputs))
    assert all(a.shape[0] == 2 for a in result["activities"])
    assert all(not np.isfinite(np.asarray(a)).all() for a in result["activities"])


@pytest.mark.parametrize("bad_layer,bad_shape", [
    (0, (2, 2)), (0, (3, 2, 1)), (1, (3, 4)),
])
def test_task2_shape_assertion_rejects_bad_vmap_results(
        notebook, bad_layer, bad_shape):
    single = [jnp.zeros((2,)), jnp.zeros((3,))]
    mapped = [jnp.zeros((3, 2)), jnp.zeros((3, 3))]
    mapped[bad_layer] = jnp.zeros(bad_shape)
    fake_vmap = Mock(return_value=Mock(return_value=mapped))
    namespace = dict(notebook, model=object(), img_batch=np.zeros((3, 2)),
                     init_activities_with_ffwd=Mock(return_value=single),
                     jax=SimpleNamespace(vmap=fake_vmap))
    with pytest.raises(AssertionError):
        execute_task(2, namespace)
    fake_vmap.assert_called_once_with(
        namespace["init_activities_with_ffwd"], in_axes=(None, 0))


def test_task2_empty_activity_iterables(notebook, capsys):
    namespace = dict(notebook, model=object(), img_batch=np.zeros((3, 2)),
                     init_activities_with_ffwd=Mock(return_value=[]),
                     jax=SimpleNamespace(vmap=Mock(return_value=Mock(return_value=[]))))
    execute_task(2, namespace)
    assert namespace["activities_one"] == namespace["activities"] == []
    assert capsys.readouterr().out == ""


def plotting_proxy():
    def subplots(*args, **kwargs):
        figure, axes = plt.subplots(*args, **kwargs)
        figure.tight_layout = Mock(wraps=figure.tight_layout)
        return figure, axes
    return SimpleNamespace(subplots=Mock(side_effect=subplots), show=Mock())


def task3_namespace(notebook, *, aux=None, model=None, inputs=None, labels=None,
                    activity_lr=0.1, param_lr=0.001, real_training=False):
    if model is None:
        model = make_model(notebook)
    if inputs is None:
        inputs = np.array([[1, -1], [0, 2], [-2, 1]], dtype=np.float32)
    if labels is None:
        labels = np.eye(2, dtype=np.float32)[[0, 1, 0]]
    namespace = dict(notebook, model=model, img_batch=inputs, label_batch=labels,
                     ACTIVITY_LR=activity_lr, PARAM_LR=param_lr,
                     plt=plotting_proxy())
    if not real_training:
        namespace["train_step"] = Mock(return_value={"aux": aux})
    return namespace


def assert_plot(namespace, expected_aux):
    figure, axes = namespace["fig"], namespace["ax"]
    namespace["plt"].subplots.assert_called_once_with(figsize=(8, 5))
    namespace["plt"].show.assert_called_once_with()
    figure.tight_layout.assert_called_once_with()
    np.testing.assert_allclose(figure.get_size_inches(), [8, 5])
    assert axes.get_xlabel() == "Settling step (before update)"
    assert axes.get_ylabel() == "Energy (sum over batch)"
    assert axes.get_title() == "Per-layer energies over 50 settling steps"
    assert len(axes.lines) == len(expected_aux)
    expected_labels = [name.removeprefix("energy/") for name in expected_aux]
    assert [text.get_text() for text in axes.get_legend().texts] == expected_labels
    for line, (name, energies) in zip(axes.lines, expected_aux.items()):
        np.testing.assert_array_equal(line.get_xdata(), np.arange(50))
        np.testing.assert_array_equal(line.get_ydata(), np.asarray(energies))
        assert line.get_label() == name.removeprefix("energy/")
    grid_lines = axes.get_xgridlines() + axes.get_ygridlines()
    assert grid_lines
    assert all(line.get_visible() and line.get_alpha() == 0.3 for line in grid_lines)


@pytest.mark.parametrize("activity_lr,param_lr", [(0.1, 0.001), (0, 0), (0.05, 0.01)])
def test_task3_optimizer_initialization_and_train_step_arguments(
        notebook, activity_lr, param_lr):
    aux = {"energy/layer_0": jnp.zeros(50), "energy/output": jnp.ones(50)}
    namespace = task3_namespace(
        notebook, aux=aux, activity_lr=activity_lr, param_lr=param_lr)
    namespace["optax"] = SimpleNamespace(
        sgd=Mock(wraps=optax.sgd), adam=Mock(wraps=optax.adam))
    namespace["eqx"] = SimpleNamespace(filter=Mock(wraps=eqx.filter), is_array=eqx.is_array)
    original_model = array_snapshot(namespace["model"])
    execute_task(3, namespace)
    assert namespace["settling_steps"] == 50
    namespace["optax"].sgd.assert_called_once_with(activity_lr)
    namespace["optax"].adam.assert_called_once_with(param_lr)
    namespace["eqx"].filter.assert_called_once_with(namespace["model"], eqx.is_array)
    args, kwargs = namespace["train_step"].call_args
    assert len(args) == 6
    for actual, name in zip(args, [
            "model", "img_batch", "label_batch", "p_optim", "p_opt_state", "x_optim"]):
        assert actual is namespace[name]
    assert kwargs == {"T": 50, "record_aux_details": True}
    namespace["train_step"].assert_called_once()
    assert namespace["settling_out"] is namespace["train_step"].return_value
    expected_state = namespace["p_optim"].init(eqx.filter(namespace["model"], eqx.is_array))
    assert jax.tree_util.tree_structure(namespace["p_opt_state"]) == jax.tree_util.tree_structure(expected_state)
    for actual, expected in zip(array_snapshot(namespace["p_opt_state"]), array_snapshot(expected_state)):
        np.testing.assert_array_equal(actual, expected)
    assert_arrays_unchanged(namespace["model"], original_model)
    assert_plot(namespace, aux)


PLOT_CASES = [
    pytest.param({}, id="no-curves"),
    pytest.param({"energy/output": np.zeros(50)}, id="one-zero-curve"),
    pytest.param({"energy/layer_0": np.arange(50),
                  "energy/layer_1": jnp.arange(50)[::-1],
                  "energy/output": list(np.linspace(2, 0, 50))}, id="mixed-containers"),
    pytest.param({"energy/output": np.ones(50), "energy/layer_0": np.zeros(50)},
                 id="insertion-order"),
    pytest.param({"custom/energy/output": np.ones(50),
                  "energy/energy/output": np.zeros(50)}, id="prefix-only-removal"),
    pytest.param({"energy/output": np.tile([0, np.nan, np.inf, -np.inf, 1], 10)},
                 id="nonfinite-plot-values"),
]


@pytest.mark.parametrize("aux", PLOT_CASES)
def test_task3_plot_contents_and_formatting(notebook, aux):
    namespace = task3_namespace(notebook, aux=aux)
    execute_task(3, namespace)
    assert_plot(namespace, aux)


def test_task3_training_failure_propagates_before_plot_creation(notebook):
    namespace = task3_namespace(notebook, aux={})
    error = RuntimeError("training failed")
    namespace["train_step"].side_effect = error
    with pytest.raises(RuntimeError) as caught:
        execute_task(3, namespace)
    assert caught.value is error
    namespace["plt"].subplots.assert_not_called()
    namespace["plt"].show.assert_not_called()


@pytest.mark.parametrize("length", [0, 1, 49, 51])
def test_task3_trace_length_mismatch_raises_value_error(notebook, length):
    namespace = task3_namespace(notebook, aux={"energy/output": np.zeros(length)})
    with pytest.raises(ValueError, match="same first dimension"):
        execute_task(3, namespace)
    namespace["plt"].show.assert_not_called()


def test_task3_missing_aux_raises_key_error(notebook):
    namespace = task3_namespace(notebook)
    namespace["train_step"].return_value = {}
    with pytest.raises(KeyError, match="aux"):
        execute_task(3, namespace)
    namespace["plt"].show.assert_not_called()


@pytest.mark.parametrize("aux", [None, [], {1: np.ones(50)}])
def test_task3_malformed_aux_schema_fails(notebook, aux):
    namespace = task3_namespace(notebook, aux=aux)
    with pytest.raises(AttributeError):
        execute_task(3, namespace)
    namespace["plt"].show.assert_not_called()


@pytest.mark.parametrize("missing", [
    "ACTIVITY_LR", "PARAM_LR", "model", "img_batch", "label_batch",
])
def test_task3_missing_notebook_prerequisite_fails(notebook, missing):
    namespace = task3_namespace(notebook, aux={})
    del namespace[missing]
    with pytest.raises(NameError, match=missing):
        execute_task(3, namespace)
    namespace["train_step"].assert_not_called()
    namespace["plt"].show.assert_not_called()


@pytest.mark.parametrize("task", [2, 3])
def test_coverage_source_is_identical_to_notebook(task):
    paths = extract_task_cells()
    assert paths[task].read_text() == task_source(task)


def stable_model(notebook, *, n_layers=3, use_bias=True, activation="identity"):
    model = make_model(notebook, n_layers=n_layers, use_bias=use_bias,
                       activation=activation)
    for index, layer in enumerate(model.layers):
        linear = layer.layers[1]
        weights = jnp.linspace(-0.25, 0.3, linear.weight.size).reshape(linear.weight.shape)
        model = eqx.tree_at(lambda m: m.layers[index].layers[1].weight, model, weights)
        if use_bias:
            biases = jnp.linspace(-0.12, 0.08, linear.bias.size)
            model = eqx.tree_at(lambda m: m.layers[index].layers[1].bias, model, biases)
    return model


def activation_derivative(name, values):
    if name == "identity":
        return np.ones_like(values)
    if name == "tanh":
        return 1 - np.tanh(values) ** 2
    if name == "relu":
        return (values > 0).astype(values.dtype)
    if name == "sigmoid":
        sigmoid = numpy_activation("sigmoid", values)
        return sigmoid * (1 - sigmoid)
    raise ValueError(name)


def numpy_settling(model, inputs, labels, *, activation="identity", steps=50,
                   activity_lr=0.1):
    """Independent analytic PC gradients, without JAX/autodiff/Optax."""
    inputs, labels = np.asarray(inputs), np.asarray(labels)
    activities = numpy_feedforward(model, inputs, activation)
    initial = [activity.copy() for activity in activities]
    weights = [np.asarray(layer.layers[1].weight) for layer in model.layers]
    biases = [None if layer.layers[1].bias is None else np.asarray(layer.layers[1].bias)
              for layer in model.layers]
    keys = [f"energy/layer_{index}" for index in range(len(model) - 1)] + ["energy/output"]
    traces = {key: [] for key in keys}

    def residuals():
        errors = []
        previous = inputs
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            transformed = previous if index == 0 else numpy_activation(activation, previous)
            prediction = transformed @ weight.T
            if bias is not None:
                prediction = prediction + bias
            target = labels if index == len(model) - 1 else activities[index]
            errors.append(target - prediction)
            previous = activities[index]
        return errors

    for _ in range(steps):
        errors = residuals()
        for key, error in zip(keys, errors):
            traces[key].append(np.float32(0.5) * np.sum(error ** 2))
        gradients = [
            errors[index] - (errors[index + 1] @ weights[index + 1])
            * activation_derivative(activation, activities[index])
            for index in range(len(model) - 1)
        ]
        for index, gradient in enumerate(gradients):
            activities[index] = activities[index] - np.float32(activity_lr) * gradient

    errors = residuals()
    weight_gradients, bias_gradients = [], []
    for index, error in enumerate(errors):
        previous = inputs if index == 0 else numpy_activation(activation, activities[index - 1])
        weight_gradients.append(-(error.T @ previous) / len(inputs))
        bias_gradients.append(-np.mean(error, axis=0))
    energy = sum(float(np.float32(0.5) * np.sum(error ** 2)) for error in errors) / len(inputs)
    return dict(traces={key: np.asarray(values) for key, values in traces.items()},
                activities=activities, initial=initial, energy=energy,
                weight_gradients=weight_gradients, bias_gradients=bias_gradients)


def assert_real_training(namespace, oracle, *, tolerance=3e-6):
    result = namespace["settling_out"]
    assert set(result) == {"model", "p_opt_state", "activities", "energy", "aux"}
    assert set(result["aux"]) == set(oracle["traces"])
    for key, expected in oracle["traces"].items():
        actual = np.asarray(result["aux"][key])
        assert actual.shape == (50,)
        assert np.isfinite(actual).all()
        assert (actual >= 0).all()
        # Aux values are sums across examples and are recorded BEFORE updates.
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=tolerance)
    for actual, expected in zip(result["activities"], oracle["activities"]):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=tolerance)
    np.testing.assert_allclose(result["activities"][-1], oracle["initial"][-1],
                               rtol=3e-5, atol=tolerance)
    # Returned scalar energy is the batch mean AFTER the 50 activity updates.
    np.testing.assert_allclose(result["energy"], oracle["energy"],
                               rtol=3e-5, atol=tolerance)
    totals = np.sum(np.stack(list(result["aux"].values())), axis=0)
    assert (np.diff(totals) <= tolerance).all()

    state = result["p_opt_state"][0]
    assert int(state.count) == 1
    assert int(namespace["p_opt_state"][0].count) == 0
    for index, (old_layer, new_layer) in enumerate(zip(
            namespace["model"].layers, result["model"].layers)):
        old, new = old_layer.layers[1], new_layer.layers[1]
        first_moment = state.mu.layers[index].layers[1]
        second_moment = state.nu.layers[index].layers[1]
        np.testing.assert_allclose(first_moment.weight, 0.1 * oracle["weight_gradients"][index],
                                   rtol=5e-5, atol=tolerance)
        np.testing.assert_allclose(second_moment.weight, 0.001 * oracle["weight_gradients"][index] ** 2,
                                   rtol=1e-4, atol=tolerance)
        for attribute in ("weight", "bias"):
            before = getattr(old, attribute)
            after = getattr(new, attribute)
            if before is None:
                assert after is None
                continue
            mean = np.asarray(getattr(first_moment, attribute)) / (1 - 0.9)
            variance = np.asarray(getattr(second_moment, attribute)) / (1 - 0.999)
            expected = np.asarray(before) - namespace["PARAM_LR"] * mean / (np.sqrt(variance) + 1e-8)
            np.testing.assert_allclose(after, expected, rtol=3e-5, atol=tolerance)
        if old.bias is not None:
            np.testing.assert_allclose(first_moment.bias, 0.1 * oracle["bias_gradients"][index],
                                       rtol=5e-5, atol=tolerance)
            np.testing.assert_allclose(second_moment.bias, 0.001 * oracle["bias_gradients"][index] ** 2,
                                       rtol=1e-4, atol=tolerance)
    assert_plot(namespace, result["aux"])


@pytest.mark.parametrize("n_layers", [2, 3, 4])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_task3_real_training_matches_analytic_numpy_oracle(
        notebook, n_layers, use_bias, batch_size):
    model = stable_model(notebook, n_layers=n_layers, use_bias=use_bias)
    inputs = np.linspace(-1, 1, batch_size * 2, dtype=np.float32).reshape(batch_size, 2)
    labels = np.eye(2, dtype=np.float32)[np.arange(batch_size) % 2]
    namespace = task3_namespace(notebook, model=model, inputs=inputs,
                                labels=labels, real_training=True)
    original_model = array_snapshot(model)
    original_inputs, original_labels = inputs.copy(), labels.copy()
    oracle = numpy_settling(model, inputs, labels)
    execute_task(3, namespace)
    assert_real_training(namespace, oracle)
    assert_arrays_unchanged(model, original_model)
    np.testing.assert_array_equal(inputs, original_inputs)
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
def test_task3_real_nonlinear_training_matches_numpy_oracle(notebook, activation):
    model = stable_model(notebook, activation=activation)
    namespace = task3_namespace(notebook, model=model, real_training=True)
    oracle = numpy_settling(model, namespace["img_batch"], namespace["label_batch"],
                            activation=activation)
    execute_task(3, namespace)
    assert_real_training(namespace, oracle)


@pytest.mark.parametrize("activity_lr,param_lr", [(0, 0.001), (0.1, 0), (0, 0)])
def test_task3_zero_learning_rates(notebook, activity_lr, param_lr):
    model = stable_model(notebook)
    namespace = task3_namespace(notebook, model=model, real_training=True,
                                activity_lr=activity_lr, param_lr=param_lr)
    oracle = numpy_settling(model, namespace["img_batch"], namespace["label_batch"],
                            activity_lr=activity_lr)
    execute_task(3, namespace)
    assert_real_training(namespace, oracle)
    if activity_lr == 0:
        for actual, expected in zip(namespace["settling_out"]["activities"], oracle["initial"]):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        for values in namespace["settling_out"]["aux"].values():
            np.testing.assert_allclose(values, np.full(50, values[0]), atol=1e-6)
    if param_lr == 0:
        assert_arrays_unchanged(namespace["settling_out"]["model"], array_snapshot(model))


@pytest.mark.parametrize("inputs,labels", [
    (np.ones((3, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)),
    (np.ones((3, 2), dtype=np.float32), np.ones((3, 3), dtype=np.float32)),
    (np.ones((3, 3), dtype=np.float32), np.ones((3, 2), dtype=np.float32)),
])
def test_task3_real_training_rejects_incompatible_data_shapes(notebook, inputs, labels):
    namespace = task3_namespace(notebook, inputs=inputs, labels=labels, real_training=True)
    with pytest.raises((TypeError, ValueError)):
        execute_task(3, namespace)
    namespace["plt"].subplots.assert_not_called()
    namespace["plt"].show.assert_not_called()


def local_mnist_batch(batch_size=64):
    """Read existing IDX files; no dataset download or Torch dependency."""
    folder = ROOT / "data" / "MNIST" / "raw"
    image_file = folder / "train-images-idx3-ubyte"
    label_file = folder / "train-labels-idx1-ubyte"
    if not image_file.is_file() or not label_file.is_file():
        pytest.skip("Local MNIST IDX files are unavailable; synthetic MNIST-size tests still run")
    with image_file.open("rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        assert magic == 2051 and count >= batch_size and (rows, columns) == (28, 28)
        images = np.frombuffer(handle.read(batch_size * rows * columns), dtype=np.uint8)
    with label_file.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        assert magic == 2049 and count >= batch_size
        labels = np.frombuffer(handle.read(batch_size), dtype=np.uint8)
    inputs = images.astype(np.float32).reshape(batch_size, 784) / np.float32(255)
    inputs = (inputs - np.float32(0.1307)) / np.float32(0.3081)
    return inputs, np.eye(10, dtype=np.float32)[labels]


def test_task2_and_task3_with_local_mnist_and_notebook_dimensions(notebook):
    inputs, labels = local_mnist_batch()
    model = make_model(notebook, input_size=784, hidden_size=300,
                       output_size=10, n_layers=3, seed=42)
    namespace = task3_namespace(notebook, model=model, inputs=inputs,
                                labels=labels, real_training=True)
    execute_task(2, namespace)
    assert [activity.shape for activity in namespace["activities"]] == [(64, 300), (64, 300), (64, 10)]
    oracle = numpy_settling(model, inputs, labels, activation="tanh")
    execute_task(3, namespace)
    assert_real_training(namespace, oracle, tolerance=5e-5)
    namespace["fig"].savefig(RESULTS / "task3_mnist_energy_trace.png", dpi=140)


def test_task3_one_layer_model_clamps_output_to_labels(notebook):
    """The PC energy equation also requires output clamping at depth one."""
    model = stable_model(notebook, n_layers=1)
    namespace = task3_namespace(notebook, model=model, real_training=True)
    oracle = numpy_settling(model, namespace["img_batch"], namespace["label_batch"])
    execute_task(3, namespace)
    # This exposes a dependency defect if the first-layer branch wins over
    # the final-layer branch in energy_fn when the model has only one layer.
    np.testing.assert_allclose(
        namespace["settling_out"]["energy"], oracle["energy"],
        rtol=3e-5, atol=3e-6,
        err_msg="A one-layer PC model must compare its prediction with the target labels.",
    )
    assert_real_training(namespace, oracle)
