"""Snapshot and execute notebook cells verbatim under their coverage filenames."""

import ast
import json
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "pc_in_jax.ipynb"
RESULTS = ROOT / "test-results" / "pc_in_jax"
NOTEBOOK_SOURCES = RESULTS / "notebook_cells"
TASK_SOURCES = NOTEBOOK_SOURCES / "tasks_23"
TASK45_SOURCES = NOTEBOOK_SOURCES / "tasks_45"
SUPPORT_SOURCES = NOTEBOOK_SOURCES / "data_and_training"

SUPPORT_SYMBOLS = {
    "FeedforwardPCModel",
    "energy_fn",
    "batched_energy_fn",
    "batched_energy_sum",
    "batched_energy_mean",
    "init_activities_with_ffwd",
    "compute_activity_grad",
    "compute_param_grad",
    "update_activities",
    "update_params",
    "train_step",
}


@lru_cache(maxsize=1)
def notebook_cells():
    """Read once so an autosave cannot mix different versions during a run."""
    return json.loads(NOTEBOOK.read_text())["cells"]


def task_cell_index(task):
    cells = notebook_cells()
    headings = [
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "markdown"
        and re.search(rf"^#+\s+Task {task}:\s", "".join(cell["source"]), re.M)
    ]
    if len(headings) != 1:
        raise ValueError(f"Expected one Task {task} heading, found {headings}")
    cell = cells[headings[0] + 1]
    if cell["cell_type"] != "code":
        raise ValueError(f"Task {task} is not immediately followed by a code cell")
    return headings[0] + 1


def task_source(task):
    return "".join(notebook_cells()[task_cell_index(task)]["source"])


@lru_cache(maxsize=1)
def extract_notebook_cells():
    """Extract every Python cell, excluding only the IPython pip setup cell."""
    task_indices = {task_cell_index(task) for task in (2, 3)}
    paths = {}
    task = None
    for index, cell in enumerate(notebook_cells()):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = re.search(r"^#+\s+(?:Bonus\s+)?Task (\d+):", source, re.M)
            if heading:
                task = int(heading.group(1))
            continue
        if cell["cell_type"] != "code":
            continue
        if source.lstrip().startswith("%pip "):
            continue
        tree = ast.parse(source)  # Other syntax errors must fail the run.
        definitions = {node.name for node in tree.body if isinstance(
            node, (ast.FunctionDef, ast.ClassDef))}
        if index in task_indices:
            folder = TASK_SOURCES
        elif task in (4, 5):
            folder = TASK45_SOURCES
        elif definitions & {"get_mnist_loaders", "train"}:
            folder = SUPPORT_SOURCES
        elif definitions:
            folder = NOTEBOOK_SOURCES / "core"
        else:
            folder = NOTEBOOK_SOURCES / "examples"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"cell_{index}.py"
        path.write_text(source)
        paths[index] = path
    return paths


def extract_task_cells():
    paths = extract_notebook_cells()
    return {task: paths[task_cell_index(task)] for task in (2, 3)}


def execute_notebook_cell(index, namespace, *, definitions_only=False):
    path = extract_notebook_cells()[index]
    source = "".join(notebook_cells()[index]["source"])
    if path.read_text() != source:
        raise AssertionError("Extracted test source differs from the notebook snapshot")
    if definitions_only:
        tree = ast.parse(source, filename=str(path))
        tree.body = [node for node in tree.body if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
        code = compile(tree, str(path), "exec")
    else:
        code = compile(source, str(path), "exec")
    exec(code, namespace)
    return namespace


def load_support(cells=None):
    """Execute only the definitions the cells need; use the actual notebook code."""
    from typing import Callable

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import optax

    namespace = dict(Callable=Callable, eqx=eqx, jax=jax, jnp=jnp,
                     jr=jr, plt=plt, optax=optax)
    found = set()
    for index, cell in enumerate(notebook_cells() if cells is None else cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:  # The installation cell contains IPython magic.
            continue
        definitions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        if definitions & SUPPORT_SYMBOLS:
            path = extract_notebook_cells()[index]
            if path.read_text() != source:
                raise AssertionError("Support source differs from the notebook snapshot")
            exec(compile(source, str(path), "exec"), namespace)
            found.update(definitions & SUPPORT_SYMBOLS)
    if found != SUPPORT_SYMBOLS:
        raise ValueError(f"Missing notebook definitions: {SUPPORT_SYMBOLS - found}")
    return namespace


def execute_task(task, namespace):
    return execute_notebook_cell(task_cell_index(task), namespace)
