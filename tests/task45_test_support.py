"""Execute all Task 4/5 cells verbatim, with controllable external services."""

import re
from functools import lru_cache
from pathlib import Path

from notebook_test_support import (
    ROOT, SUPPORT_SOURCES, TASK45_SOURCES, execute_notebook_cell,
    extract_notebook_cells, load_support, notebook_cells,
)


RESULTS = ROOT / "test-results" / "pc_in_jax_task45"
TASK_SOURCES = TASK45_SOURCES


@lru_cache(maxsize=None)
def source_snapshot():
    # A running notebook may autosave while tests run. Read its cells once.
    return notebook_cells()


@lru_cache(maxsize=None)
def selected_cells():
    """Include every code cell in each section, including commented examples."""
    selected = {}
    task = None
    for index, cell in enumerate(source_snapshot()):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = re.search(r"^#+\s+(?:Bonus\s+)?Task (\d+):", source, re.M)
            if heading:
                task = int(heading.group(1))
        elif task in (4, 5):
            selected[index] = (task, source)
    if {task for task, _ in selected.values()} != {4, 5}:
        raise ValueError("Could not locate both Task 4 and Task 5")
    return selected


@lru_cache(maxsize=None)
def extract_cells():
    all_paths = extract_notebook_cells()
    paths = {}
    for index, (task, source) in selected_cells().items():
        paths[index] = all_paths[index]
    for index, cell in enumerate(source_snapshot()):
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and (
                "def get_mnist_loaders(" in source or "def train(" in source):
            paths[index] = all_paths[index]
    RESULTS.mkdir(parents=True, exist_ok=True)
    return paths


def execute_cell(index, namespace, *, definitions_only=False):
    return execute_notebook_cell(index, namespace, definitions_only=definitions_only)


def load_namespace():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    namespace = load_support(source_snapshot())
    namespace.update(torch=torch, DataLoader=DataLoader,
                     datasets=datasets, transforms=transforms)
    cells = source_snapshot()
    hyperparameters = next(i for i, cell in enumerate(cells)
                           if cell["cell_type"] == "code"
                           and "SEED = 42" in "".join(cell["source"]))
    execute_notebook_cell(hyperparameters, namespace)
    for index in extract_cells():
        if index not in selected_cells():
            execute_cell(index, namespace)
    for index, (_, source) in selected_cells().items():
        if "TASK4_BASE_CONFIG =" in source or "WANDB_PROJECT =" in source:
            execute_cell(index, namespace)
        elif "def fetch_task4_results(" in source:
            execute_cell(index, namespace, definitions_only=True)
    return namespace


def cell_with(fragment):
    matches = [index for index, (_, source) in selected_cells().items()
               if fragment in source]
    if len(matches) != 1:
        raise ValueError(f"Expected one cell containing {fragment!r}: {matches}")
    return matches[0]
