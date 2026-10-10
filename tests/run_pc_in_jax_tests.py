"""Run the complete notebook suite with enforced line and branch coverage."""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from notebook_test_support import RESULTS, ROOT, NOTEBOOK_SOURCES, extract_notebook_cells

os.environ["PC_IN_JAX_RESULTS"] = str(RESULTS)


def main():
    extract_notebook_cells()  # Coverage needs its source directory to exist first.
    import pytest

    return pytest.main([
        "-p", "pytest_cov.plugin",
        str(ROOT / "tests"),
        "-ra",
        "--cov=" + str(NOTEBOOK_SOURCES),
        "--cov-branch",
        "--cov-config=" + str(ROOT / "tests" / "pc_in_jax.coveragerc"),
        "--cov-report=term-missing",
        "--cov-report=html:" + str(RESULTS / "htmlcov"),
        "--cov-report=json:" + str(RESULTS / "coverage.json"),
        "--cov-report=xml:" + str(RESULTS / "coverage.xml"),
        "--cov-fail-under=100",
        "--junitxml=" + str(RESULTS / "junit.xml"),
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
