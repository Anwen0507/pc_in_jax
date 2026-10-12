"""Run Task 4/5 tests, enforcing complete line and branch coverage."""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from task45_test_support import (
    RESULTS, ROOT, SUPPORT_SOURCES, TASK_SOURCES, extract_cells,
)

os.environ["PC_IN_JAX_TASK45_RESULTS"] = str(RESULTS)


def main():
    extract_cells()
    import pytest

    return pytest.main([
        "-p", "pytest_cov.plugin",
        *map(str, sorted((ROOT / "tests").glob("test_pc_in_jax_task45*.py"))),
        str(ROOT / "tests" / "test_pc_in_jax_activity_optimizers.py"),
        "-ra",
        "--cov=" + str(TASK_SOURCES),
        "--cov=" + str(SUPPORT_SOURCES),
        "--cov-branch",
        "--cov-config=" + str(ROOT / "tests" / "pc_in_jax_task45.coveragerc"),
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
