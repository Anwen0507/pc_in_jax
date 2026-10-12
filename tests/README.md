These tests execute the code cells immediately after Task 2 and Task 3 in
`pc_in_jax.ipynb`. The runner locates each heading and extracts the following cell
verbatim before starting coverage. Required definitions also come directly from
the notebook. The notebook itself is never rewritten, and the installation,
dataset download, and complete training-loop cells are not executed.

The final verified run completed in 54.63 seconds: **115 passed, 1 failed, no
skips**. Both requested cells reached **100% line and branch coverage**: Task 2
covered all 7 executable statements and 4 branches; Task 3 covered all 13
executable statements and 2 branches. The full local MNIST test passed.

The remaining failure is
`test_task3_one_layer_model_clamps_output_to_labels`. With a one-layer model,
the `layer_idx == 0` branch in the notebook's `energy_fn` takes precedence over
the final-layer branch. It compares the initialized activity with its own
prediction and never uses the target label. The returned energy is therefore
0 instead of the independently calculated 0.3948259, and parameter learning
receives zero gradients. This defect is in a dependency outside the two requested
cells. The regression test remains an ordinary failing test, so the runner exits
with status 1. No notebook code was changed to conceal or repair the failure.

The tests check:

- Task 2: shapes, numerical values against NumPy and separate example calls,
  console output, one to five layers, singleton and MNIST-sized batches, NumPy
  and JAX inputs, four input dtypes, four activations, input activations, biases,
  permutation and duplication, repeated execution, empty batches, invalid
  dimensions and dtypes, nonfinite values, and the shape assertion's failure path.
- Task 3: optimizer setup, parameter filtering, all training call arguments,
  plotted values and step indices, labels, legend, grid and layout, empty and
  malformed auxiliary data, missing prerequisites, failure propagation, and
  invalid training data dimensions.
- Real training: 50-step traces and settled activities against independent NumPy
  predictions and analytic activity gradients; batch-sum trace versus batch-mean
  final energy; analytic parameter gradients and Adam moments/updates; zero
  learning rates; multiple depths, bias settings and activations; and unchanged
  original model, inputs, and targets.
- Existing local MNIST data: the notebook's seed, batch size 64, input size 784,
  hidden size 300, output size 10, three layers, and 50 settling steps. This test
  skips explicitly if the local IDX files are absent; it never downloads data.
- A one-layer regression check: output clamping must still use the target label
  when the first layer is also the final layer. A failure here identifies a
  defect in the notebook's `energy_fn` dependency.

To run in an environment containing the notebook's pinned dependencies:

```bash
python -m pip install -r tests/requirements-pc-in-jax.txt
python tests/run_pc_in_jax_tests.py
```

The current workspace also has an isolated temporary test environment:

```bash
/tmp/pc-in-jax-tests-venv/bin/python tests/run_pc_in_jax_tests.py
```

Reports are written to `test-results/pc_in_jax/`: JUnit XML, coverage JSON/XML,
HTML coverage, exact extracted source cells, and the MNIST energy plot. Coverage
is restricted to the two requested cells and requires 100% line and branch
coverage. Notebook dependencies are tested behaviorally but are outside that
coverage percentage. Coverage alone cannot prove correctness for every possible
input, which is an unbounded set.
