# Local verification — 2026-09-28

Environment: Windows, Python 3.10.4, AMD Ellesmere GPU / AMD OpenCL 2.1 runtime.
Initial smoke checks used PyOpenCL 2026.1.1; release cache checks below use 2026.1.2.
Other direct dependency versions are recorded in requirements.txt. Verification used the sibling custom SfM
`datasets/test_dataset/data/solve/reconstructionBAcl.npz` and its original images.

Completed checks:

- Five CPU tests: BA w2c conversion and relocated paths; Stage 2 posed/keep filtering;
  intrinsics and bounds; config path resolution and unknown-key rejection; matching
  training/review rays with calibrated intrinsics.
- Full data inspection: 56 registered cameras, 10,883 finite sparse points, all
  image paths readable and dimensions consistent (2250×4000).
- GPU training: 4 steps at 450×800 training resolution, final checkpoint and fixed
  multiview image saved under a separate smoke model ID.
- Resume: 4 additional steps, checkpoint counter advanced from 4 to 8, finite MLP
  weights/optimizer state checked.
- Snapshot: loaded that checkpoint and rendered a finite 120×192 image from the
  exact configured dataset pose without a GUI.
- Browser integration: local HTTP page, WebSocket camera initialization and three
  JPEG frames, Save frame, and Stop training. A separate short GPU run produced
  loss CSV rows, a saved live PNG, interval checkpoints, and a final checkpoint.
- Python compilation, CLI help, and Git whitespace checks.

These initial checks established execution and data-contract behavior. Since
then, the maintainer has tested the custom SfM pipeline, training and resume,
smoke tests, and the preview script. The included examples show the test scene
through 40,000 training steps. The low-quality input dataset limits the result;
these tests do not establish general convergence or support for other GPUs.
The unusually short save/preview intervals used for the browser check are local
test settings, not the shipped defaults.

Local logs and the browser integration driver are preserved in `local/`; model
outputs are in ignored `training/test_dataset/`. They are not public release data.

## Final release checks

- All 10 scene-contract and training-regression tests passed; Python compilation
  and dependency consistency checks passed.
- LICENSE matches the companion SfM MIT license byte for byte.
- All local README image and document links resolve.
- Removed forced `PYOPENCL_NO_CACHE` assignments and pinned PyOpenCL 2026.1.2,
  whose compiler cache fixes the bytes/string formatting failure in 2026.1.1.
- Built and rendered a copied checkpoint in two separate processes: the first
  took 48.80 seconds; the second took 3.64 seconds and logged four binary cache
  hits. The resulting snapshot pixels matched exactly. These are startup/render
  measurements for this machine, not general training benchmarks.

Cache verification used an ignored `local/release-cache/` directory and a copy
of the trained checkpoint; it did not modify the trained model.
