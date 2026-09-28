# Experimental-release cleanup

No dataset or checkpoint was deleted or moved. These directories remain ignored
in place: `training/`, `scene_datas/`, `test_data_gen/`, `test_overfit/`,
`test_overfit_2img/`, `pyramid_data/`, and `pyramid_data_old/`.

`local/archive/pre-release-20260928/` contains copies of all original root files
and `working-tree.patch`, preserving the pre-existing bounds and viewer model ID.

The following were moved, not erased, into `local/archive/experiments/`:

- `ffnerf/`: unfinished package refactor (previously ignored).
- `ffnerf_OG_checkpoint.cl`, `ffnerf_OG_checkpoint copy.cl`: older kernel variants.
- `compare_ffnerf.py`, `get_RT.py`, `simulated_images2.py`,
  `show_image_detail_map.py`: standalone experiments with old assumptions or paths.
- `cl_kernel_estimator.py`, `sandbox.py`, original `inspect_npz.py`: scratch tools.
- `test_data.txt`, `current_diff.txt`, `handle_previews_in_one_ans.txt`, `todo.txt`:
  development notes and data dumps.
- `detail_1714695134449.png`, `grad_1714695134449.png`: generated debug images.

This note describes the maintainer's private development checkout. Archived files
remain locally under the ignored archive, and earlier committed versions remain
in the private development history. The public release starts from an independent
snapshot and contains neither that history nor those local archives. To restore
an archived file in the development checkout, copy it to its original path.

Smoke outputs use a separate `release_smoke_20260928` model ID in
`training/test_dataset/`; verification scripts and logs are under `local/`.
Public configuration uses no personal absolute paths.
