# Python cross-check release gate

Development-only check that the R port reproduces upstream faiss
`NumpyEmeans` (`contrib/e_means` at `2b648f0605f2ef3d14c4705350f8a0b214463878`)
pass by pass. Both sides get the same data, the same starting centroids, the
same batch orders and R's revival uniforms.

Run it from the package root:

```sh
Rscript tools/crosscheck/crosscheck.R && \
  uv run --project tools/crosscheck python tools/crosscheck/crosscheck.py
```

## How it works

- `crosscheck.R` loads the package with `devtools::load_all()` and switches to
  the direct kernel with one thread. For each configuration it generates the
  data, then turns on the `emeans.rng_recorder` hook. It draws the subset
  (mppc only), the starting rows and one permutation per epoch, in
  `emeans()`'s order, and runs the `em_*()` loop; revival uniforms are drawn
  inside `em_update()`. It writes the inputs and R's centroids after every
  batch update and every epoch, plus the final centers and last-pass labels,
  to `runs/<name>/`. That directory is git-ignored.
- Then it replays `emeans()` itself from the same RNG state. The trainer must
  make the same draws and return the same centers, labels and `n_splits` as
  the loop. That ties the gate to the user-facing function, including the
  `max_points_per_centroid` subset and its `NA` labels.
- `crosscheck.py` builds `NumpyEmeans` with the same options and calls
  `begin(n, init)`, then sets `em._split_rng` to a replay object. The replay
  object returns R's uniforms in order and checks each length against `k`;
  every draw must be consumed by the end. The script feeds the same 0-based
  batches, compares the recipe, and compares the centroids after every update
  and every `end_epoch()`, the `finalize()` centers and the last-pass labels.
- Upstream's Python is vendored unmodified in `upstream_e_means/`, because the
  `faiss-cpu` 1.16.0 wheel ships a different revision of `e_means.py`.
  `faiss-cpu` supplies `faiss.knn`. SciPy is installed, so the pinned code
  sums batches through its sparse `_add_rows_at` path.

## Data

- Euclidean blobs sit at ±15 sd on the coordinate axes, so the closest pair is
  21.2 sd apart.
- Spherical blobs are unit axis directions with noise sd 0.05, projected to
  unit length; the closest pair is 28.3 sd apart.
- The data are rounded to float32 values in R, so upstream's float32 cast is
  exact and both sides cluster identical numbers.
- The starting centroids are random rows of the data, drawn by
  `draw_init_rows()`.

| config | what it covers |
|---|---|
| plain | d = 8, k = 16, 16 blobs, 3 epochs, auto (quality) batch law |
| spherical | as plain, `spherical = TRUE` |
| revival | k = 64 on 8 blobs, n = 8000: revival splits in the first pass, B = 64 |
| fractional | `nepochs = 2.5`: epoch rows n / n/2 / n |
| mppc | n = 6000, `max_points_per_centroid = 150`: R draws the 2400-row subset, both sides train on it |
| manual | `batch_size = 64`, `alpha0 = 0.3` (recipe `"manual"`) |
| lowdim | d = 2, k = 4: `unused_threshold` resolves to 0, so there are no revival draws |

## Pass criteria

- Last-pass labels identical.
- `max|R - Py| / max(1, max|R|) <= 1e-4` for every per-pass snapshot and for
  the final centers.
- The gate also holds every per-batch snapshot to the same bound ("worst
  step"). That is stricter than checking epoch ends alone: an `alpha` schedule shifted by
  one step stayed under 1e-4 at the epoch ends on `lowdim`, but not after
  individual updates.
- The recipe (batch size, `alpha0`, epoch rows, total steps, threshold, name)
  matches exactly.
- The splits per pass match, and every revival draw is consumed.
- `emeans()` is identical to the `em_*()` loop.

## Negative controls (run once, not repeated by the scripts)

The gate failed as expected after each of these deliberate breakages, each
reverted afterwards:

- revival targets taken from the last unused clusters instead of the first:
  plain, revival and fractional failed;
- the `alpha` schedule evaluated one step ahead: six configurations failed
  (lowdim only on per-step snapshots);
- spherical closing means left unnormalised: spherical failed;
- upstream fed `1 - u` instead of R's uniforms: a replay mismatch or failure
  in all five configurations that revive.

## Run 2026-11-11 11:15 CST

Environment: R 4.6.2 ({emeans} 0.0.0.9000, RNG Mersenne-Twister/Inversion/Rejection, kernel direct, threads 1); Python 3.14.10, NumPy 2.5.4, SciPy 1.18.1, faiss-cpu 1.16.0; upstream NumpyEmeans vendored at 2b648f0605f2; Darwin arm64.

Pass criteria: last-pass labels identical; max|R - Py| / max(1, max|R|) <= 0.0001 for the centroids after every pass, after every batch update (worst step) and for the final centers; same splits per pass; every revival draw consumed; emeans() identical to the em_*() loop.

| config | data | recipe | n_splits | last-pass labels | max rel diff per pass | final | worst step | emeans() vs loop | result |
|---|---|---|---|---|---|---|---|---|---|
| plain | n=4000 d=8 k=16, 16 blobs 21.2 sd apart | canonical_v1, B=16, alpha0=0.0593782, epochs 4000/4000/4000 | 4 (per pass 4/0/0; R same) | identical (4000 rows) | 1.0e-07 / 1.5e-07 / 2.0e-07 | 4.2e-07 | 2.3e-07 | identical | PASS |
| spherical | n=4000 d=8 k=16, 16 blobs 28.3 sd apart, spherical | canonical_v1, B=16, alpha0=0.0593782, epochs 4000/4000/4000 | 3 (per pass 3/0/0; R same) | identical (4000 rows) | 8.0e-08 / 8.5e-08 / 1.2e-07 | 1.7e-07 | 3.1e-07 | identical | PASS |
| revival | n=8000 d=8 k=64, 8 blobs 21.2 sd apart | canonical_v1, B=64, alpha0=0.0798522, epochs 8000/8000/8000 | 12 (per pass 12/0/0; R same) | identical (8000 rows) | 1.1e-07 / 1.9e-07 / 3.0e-07 | 2.5e-07 | 3.1e-07 | identical | PASS |
| fractional | n=4000 d=8 k=16, 16 blobs 21.2 sd apart | canonical_v1, B=16, alpha0=0.0605336, epochs 4000/2000/4000 | 3 (per pass 3/0/0; R same) | identical (4000 rows) | 8.9e-08 / 1.4e-07 / 3.6e-07 | 3.0e-07 | 3.6e-07 | identical | PASS |
| mppc | n=2400 d=8 k=16, 16 blobs 21.2 sd apart, subset of 6000 | canonical_v1, B=16, alpha0=0.0747161, epochs 2400/2400/2400 | 1 (per pass 1/0/0; R same) | identical (2400 rows) | 1.2e-07 / 1.2e-07 / 2.5e-07 | 1.3e-07 | 2.8e-07 | identical | PASS |
| manual | n=4000 d=8 k=16, 16 blobs 21.2 sd apart | manual, B=64, alpha0=0.3, epochs 4000/4000/4000 | 0 (per pass 0/0/0; R same) | identical (4000 rows) | 6.1e-08 / 7.6e-08 / 1.6e-07 | 2.7e-07 | 1.6e-07 | identical | PASS |
| lowdim | n=2000 d=2 k=4, 4 blobs 21.2 sd apart | canonical_v1, B=8, alpha0=0.0824107, epochs 2000/2000/2000 | 0 (per pass 0/0/0; R same) | identical (2000 rows) | 6.6e-08 / 7.3e-08 / 1.0e-07 | 1.2e-07 | 2.0e-07 | identical | PASS |

**Gate: PASS (7 of 7 configurations)**
