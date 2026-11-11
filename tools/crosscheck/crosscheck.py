"""Python cross-check release gate, upstream side (development only).

Replays every run written by crosscheck.R through upstream faiss
``NumpyEmeans`` (contrib/e_means at the pinned commit, vendored in
``upstream_e_means/``): same data, same starting centroids, same batch
orders, and R's revival uniforms in place of upstream's private
``_split_rng``. Compares the last-pass labels, the centroids after every
epoch and the final centers with R's, prints a table, appends it to
``results.md`` and exits non-zero on any failure.

Run from the package root, after ``Rscript tools/crosscheck/crosscheck.R``:

    uv run --project tools/crosscheck python tools/crosscheck/crosscheck.py
"""

import datetime
import pathlib
import platform
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from upstream_e_means import NumpyEmeans  # noqa: E402

PINNED_COMMIT = "2b648f0605f2ef3d14c4705350f8a0b214463878"
TOLERANCE = 1e-4
RUN_ROOT = HERE / "runs"
RESULTS = HERE / "results.md"
ORDER = ["plain", "spherical", "revival", "fractional", "mppc", "manual", "lowdim"]


class GateError(Exception):
    """A broken precondition of the replay (not a numeric comparison)."""


def check(condition, message):
    # Explicit checks: `assert` is stripped under python -O.
    if not condition:
        raise GateError(message)


class Replay:
    """Stand-in for upstream's ``_split_rng``: returns R's revival uniforms."""

    def __init__(self, arrays):
        self._arrays = list(arrays)
        self._next = 0

    def random_sample(self, size):
        check(
            self._next < len(self._arrays),
            "upstream asked for revival draw %d, but R made only %d"
            % (self._next + 1, len(self._arrays)),
        )
        value = self._arrays[self._next]
        check(
            len(value) == size,
            "revival draw %d has %d values; upstream asked for %d"
            % (self._next + 1, len(value), size),
        )
        self._next += 1
        return value.copy()

    def assert_consumed(self):
        check(
            self._next == len(self._arrays),
            "upstream used %d of R's %d revival draws"
            % (self._next, len(self._arrays)),
        )


def read_params(path):
    params = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        params[key] = value
    return params


def optional(value, cast):
    return None if value == "None" else cast(value)


def flag(value):
    check(value in ("True", "False"), "not a flag: %r" % value)
    return value == "True"


def read_f64(path, shape=None):
    values = np.fromfile(path, dtype="<f8")
    return values if shape is None else values.reshape(shape)


def rel_diff(r, py):
    r = np.asarray(r, dtype="float64")
    py = np.asarray(py, dtype="float64")
    return float(np.max(np.abs(r - py)) / max(1.0, float(np.max(np.abs(r)))))


def replay(run_dir):
    p = read_params(run_dir / "params.txt")
    n_full, n = int(p["n_full"]), int(p["n"])
    d, k = int(p["d"]), int(p["k"])
    nepochs = float(p["nepochs"])
    spherical = flag(p["spherical"])

    x_full = read_f64(run_dir / "x.f64", (n_full, d))
    subset_path = run_dir / "subset.f64"
    if subset_path.exists():
        keep = read_f64(subset_path).astype("int64")
        check(len(keep) == n and np.all(np.diff(keep) > 0),
              "the subset must be %d sorted, distinct rows" % n)
        x = x_full[keep]
    else:
        check(n == n_full, "no subset file, but n != n_full")
        x = x_full
    # R rounds the data to float32 values, so upstream's float32 cast is exact.
    check(np.array_equal(x.astype("float32").astype("float64"), x),
          "the data are not float32-representable")
    init = read_f64(run_dir / "init.f64", (k, d))
    uniforms = read_f64(run_dir / "revival.f64")
    n_draws = int(p["n_revival_draws"])
    check(uniforms.size == n_draws * k, "revival.f64 has the wrong length")
    uniforms = uniforms.reshape(n_draws, k) if n_draws else uniforms.reshape(0, k)

    em = NumpyEmeans(
        d,
        k,
        nepochs=nepochs,
        batch_size=optional(p["batch_size"], lambda v: int(float(v))),
        alpha0=optional(p["alpha0"], float),
        batch_law=p["batch_law"],
        spherical=spherical,
        unused_threshold=optional(p["unused_threshold"], float),
        max_points_per_centroid=optional(
            p["max_points_per_centroid"], lambda v: int(float(v))
        ),
    )
    em.begin(n, init)
    rng = Replay(uniforms)
    em._split_rng = rng

    recipe = em.recipe
    r_batch, r_alpha0 = int(float(p["r_batch_size"])), float(p["r_alpha0"])
    check(recipe["batch_size"] == r_batch,
          "batch_size: upstream %d, R %d" % (recipe["batch_size"], r_batch))
    check(recipe["alpha0"] == r_alpha0,
          "alpha0: upstream %r, R %r" % (recipe["alpha0"], r_alpha0))
    r_rows = [int(float(v)) for v in p["r_epoch_rows"].split(",")]
    check(list(recipe["epoch_rows"]) == r_rows,
          "epoch_rows: upstream %s, R %s" % (recipe["epoch_rows"], r_rows))
    check(recipe["total_steps"] == int(float(p["r_total_steps"])),
          "total_steps differ")
    check(recipe["unused_threshold"] == float(p["r_unused_threshold"]),
          "unused_threshold differs")
    check(recipe["name"] == p["r_recipe_name"], "recipe name differs")

    r_steps = read_f64(run_dir / "r_centers_steps.f64",
                       (recipe["total_steps"], k, d))
    labels = np.full(n, -1, dtype="int64")
    step_diffs, epoch_diffs = [], []
    batch = recipe["batch_size"]
    step = 0
    for epoch, rows in enumerate(recipe["epoch_rows"]):
        perm = read_f64(run_dir / ("perm_%d.f64" % epoch)).astype("int64")
        check(np.array_equal(np.sort(perm), np.arange(n)),
              "perm_%d is not a permutation of the rows" % epoch)
        for begin in range(0, rows, batch):
            idx = perm[begin:min(begin + batch, rows)]
            stats = em.accumulate(x[idx])
            if "labels" in stats:
                labels[idx] = stats["labels"]
            em.update(stats)
            step_diffs.append(rel_diff(r_steps[step], em.current_centroids()))
            step += 1
        em.end_epoch()
        r_epoch = read_f64(run_dir / ("r_centers_epoch_%d.f64" % epoch), (k, d))
        epoch_diffs.append(rel_diff(r_epoch, em.current_centroids().copy()))
    final = em.finalize()
    rng.assert_consumed()

    r_final = read_f64(run_dir / "r_centers_final.f64", (k, d))
    r_labels = read_f64(run_dir / "r_labels.f64")
    check(not np.isnan(r_labels).any(), "R left last-pass labels missing")
    check((labels >= 0).all(), "upstream left last-pass labels missing")
    label_mismatch = int(np.sum(r_labels.astype("int64") - 1 != labels))
    r_nsplit = [int(float(v)) for v in p["r_nsplit_per_epoch"].split(",")]
    py_nsplit = [int(stat["nsplit"]) for stat in em.iteration_stats]
    final_diff = rel_diff(r_final, final)
    trainer = flag(p["r_trainer_identical"])
    # Every per-step snapshot is held to the same tolerance as the per-pass
    # ones: a schedule off by one step can stay under it at epoch ends.
    passed = (
        label_mismatch == 0
        and max(step_diffs + epoch_diffs + [final_diff]) <= TOLERANCE
        and py_nsplit == r_nsplit
        and em.n_splits == int(float(p["r_n_splits"]))
        and trainer
    )
    worst = int(np.argmax(step_diffs))
    first_bad = next(
        (i for i, v in enumerate(step_diffs) if v > TOLERANCE), None
    )
    return {
        "params": p,
        "recipe": recipe,
        "label_mismatch": label_mismatch,
        "epoch_diffs": epoch_diffs,
        "final_diff": final_diff,
        "max_step_diff": step_diffs[worst],
        "first_bad_step": first_bad,
        "r_nsplit": r_nsplit,
        "py_nsplit": py_nsplit,
        "n_splits": em.n_splits,
        "trainer": trainer,
        "passed": passed,
        "error": None,
    }


def describe(name, result):
    p = result["params"]
    if result["error"] is not None:
        return "| %s | %s | | | | | | FAIL: %s |" % (
            name, p.get("r_recipe_name", "?"), result["error"])
    recipe = result["recipe"]
    data = "n=%s d=%s k=%s, %s blobs %.1f sd apart" % (
        p["n"], p["d"], p["k"], p["n_blobs"], float(p["separation_sd"]))
    if p["n"] != p["n_full"]:
        data += ", subset of %s" % p["n_full"]
    if flag(p["spherical"]):
        data += ", spherical"
    rec = "%s, B=%d, alpha0=%.6g, epochs %s" % (
        recipe["name"], recipe["batch_size"], recipe["alpha0"],
        "/".join(str(r) for r in recipe["epoch_rows"]))
    splits = "%d (per pass %s; R %s)" % (
        result["n_splits"], "/".join(map(str, result["py_nsplit"])),
        "same" if result["py_nsplit"] == result["r_nsplit"]
        else "/".join(map(str, result["r_nsplit"])))
    labels = ("identical (%s rows)" % p["n"] if result["label_mismatch"] == 0
              else "%d differ" % result["label_mismatch"])
    passes = " / ".join("%.1e" % v for v in result["epoch_diffs"])
    verdict = "PASS" if result["passed"] else "FAIL"
    if result["first_bad_step"] is not None:
        verdict += " (first step over tolerance: %d)" % (
            result["first_bad_step"] + 1)
    return "| %s | %s | %s | %s | %s | %s | %.1e | %.1e | %s | %s |" % (
        name, data, rec, splits, labels, passes, result["final_diff"],
        result["max_step_diff"], "identical" if result["trainer"] else "DIFFERS",
        verdict)


def main():
    runs = [RUN_ROOT / name for name in ORDER if (RUN_ROOT / name).is_dir()]
    if len(runs) != len(ORDER):
        missing = sorted(set(ORDER) - {r.name for r in runs})
        print("Missing runs %s: run Rscript tools/crosscheck/crosscheck.R first"
              % missing, file=sys.stderr)
        return 2
    results = {}
    for run_dir in runs:
        try:
            results[run_dir.name] = replay(run_dir)
        except GateError as error:
            results[run_dir.name] = {
                "params": read_params(run_dir / "params.txt"),
                "passed": False,
                "error": str(error),
            }

    first = results[ORDER[0]]["params"]
    lines = [
        "## Run %s" % datetime.datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M %Z"),
        "",
        "Environment: R %s ({emeans} %s, RNG %s, kernel %s, threads %s); "
        "Python %s, NumPy %s, SciPy %s, faiss-cpu %s; upstream NumpyEmeans "
        "vendored at %s; %s %s." % (
            first["r_version"], first["r_emeans_version"], first["r_rng_kind"],
            first["r_kernel"], first["r_threads"], platform.python_version(),
            np.__version__, scipy.__version__, faiss.__version__,
            PINNED_COMMIT[:12], platform.system(), platform.machine()),
        "",
        "Pass criteria: last-pass labels identical; max|R - Py| / "
        "max(1, max|R|) <= %g for the centroids after every pass, after "
        "every batch update (worst step) and for the final centers; same "
        "splits per pass; every revival draw consumed; emeans() identical "
        "to the em_*() loop." % TOLERANCE,
        "",
        "| config | data | recipe | n_splits | last-pass labels | "
        "max rel diff per pass | final | worst step | emeans() vs loop | result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [describe(name, results[name]) for name in ORDER]
    failed = [name for name in ORDER if not results[name]["passed"]]
    lines += [
        "",
        "**Gate: %s**" % ("PASS (%d of %d configurations)" % (
            len(ORDER), len(ORDER)) if not failed
            else "FAIL (%s)" % ", ".join(failed)),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    with RESULTS.open("a") as handle:
        handle.write("\n" + text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
