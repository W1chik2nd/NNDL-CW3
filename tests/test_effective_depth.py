"""Offline tests for the actual NB2 §8 cells; no notebook setup or training."""
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C


def notebook_cells():
    cells = json.loads((ROOT / "nb2_experiment.ipynb").read_text())["cells"]
    return {cell["id"]: "".join(cell["source"]) for cell in cells}


CELLS = notebook_cells()
NS = {"C": C, "QUICK": True}
exec(compile(CELLS["c-finals"], "NB2 §5", "exec"), NS)
exec(compile(CELLS["nb2-24"], "NB2 §8", "exec"), NS)
FINAL_TAG = NS["FINAL_TAG"]
LRS = {(depth, a): 0.003 for depth, alphas in C.ALPHAS.items() for a in alphas}


def predict(records, selected_lrs=LRS):
    return NS["effective_depth_test"](records, selected_lrs=selected_lrs, final_tag=FINAL_TAG)


def fixture(slope=0.0, offset=0.0):
    records = []
    for (depth, a), lr in LRS.items():
        base = 0.4 + slope * math.log(C.effective_depth(depth, a) / 16)
        if depth == 16 and a == 0:
            base += offset
        for seed in C.final_seeds(depth, a):
            records.append(dict(depth=depth, alpha=a, lr=lr, seed=seed, width=C.WIDTH,
                                epochs=C.EPOCHS, diverged=False, _meta={"tag": FINAL_TAG},
                                final_train_loss=base + 0.001 * (seed - 2)))
    return records


class SectionTest(unittest.TestCase):
    def test_flat_curve_and_coverage(self):
        report = predict(fixture())
        rows = report["rows"]
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["alpha"] for r in rows if r["method"] == "matched"},
                         {target[1] for _, target in C.MATCHED_PAIRS})
        self.assertEqual({r["alpha"] for r in rows if r["method"] == "interpolated"},
                         {C.ALPHA_UDL, 0.9})
        self.assertNotIn(0.97, [r["alpha"] for r in rows])
        # The full 32-block means include two extra synthetic seed offsets for matched conditions.
        self.assertAlmostEqual(report["epsilon"], 0.0002)
        for r in rows:
            seeds = C.MATCHED_PAIR_SEEDS if r["method"] == "matched" else C.FINAL_SEEDS
            self.assertEqual(r["seeds"], seeds)
            self.assertEqual(r["ci95"], (0, 0))

    def test_log_interpolation_and_error_sign(self):
        for row in predict(fixture(0.06, 0.03)[::-1])["rows"]:
            seed_shift = 0.001 * (sum(row["seeds"]) / len(row["seeds"]) - 2)
            expected = 0.4 + 0.06 * math.log(row["effective_depth"] / 16) + seed_shift
            self.assertAlmostEqual(row["predicted"], expected)
            self.assertAlmostEqual(row["error"], 0.03 if row["alpha"] == 0 else 0)

    def test_interpolation_uses_only_matching_seeds(self):
        records = fixture(0.06)
        # These belong to the 32-block curve, but not to the three interpolated pairs.
        for r in records:
            if r["depth"] == 32 and r["seed"] > 3:
                r["final_train_loss"] += 0.15
        report = predict(records)
        for row in report["rows"]:
            expected = -0.06 if row["method"] == "matched" else 0
            self.assertAlmostEqual(row["error"], expected)
        means = [sum(r["final_train_loss"] for r in records if (r["depth"], r["alpha"]) == (32, a))
                 / len(C.final_seeds(32, a)) for a in C.ALPHAS_32]
        self.assertAlmostEqual(report["epsilon"], 0.20 * (max(means) - min(means)))

    def test_paired_ci_uses_seed_error_variance_and_correct_df(self):
        records = fixture()
        for r in records:
            if r["depth"] == 16 and r["alpha"] == 0:
                r["final_train_loss"] += 0.01 + 0.005 * (r["seed"] - 3)
        [row] = [r for r in predict(records)["rows"] if r["alpha"] == 0]
        half = 2.77644511 * 0.005 / math.sqrt(2)
        self.assertAlmostEqual(row["error"], 0.01)
        self.assertAlmostEqual(row["ci95"][0], 0.01 - half)
        self.assertAlmostEqual(row["ci95"][1], 0.01 + half)
        lo, hi = NS["paired_t_ci95"]([-0.01, 0, 0.01])
        self.assertAlmostEqual(hi, 4.30265273 * 0.01 / math.sqrt(3))
        self.assertAlmostEqual(lo, -hi)

    def test_verdict_all_three_paths_and_strict_margin(self):
        verdict = NS["effective_depth_verdict"]
        small = dict(error=0.005, ci95=(0.0, 0.01))
        large = dict(error=0.01, ci95=(-0.02, 0.03))
        positive = dict(error=0.016, ci95=(0.012, 0.02))
        negative = dict(error=-0.016, ci95=(-0.02, -0.012))
        for rows, expected in (([small, small], "enough"), ([small, large], "not sure"),
                               ([positive, negative], "not sure"),
                               ([positive, positive], "not enough"),
                               ([negative, negative], "not enough")):
            with self.subTest(expected=expected, rows=rows):
                self.assertEqual(verdict(rows, 0.01)[0], expected)

    def test_unrelated_runs_do_not_enter(self):
        records = fixture()
        noise = [dict(records[0], final_train_loss=99, **changes) for changes in
                 (dict(seed=0), dict(lr=0.03), dict(width=362), dict(epochs=12),
                  dict(_meta={"tag": "quick"}))]
        self.assertEqual(predict(records + noise), predict(records))

    def test_bad_finals_stop_analysis(self):
        records = fixture()
        for invalid, message in (
            (records[1:], "Missing finals"),
            (records + [records[0]], "Duplicate final"),
            ([dict(records[0], diverged=True)] + records[1:], "Diverged or nonfinite final"),
            ([dict(records[0], final_train_loss=float("nan"))] + records[1:],
             "Diverged or nonfinite final"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                predict(invalid)
        with self.assertRaisesRegex(ValueError, "Missing selected learning rates"):
            predict(records, {})

    def test_cached_notebook_report(self):
        report, text = cached_report()
        self.assertEqual(report["n_finals"], sum(len(C.final_seeds(*key)) for key in LRS))
        self.assertEqual(len(report["rows"]), 6)
        self.assertIn("95% paired t CI", text)
        self.assertIn(f"Effective depth: {report['verdict']}", text)


def cached_report():
    def load_results(*, tag, **filters):
        records = [json.loads(p.read_text()) for p in sorted((ROOT / C.RESULTS_DIR).glob("*.json"))]
        return [r for r in records if r.get("_meta", {}).get("tag") == tag
                and all(r[k] == value for k, value in filters.items())]

    def load_selected_lr():
        data = json.loads((ROOT / "selected_lr.json").read_text())
        return {(r["depth"], r["alpha"]): r["lr"] for r in data["selected_lr"]}

    ns = dict(NS, load_results=load_results, load_selected_lr=load_selected_lr)
    output = StringIO()
    with redirect_stdout(output):
        exec(compile(CELLS["effective-depth-results"], "NB2 §8 results", "exec"), ns)
    return ns["prediction"], output.getvalue()


if __name__ == "__main__":
    if sys.argv[1:] == ["--results"]:
        print(cached_report()[1], end="")
    elif sys.argv[1:] == ["--demo"]:
        print("SYNTHETIC ONLY")
        for row in predict(fixture(0.06, 0.03))["rows"]:
            print(f"α={row['alpha']:.5f}  {row['method']:<12}  error={row['error']:+.5f}")
    else:
        unittest.main()
