"""Offline tests for the NB1 §7 range-check cells. No notebook setup, downloads or training.

Run with `python tests/test_range_check.py`, or `--demo` to print a synthetic table.
The functions are loaded out of the notebook itself, so the tests cover the cell that ships.
"""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C


def load_section():
    cells = json.loads((ROOT / "nb1_baseline.ipynb").read_text())["cells"]
    [cell] = [c for c in cells if "def range_check(" in "".join(c["source"])]
    ns = {"C": C}
    exec(compile("".join(cell["source"]), "NB1 §7 range check", "exec"), ns)
    return ns


NS = load_section()
range_check = NS["range_check"]
_t_crit = NS["_t_crit"]
RANGE_EPOCHS = NS["RANGE_EPOCHS"]
RANGE_SEEDS = NS["RANGE_SEEDS"]
RANGE_LRS = NS["RANGE_LRS"]
RANGE_ALPHAS = NS["RANGE_ALPHAS"]
TAG = NS["RANGE_TAG"]
ANCHORS = (C.EARLY_EPOCH, C.EPOCHS, RANGE_EPOCHS)


def make_curve(v_early, v_mid, v_final):
    """A 12-epoch curve through the three anchor epochs, linear in between."""
    xs = list(ANCHORS)
    ys = [v_early, v_mid, v_final]
    curve = []
    for epoch in range(1, RANGE_EPOCHS + 1):
        if epoch in xs:
            curve.append(ys[xs.index(epoch)])
            continue
        for k in range(len(xs) - 1):
            if xs[k] < epoch < xs[k + 1]:
                t = (epoch - xs[k]) / (xs[k + 1] - xs[k])
                curve.append(ys[k] + t * (ys[k + 1] - ys[k]))
                break
    return curve


def record(alpha, lr, seed, curve, *, tag=TAG, width=None, epochs=None, **over):
    rec = dict(depth=C.DEPTH_MAIN, width=C.WIDTH, alpha=alpha, lr=lr, seed=seed,
               epochs=RANGE_EPOCHS, train_loss=list(curve),
               final_train_loss=curve[-1], diverged=False, seconds=1.0,
               _meta={"tag": tag})
    rec.update(over)
    if width is not None:
        rec["width"] = width
    if epochs is not None:
        rec["epochs"] = epochs
    return rec


def build(levels, spread=0.002):
    """levels: {(alpha, lr): (early, mid, final)}. Seed i is offset symmetrically about 0."""
    records = []
    for (alpha, lr), anchors in levels.items():
        base = make_curve(*anchors)
        middle = (len(RANGE_SEEDS) - 1) / 2
        for i, seed in enumerate(RANGE_SEEDS):
            offset = (i - middle) * spread
            records.append(record(alpha, lr, seed, [v + offset for v in base]))
    return records


def flat():
    """Both alphas identical at the one learning rate."""
    return build({(0.5, 3e-3): (1.0, 0.5, 0.3), (0.9, 3e-3): (1.0, 0.5, 0.3)})


class RangeCheckTest(unittest.TestCase):
    def test_flat_is_not_distinguishable(self):
        result = range_check(flat())
        self.assertEqual(result["verdict"], "not distinguishable at any reported epoch")
        self.assertEqual(result["separating_epochs"], [])
        self.assertEqual([row["epoch"] for row in result["rows"]], list(ANCHORS))
        for row in result["rows"]:
            self.assertAlmostEqual(row["gap"], 0.0)
            self.assertAlmostEqual(row["lo"], -row["hi"])  # symmetric about zero
            self.assertFalse(row["excludes_zero"])

    def test_clear_gap_separates_at_every_epoch(self):
        records = build({(0.5, 3e-3): (1.0, 0.5, 0.3), (0.9, 3e-3): (0.8, 0.3, 0.2)})
        result = range_check(records)
        self.assertEqual(result["separating_epochs"], list(ANCHORS))
        self.assertTrue(result["verdict"].startswith("distinguishable at epoch"))
        for row in result["rows"]:
            self.assertLess(row["gap"], 0)                # alpha = 0.9 has the lower loss
            self.assertFalse(row["lo"] <= 0 <= row["hi"])

    def test_separates_early_only(self):
        records = build({(0.5, 3e-3): (1.0, 0.40, 0.30), (0.9, 3e-3): (0.6, 0.40, 0.30)})
        result = range_check(records)
        self.assertEqual(result["separating_epochs"], [C.EARLY_EPOCH])
        by_epoch = {row["epoch"]: row for row in result["rows"]}
        self.assertTrue(by_epoch[C.EARLY_EPOCH]["excludes_zero"])
        self.assertFalse(by_epoch[RANGE_EPOCHS]["excludes_zero"])

    def test_picks_the_best_learning_rate_for_each_alpha(self):
        records = build({
            (0.5, 1e-3): (1.2, 0.8, 0.6), (0.5, 3e-3): (1.0, 0.5, 0.3), (0.5, 1e-2): (1.5, 0.9, 0.7),
            (0.9, 1e-3): (1.2, 0.8, 0.6), (0.9, 3e-3): (1.1, 0.6, 0.4), (0.9, 1e-2): (0.9, 0.4, 0.25),
        })
        result = range_check(records)
        self.assertAlmostEqual(result["chosen"][0.5]["lr"], 3e-3)
        self.assertAlmostEqual(result["chosen"][0.9]["lr"], 1e-2)
        self.assertAlmostEqual(result["chosen"][0.5]["mean_final"], 0.3)
        self.assertAlmostEqual(result["chosen"][0.9]["mean_final"], 0.25)
        final = {row["epoch"]: row for row in result["rows"]}[RANGE_EPOCHS]
        self.assertAlmostEqual(final["gap"], 0.25 - 0.3)

    def test_diverged_run_is_listed_and_not_averaged(self):
        records = flat()
        target = [r for r in records if r["alpha"] == 0.5 and r["seed"] == 1][0]
        target["diverged"] = True
        target["train_loss"] = target["train_loss"][:3]
        result = range_check(records)
        self.assertEqual(len(result["unfinished"]), 1)
        self.assertEqual(result["unfinished"][0]["seed"], 1)
        self.assertEqual(result["chosen"][0.5]["n"], 2)
        self.assertEqual(result["verdict"], "not distinguishable at any reported epoch")

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            range_check([])
        with self.assertRaises(ValueError):
            range_check(flat() + [dict(flat()[0])])              # duplicate run
        one_seed = [r for r in flat() if r["seed"] == 0]
        with self.assertRaises(ValueError):                      # no spread to put an interval on
            range_check(one_seed)
        with self.assertRaises(ValueError):                      # three alphas
            range_check(flat(), alphas=(0.5, 0.9, 0.97))

    def test_unrelated_runs_do_not_enter(self):
        clean = range_check(flat())
        noise = [record(0.5, 3e-3, 0, make_curve(9.0, 9.0, 9.0), tag="other"),
                 record(0.5, 3e-3, 0, make_curve(9.0, 9.0, 9.0), epochs=6),
                 record(0.5, 3e-3, 0, make_curve(9.0, 9.0, 9.0), width=362),
                 record(0.5, 5e-3, 0, make_curve(9.0, 9.0, 9.0)),
                 record(0.5, 3e-3, 7, make_curve(9.0, 9.0, 9.0)),
                 record(0.97, 3e-3, 0, make_curve(9.0, 9.0, 9.0)),
                 dict(record(0.5, 3e-3, 0, make_curve(9.0, 9.0, 9.0)), depth=16)]
        self.assertEqual(range_check(flat() + noise), clean)


class TCritTest(unittest.TestCase):
    def test_table_and_tail(self):
        self.assertAlmostEqual(_t_crit(0.95, 1), 12.706, places=3)
        self.assertAlmostEqual(_t_crit(0.95, 4), 2.776, places=3)
        self.assertAlmostEqual(_t_crit(0.95, 30), 2.042, places=3)
        self.assertAlmostEqual(_t_crit(0.95, 200), 1.9718, places=3)
        values = [_t_crit(0.95, df) for df in (2, 5, 10, 20, 30, 60, 200)]
        self.assertEqual(values, sorted(values, reverse=True))   # wider for fewer df
        self.assertEqual(_t_crit(0.95, 0), math.inf)


if __name__ == "__main__":
    if sys.argv[1:] == ["--demo"]:
        demo = build({(0.5, 3e-3): (1.0, 0.50, 0.30), (0.9, 3e-3): (0.6, 0.45, 0.32)})
        result = range_check(demo)
        lo, hi = result["alphas"]
        print("SYNTHETIC ONLY | epoch  alpha=0.50        alpha=0.90        difference"
              "   95% interval          distinguishable")
        for row in result["rows"]:
            print(f"{row['epoch']:>18}  {row['mean'][lo]:.4f} ± {row['sd'][lo]:.3f}  "
                  f"{row['mean'][hi]:.4f} ± {row['sd'][hi]:.3f}  {row['gap']:+.4f}  "
                  f"{row['lo']:+.4f} to {row['hi']:+.4f}  {row['excludes_zero']}")
        print("Verdict:", result["verdict"])
    else:
        unittest.main()
