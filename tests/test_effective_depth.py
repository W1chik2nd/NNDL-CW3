"""Offline tests for the actual NB2 §8 cell. No notebook setup, downloads or writes."""
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C


def load_section():
    cells = json.loads((ROOT / "nb2_experiment.ipynb").read_text())["cells"]
    [cell] = [c for c in cells if "def effective_depth_test(" in "".join(c["source"])]
    ns = {"C": C}
    exec(compile("".join(cell["source"]), "NB2 §8", "exec"), ns)
    return ns["effective_depth_test"]


predict = load_section()
lrs = {(L, a): 0.003 for L, alphas in C.ALPHAS.items() for a in alphas}


def fixture(slope=0.0, offset=0.0):
    records = []
    for (L, a), lr in lrs.items():
        seeds = C.final_seeds(L, a)
        mean = 0.4 + slope * math.log(C.effective_depth(L, a) / 16)
        if L == 16 and a == 0:
            mean += offset
        for seed in seeds:
            records.append(dict(depth=L, alpha=a, lr=lr, seed=seed, width=C.WIDTH,
                                epochs=C.EPOCHS, diverged=False, _meta={"tag": ""},
                                final_train_loss=mean + .001 * (seed - sum(seeds)/len(seeds))))
    return records


class SectionTest(unittest.TestCase):
    def test_flat_curve_and_coverage(self):
        rows = predict(fixture(), selected_lrs=lrs)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["alpha"] for r in rows if r["method"] == "matched"},
                         {q[1] for _, q in C.MATCHED_PAIRS})
        self.assertEqual({r["alpha"] for r in rows if r["method"] == "interpolated"},
                         {C.ALPHA_UDL, .9})
        self.assertNotIn(.97, [r["alpha"] for r in rows])
        for r in rows:
            self.assertAlmostEqual(r["error"], 0)

    def test_log_interpolation_and_error_sign(self):
        rows = predict(fixture(.06, .03)[::-1], selected_lrs=lrs)
        for r in rows:
            self.assertAlmostEqual(r["predicted"], .4 + .06 * math.log(r["effective_depth"]/16))
            self.assertAlmostEqual(r["error"], .03 if r["alpha"] == 0 else 0)

    def test_unrelated_runs_do_not_enter(self):
        records = fixture()
        noise = [dict(records[0], final_train_loss=99, **changes) for changes in
                 (dict(seed=0), dict(lr=.03), dict(width=362), dict(epochs=12),
                  dict(_meta={"tag": "quick"}))]
        self.assertEqual(predict(records+noise, selected_lrs=lrs), predict(records, selected_lrs=lrs))

    def test_bad_finals_stop_analysis(self):
        records = fixture()
        for invalid in (records[1:], records + [records[0]],
                        [dict(records[0], diverged=True)] + records[1:],
                        [dict(records[0], final_train_loss=float("nan"))] + records[1:]):
            with self.subTest(), self.assertRaises(ValueError):
                predict(invalid, selected_lrs=lrs)


if __name__ == "__main__":
    if sys.argv[1:] == ["--demo"]:
        rows = predict(fixture(.06, .03), selected_lrs=lrs)
        print("SYNTHETIC ONLY | depth16 alpha  eff.depth  method        observed predicted error")
        for r in rows:
            print(f"{r['alpha']:28.5f}  {r['effective_depth']:9.4f}  {r['method']:12}"
                  f"  {r['observed']:.5f}  {r['predicted']:.5f}  {r['error']:+.5f}")
        print("Excluded: depth=16, alpha=0.97 (outside the 32-block range)")
    else:
        unittest.main()
