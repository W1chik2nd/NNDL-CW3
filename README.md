# COMP3620 CW3 — Does path structure still matter once variance is fixed?

**Group:** _<group name>_ · COMP3620 Neural Networks and Deep Learning, Coursework 3

| Member | Role | Owns |
| --- | --- | --- |
| _name_ | A – theory & variance control | `init_weights`, `calibrate`; NB1 §2–4, §7; NB2 §0–1, §3, §11 (with D); `hypotheses.md`; challenge statement |
| _name_ | B – pipeline & engineering | `ResMLP`, `train_one_run`, `sweep`; NB1 §1, §5–6; NB2 §2, §10; notebook merging; QUICK mode; packaging |
| _name_ | C – 32-block experiment & results | range check, `select_lr`; 32-block sweeps and finals; NB2 §4–6, §9 (sensitivity) |
| _name_ | D – 16-block, mechanism & effective-depth test | 16-block sweeps and finals; `jacobian_metrics`, `effective_depth_test`; NB2 §7–8, §11 (with A) |
| _name_ | E – teaching, logs & video | NB3; meeting and AI logs; project log; video script and editing; external reader test |

## Question

*Understanding Deep Learning* (Ch. 11) gives two reasons why residual networks train
better than plain ones: they keep the activation variance under control, and they
behave like an ensemble of shorter paths. With the forward and backward variance held
fixed by the initialisation, how much does the skip weight α change how well a deep
residual MLP trains under a fixed budget, and does it act only through the effective
depth L(1−α²)?

## Design at a glance

| Item | Choice (values live in `config.py`) |
| --- | --- |
| Model | Pre-activation residual MLP, block `h ← α·h + W·ReLU(h)`, width 256, no bias, no normalisation |
| Manipulated variable | α only. 32 blocks: {0, 0.5, 1/√2, 0.791, 0.9, 0.97}. The same manipulation is repeated at 16 blocks with {0, 0.5, 1/√2, 0.787, 0.9, 0.939, 0.97}; 0.791, 0.787 and 0.939 give exact effective-depth matches |
| Matched pairs | (32, 1/√2)–(16, 0): 16.0 · (32, 0.791)–(16, 0.5): 12.0 · (32, 0.9)–(16, 0.787): 6.08 · (32, 0.97)–(16, 0.939): 1.89 |
| Control | `W ~ N(0, 2(1−α²)/D)` plus per-block calibration on a fixed batch of 512 training images; forward and backward variance checked per layer |
| Nuisance | Learning rate tuned per condition on `LR_GRID`; SGD momentum 0.9, batch 128, 6 epochs fixed (justified in NB2 §1) |
| Seeds | Tuning: seed 0. Final: seeds 1–3; seeds 1–5 for the eight matched-pair conditions |
| Primary metric | Training loss after 6 epochs at the selected lr, evaluated on the full training set |
| Secondary | Training loss after epoch 1 (speed); width of the usable-lr range; test accuracy |
| Effective-depth test | The 32-block curve (loss against log L(1−α²)) predicts every 16-block condition inside its range; prediction errors with seed-bootstrap 95% intervals, plus a practical-equivalence bound |
| Mechanism | Jacobian similarity, stable rank and singular-value spectrum at initialisation |
| Sensitivity | 32 blocks, 12 epochs, selected lr, final seeds |
| Optional control | 16 blocks, width 362, α = 0: same block-parameter count as 32 × 256, effective depth 16 |
| Data | Fashion-MNIST, fixed 50k / 10k / 10k split |
| Compute | ≈ 190 short runs, ≈ 4 GPU-hours (less on the lab GPUs) |

### Priorities

1. 32-block α curve with the variance controls and the sensitivity check: everything the 60–70% band needs.
2. 16-block runs, matched pairs and the prediction test: the analysis that separates explanations (70%+).
3. Jacobian mechanism metrics.
4. The optional parameter-matched control.

Before committing `hypotheses.md`, run a small range check (32 blocks, α ∈ {0.5, 0.9}, 2–3 learning
rates): if the two are indistinguishable even after one epoch, discuss whether 32 blocks is deep
enough before freezing the design.

Pre-registered hypotheses and decision rules: [`hypotheses.md`](hypotheses.md).

## Repository layout

```text
README.md            this file
config.py            design constants shared by all notebooks
utils.py             question-independent helpers (seeding, data, evaluation, result files, plots)
nb1_baseline.ipynb   NB1: reproduce variance propagation; α = 1/√2 baseline with ≥3 seeds
nb2_experiment.ipynb NB2: the controlled experiment
nb3_teaching.ipynb   NB3: teaching notebook (~30 min) with exercises
hypotheses.md        written and committed BEFORE the main experiment; never edited afterwards
results/             one JSON per training run (committed; small)
checkpoints/         only the few weights QUICK mode needs (named quick_*.pt)
logs/                meeting_log.md, ai_log.md
data/                downloaded automatically, ignored by git
```

The code that matters for the question (model, initialisation, calibration, training
loop, Jacobian metrics) lives in the notebooks, as the brief requires. `utils.py` only
holds generic helpers.

## Environment

Marking runs every notebook top to bottom from a clean **`condaenv/dl2026`** kernel on
the lab machines. Treat that environment as the reference.

1. On a lab machine, run the first two cells of any notebook (or `python utils.py`) and
   copy the printed versions into the table below.
2. Pin the same versions in `requirements.txt` for local development.
3. Anything missing from dl2026 must be installed inside the notebook with `%pip install`,
   which counts towards the 5-minute QUICK budget, so keep dependencies minimal.

| Package | dl2026 version |
| --- | --- |
| python | 3.14.10 |
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |

Local setup: `pip install -r requirements.txt`, then `python utils.py` (smoke test:
downloads the data, checks shapes, round-trips a result file).

## Running the notebooks

Each notebook starts with `QUICK = True`:

- **QUICK = True** loads `results/` (and any `checkpoints/quick_*.pt`), redraws every figure
  and trains one small live example; must finish in under 5 minutes in total.
- **QUICK = False** reruns the full experiment with the same training code that produced
  the cache.

## Workflow rules

- Keep the repository **private**.
- One branch per task; merge through pull requests reviewed by one other member.
- Each notebook has one owner. Others propose changes via PR to avoid notebook merge conflicts.
- Clear notebook outputs before committing (Kernel → Restart & Clear Output), except for the final submission.
- Never commit `data/` or large checkpoints. Gradescope's total limit is 100 MB; a 32-block model is ≈ 9 MB.
- Commit `hypotheses.md` before the main sweep and record its commit hash in NB2. Do not edit it afterwards; label any later analysis as post hoc.
- Every result JSON records the git commit that produced it, so commit before long runs.
- Log every meeting in `logs/meeting_log.md` and every AI interaction worth noting in `logs/ai_log.md` as you go.

## Timeline

Meetings every Monday and Thursday (kick-off Fri 20 Nov). Full per-person task list in the group's planning doc.

| Gate | Date | Passed when |
| --- | --- | --- |
| G1 | Sun 22 Nov | Every condition builds, initialises and calibrates; variance checks pass; one training run saves a result; range check done |
| G2 | Mon 23 Nov | `hypotheses.md` committed; hash recorded in NB2 §0 |
| G3 | Thu 26 Nov | All 104 lr-sweep runs done; grid edges checked; lr selected per condition |
| G4 | Sun 29 Nov | All 55 final runs and the Jacobian metrics pushed to `results/` |
| G5 | Tue 1 Dec | Formative feedback submission; NB2 has first figures and a first effective-depth test |
| G6 | Sat 5 Dec | Sensitivity check (and optional control) done; NB2 §6–§10 drafted |
| G7 | Tue 8 Dec | Code freeze: all notebooks run top to bottom on a clean dl2026 kernel; QUICK < 5 min; repo < 100 MB |
| G8 | Thu 10 Dec | Video (5–8 min, everyone speaks), project log, final statement |
| G9 | Fri 11 Dec | Submitted by 18:00 (deadline 23:59) |

## Submission checklist

- [ ] Final challenge statement (Gradescope)
- [ ] NB1, NB2, NB3 run top to bottom on a clean dl2026 kernel; QUICK = True under 5 minutes
- [ ] `utils.py`, `config.py`, `results/`, needed checkpoints; total under 100 MB
- [ ] Project log: meeting log + AI-use log, 2–3 pages (max 4)
- [ ] Video, 5–8 minutes, every member speaks (Minerva)
