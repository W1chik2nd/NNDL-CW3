# Hypotheses and decision rules

We fill this file in and commit it **before** the main sweep. The commit hash goes into NB2 §0.
After that we do not change it. Anything we add later is labelled *post hoc*.

**Authors and date:** Weiming Chen, 2026-11-23

## Question

Every condition has the same forward variance and the same backward variance. So the only thing
that changes between conditions is the skip weight α, and with it the path structure. Two things
we want to know:

1. How much does α change how well a deep residual MLP trains on a fixed budget?
2. Does α work only through the effective depth L(1−α²)?

The two core figures are (i) final train loss against α at 32 blocks, and (ii) final train loss
against effective depth, with the 16-block and the 32-block conditions on the same axes.

## H0, H1, H2, H3

| ID | What we think | What we expect at 32 blocks (loss vs α) | What we expect for both depths (loss vs effective depth) |
|---|---|---|---|
| H0 | Variance is enough. Once the forward and backward variance are fixed, path structure does not matter any more. | Flat. No clear difference anywhere. | Flat at both depths. |
| H1 | Path structure matters, and the effective depth is enough to describe it. | Loss goes down as α grows. | The 32-block curve can predict the 16-block results: every prediction error is small, and every interval covers 0. |
| H2 | Path structure matters, but the effective depth is not enough. | Loss goes down as α grows (same as H1). | At least two prediction errors sit clearly away from 0, in the same direction. |
| H3 | Too small an effective depth is also bad. When the branch is very weak, the network learns slowly. | Loss goes down and then up again; the lowest point is in the middle. | The curve goes up again at the small end of effective depth. |

H3 can be true at the same time as H1 or H2.

Why we expect this. At α = 0 there is no skip path, so the network is just a plain deep net, and
the literature says plain deep nets train worse. That part is not interesting. The interesting part
is what happens between α = 0.5 and α = 0.97, and whether one number — the effective depth — can
explain all of it.

## Primary and secondary metrics

- **Primary:** final train loss after 12 epochs, recomputed in `eval` mode on the whole training
  set (never a running average over training batches).
- **Secondary 1:** train loss after epoch 1. This shows training *speed*, and it still tells
  conditions apart when all the 6-epoch losses have flattened.
- **Secondary 2:** width of the usable learning-rate range (see τ below). This shows how easy the
  condition is to tune.
- **Secondary 3:** test accuracy. This shows the effect on generalisation.
- Every number is a mean ± sd over seeds. Diverged runs never enter a mean.

## Loss-curve shape criteria

For the 32-block curve, and for both depths on the effective-depth axis:

- **flat**: no neighbouring pair of conditions differs clearly, and the whole curve stays inside
  ±ε.
- **goes down**: every clear difference between neighbouring conditions points the same way
  (loss smaller at larger α, or the other way round — we say which one we see).
- **goes up at the end**: the condition with the smallest effective depth differs clearly from the
  lowest point of the curve.

## Practical difference margin ε

- A clear difference is a statistical statement. If the seeds are noisy the interval gets wide, and
  a wide interval easily covers 0. Then we would say "no difference" only because we could not
  measure well. ε stops that from happening.
- **ε = 20% of the amplitude of the 32-block curve**: ε = 0.20 × (max − min) of the 32-block
  condition means of the final train loss.
- The same ε is used for every prediction error of the effective-depth test.

## Intervals, and the effective-depth (prediction) test

- **Interval method:** 95% **t interval** on the mean over seeds (the helper is added to
  `utils.py` as part of the test code). We do not use the seed bootstrap; the bootstrap wording
  in the old template is replaced by this.
- **Clear difference:** the 95% t interval of the difference between two condition means does not
  include 0.
- **Prediction test:** put the 32-block seed means on a straight line against log L(1−α²) and
  interpolate. Predict the six 16-block conditions whose effective depth falls inside the 32-block
  range. α = 0.97 at 16 blocks has effective depth 0.95, outside the range, so we do not predict it.
  Four of the six are matched pairs, so the prediction is just the partner's value. The other two
  (α = 1/√2 and α = 0.9) use the interpolated line.
- **Pairing:** go seed by seed. A 16-block condition is compared with the 32-block prediction made
  with the **same seed** (pairs have seeds 1–5, the others 1–3). Then take the interval over those
  seed-level errors. Error = observed − predicted.
- **"Effective depth is enough":** every interval covers 0 **and** every |error| < ε.
- **"Effective depth is not enough":** at least two intervals exclude 0 and point the same way.
- **Otherwise:** the verdict is "not sure". We say which of the two conditions failed, and whether
  the problem is seed noise or one single condition sitting off the curve.

## Overall verdict rules

| What we see | What we conclude |
|---|---|
| Both core figures flat | H0: in this range, controlling the variance is enough and path structure no longer changes the final train loss |
| Loss changes with effective depth, prediction test "enough" | H1: path structure matters, and the effective depth alone describes it |
| Loss changes with effective depth, prediction test "not enough" | H2: something outside the effective depth matters too. Without the optional control we can only write "parameter count or training process" |
| Prediction test "not sure" | We report the uncertainty, with the reason. We do not pick the nicer story |
| Curve goes up again at small effective depth | H3: a very weak branch slows learning. This can hold together with H1 or H2 |
| Same final loss, but different epoch-1 loss or usable-lr width | Path structure changes training speed or tuning difficulty, not the final level. We word the conclusion that way |

If the curves differ only in the learning-rate range, we say so and do not claim a difference in
the final loss.

## Incomplete or diverged runs

- A diverged run stays in the file with `diverged: true` and is reported on its own. It never enters
  a mean and is never quietly replaced.
- If the selected learning rate diverges for some final seeds, we drop one grid step for that
  condition, write it down, and keep the seeds we have.
- A condition with fewer than 3 usable final seeds is called **incomplete**. We do not invent the
  missing seeds.

## Usable learning-rate threshold τ

- A learning rate is **usable** if the run does not diverge **and** its final train loss is ≤ τ.
- **Rule:** τ = 1.25 × L_base, where L_base is the final train loss of the baseline
  (32 blocks, α = 1/√2, selected learning rate, seed 0).
- τ is fixed **after** the baseline run and before the main sweep, then written on the line below.
  The rule is frozen here; only the number is still missing.

  `τ = 1.25 × L_base = 0.219348`  (12-epoch baseline: L_base = 0.175479 at lr = 5e-3, seed 0)

- Why 1.25: the question is "can this condition be trained at all with a sensible learning rate",
  not "does it reach the very best loss". A 25% margin still separates big failures from small
  differences between neighbouring learning rates.
- The usable range is compared by **width** only, not by position, so a condition that is simply
  shifted in learning rate is not punished.

## Learning-rate grid, extension and tie-break

- Grid: `config.LR_GRID` = 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2. Tuned per condition
  with `config.TUNE_SEED` (seed 0). No decay, no clipping.
- **Extension:** if the best learning rate, or either end of the usable range, sits on an edge of
  the grid, we add points outward (×3 and ÷3) until both ends contain a failing learning rate.
- **Selection:** the learning rate with the lowest final train loss of seed 0.
- **Tie-break:** if two neighbouring learning rates differ by less than 5%, we add one seed and
  compare the means. Still within 5%, we take the smaller learning rate.

## Conditions, seeds and sensitivity check

- 13 conditions. 32 blocks: α ∈ {0, 0.5, 1/√2, 0.791, 0.9, 0.97}. 16 blocks:
  α ∈ {0, 0.5, 1/√2, 0.787, 0.9, 0.939, 0.97} (`config.ALPHAS`).
- Main budget 12 epochs (`config.EPOCHS`). NB1 §7.1 records why: the loss was still dropping at
  epoch 6 by the group's own 5% rule, and it is still dropping at 12 — keeping 12 is a recorded
  departure from the literal rule, on the grounds that the ranking, not the level, is what the
  analysis uses, and the 24-epoch sensitivity check tests the shape directly.
- Final seeds 1–3. The eight matched-pair conditions use seeds 1–5 (`config.final_seeds`).
- Data split (seed 0) and the 512-image calibration batch (seed 1) are the same for every
  condition. A seed only changes the weight initialisation and the batch order.
- Sensitivity check: 32 blocks, all six α, the selected learning rate, 24 epochs
  (`config.EPOCHS_SENSITIVITY`), seeds 1–3. It asks whether the *shape* of loss vs α survives
  longer training, not only whether training is slower.

## What we will not do after this commit

- Change ε, the interval method, the grid or the seeds because the results look awkward.
- Add or drop a condition.
- Everything extra (mechanism figures, optional control, sensitivity check) is listed above as a
  pre-registered check; anything else we add later is marked post hoc.
