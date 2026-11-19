# Pre-registered hypotheses and decision rules

> Write this file in your own words, commit it **before** the main sweep, and do not
> edit it afterwards. NB2 quotes the commit hash below. Any analysis added later is
> labelled *post hoc* in NB2.

- Written by: _names_
- Date: _YYYY-MM-DD_
- Commit: _fill in after committing (`git log -1 --format="%h %ad"`)_

## 1. Question

_One sentence._

## 2. Hypotheses and predictions

For each hypothesis: what it says, why (theory or prior result), and what it predicts
for (a) training loss against α at 32 blocks and (b) training loss against effective
depth L(1−α²) for both depths.

| | Statement (own words) | Why | (a) loss vs α, 32 blocks | (b) loss vs L(1−α²), both depths |
| --- | --- | --- | --- | --- |
| H0 | | | | |
| H1 | | | | |
| H2 | | | | |
| H3 | | | | |

**Which do we expect, and why?** _…_

## 3. Primary metric (exact definition)

_e.g. mean cross-entropy over the full training set, evaluated in eval mode after the
final epoch, at each condition's selected learning rate; reported as mean ± sample sd
over seeds 1–3._

## 4. Decision rules

Decide these as a group and write them in your own words. The bullets in brackets are
starting points, not answers.

- A difference between two conditions counts as clear when: _…_
  [e.g. the seed-bootstrap 95% interval of the difference of means excludes 0]
- "Flat" / "decreasing" / "rises at small effective depth" means: _…_
- **Effective-depth test.** Prediction: interpolate the 32-block seed means linearly in
  log L(1−α²); every 16-block condition inside the 32-block range gets a predicted loss
  (for the four matched pairs the prediction is simply the partner's value). Error =
  observed − predicted, with a seed-bootstrap 95% interval (`config.BOOTSTRAP_SAMPLES`).
  - "Effective depth suffices" when: _…_
    [e.g. every interval contains 0 AND every |error| is below a practical bound ε]
  - "Effective depth does not suffice" when: _…_
    [e.g. at least two intervals exclude 0 in the same direction]
  - Otherwise we report: _…_ [e.g. "inconclusive"]
  - The practical bound ε = _…_ [e.g. 20% of the range of the 32-block curve], so that
    noisy seeds cannot make "suffices" the default.

## 5. Secondary metrics

- Training loss after epoch `config.EARLY_EPOCH` (training speed), same evaluation as the
  primary metric.
- Usable-lr range: a learning rate is usable if the run does not diverge and its final
  training loss is below τ = _value_ (set from the NB1 baseline: _how_). Width reported
  as _…_.
- Test accuracy: reported for the final runs only.

## 6. Conditions, tuning and seeds

- Conditions: as in `config.py` (`ALPHAS`, `MATCHED_PAIRS`); optional control
  `PARAM_MATCHED`: run / not run.
- Learning-rate grid: `config.LR_GRID`; extension rule: _…_
- Selection rule: _…_ (tie-break: _…_)
- Seeds: tuning seed 0; final seeds 1–3, and 1–5 for the eight matched-pair conditions
  (`config.final_seeds`); what the random state affects: _…_

## 7. Sensitivity check

_What changes, what stays fixed, and what result would mean the conclusion does not survive._

## 8. Commitments

- We will not change the primary metric, decision rules or conditions after seeing results.
- Diverged runs are kept and reported, never dropped.
