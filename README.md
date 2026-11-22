# COMP3620 CW3 · Residual paths and effective depth

Can a 32-block residual MLP predict the training loss of a 16-block MLP at the same effective depth, after checking signal scale at initialisation?

## Current work

| Member | Responsibility |
|---|---|
| A | Initialisation, calibration, variance and Jacobian checks; freeze the hypotheses; conclusions with D |
| B | Model and training pipeline, baseline, NB2 integration, QUICK mode and submission checks |
| C | Learning-rate selection, 32-block runs, results figures, sensitivity check and optional control |
| D | 16-block runs, effective-depth prediction test (NB2 §8), conclusions with A |
| E | Teaching notebook, meeting and AI logs, video and reader feedback |

## Handoff

1. Run a short end-to-end training job on the lab machine.
2. After the baseline and range check, freeze [hypotheses.md](hypotheses.md) before the main sweep. Record the commit in NB2 §0; label later additions *post hoc*.
3. C and D run the 32- and 16-block conditions in parallel using [config.py](config.py). Push real run JSONs to `results/` the day they finish. B integrates NB2.
4. Use the selected learning rate for each condition and the configured final seeds. Report missing or diverged finals rather than silently averaging the remaining seeds.
5. Validate QUICK mode on a clean lab kernel (under five minutes per notebook) before submission. Record meetings and AI use in `logs/`.

| Experiment record | Value |
|---|---|
| Pre-registration commit | |
| Final training results | |
| Effective-depth verdict | |
| Lab QUICK timings | |
