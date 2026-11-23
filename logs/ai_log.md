# AI-use log

| Date | Workstream | Tool | Assistance | Member's independent review / corrections |
|---|---|---|---|---|
| 2026-11-22 | D, stage 0 | OpenAI Codex | Reviewed the project plan; implemented the synthetic prediction check, tests and repository cleanup. No training data or conclusions generated. | |
| 2026-11-23 | Weiming Chen, stage 0–1 | Claude | Used to review the plan for NB1 §2–§4 and to help find one bug. The first version of `calibrate` computed the block moments as float64 tensors, which fails on the MPS development machine; the corrected version keeps the reductions on the device and finishes the scalar arithmetic in Python floats. Also used to check the derivation and to read the decision rules back for unclear wording. No training runs and no experiment results were produced. | |
