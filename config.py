"""Experimental design constants shared by NB1, NB2 and NB3.

Change values here only by group decision, and record the change in
logs/meeting_log.md. After hypotheses.md is committed, the main-experiment
settings below are frozen.
"""
import math

# ---- Architecture -----------------------------------------------------------
WIDTH = 256                      # hidden width D
IN_DIM, N_CLASSES = 784, 10      # flattened Fashion-MNIST


def effective_depth(depth: int, alpha: float) -> float:
    """Variance-weighted mean path length at initialisation, L * (1 - alpha^2)."""
    return depth * (1.0 - alpha ** 2)


def alpha_for_effective_depth(depth: int, eff_depth: float) -> float:
    """The alpha that gives a `depth`-block network the effective depth `eff_depth`."""
    return math.sqrt(1.0 - eff_depth / depth)


# ---- Manipulated variable: skip weight alpha -------------------------------
# alpha is the only manipulated variable. The same manipulation is repeated at a
# second depth (16 blocks) so that effective depth can be tested with matched pairs.
ALPHA_UDL = 2 ** -0.5            # the "0.71" condition: exactly the UDL Fig. 11.6b rescaling
DEPTH_MAIN = 32
# 32-block partner for (16, 0.5): fills the gap between effective depths 16 and 6.08
ALPHA_32_PAIR_D = alpha_for_effective_depth(32, effective_depth(16, 0.5))    # 0.7906
ALPHAS_32 = (0.0, 0.5, ALPHA_UDL, ALPHA_32_PAIR_D, 0.9, 0.97)

# 16-block partners chosen so their effective depth equals a 32-block condition exactly
ALPHA_16_PAIR_B = alpha_for_effective_depth(16, effective_depth(32, 0.9))    # 0.7874
ALPHA_16_PAIR_C = alpha_for_effective_depth(16, effective_depth(32, 0.97))   # 0.9390
ALPHAS_16 = (0.0, 0.5, ALPHA_UDL, ALPHA_16_PAIR_B, 0.9, ALPHA_16_PAIR_C, 0.97)

ALPHAS = {32: ALPHAS_32, 16: ALPHAS_16}
ALPHA_BASELINE = ALPHA_UDL       # NB1 control condition (32 blocks)

# Effective-depth-matched pairs: ((depth, alpha), (depth, alpha)); identical L(1 - alpha^2)
MATCHED_PAIRS = (
    ((32, ALPHA_UDL), (16, 0.0)),               # A: 16.00
    ((32, ALPHA_32_PAIR_D), (16, 0.5)),         # D: 12.00
    ((32, 0.9), (16, ALPHA_16_PAIR_B)),         # B:  6.08
    ((32, 0.97), (16, ALPHA_16_PAIR_C)),        # C:  1.89
)

# Optional control (priority 4): a 16-block network with as many block parameters as
# the 32-block one (16 * 362^2 ~= 32 * 256^2) and effective depth 16, to separate
# parameter count from effective depth in pair A. Save its runs with tag="w362".
PARAM_MATCHED = {"depth": 16, "width": 362, "alpha": 0.0}

# ---- Fixed settings ---------------------------------------------------------
OPTIMISER = "SGD"
MOMENTUM = 0.9
BATCH_SIZE = 128
EPOCHS = 6                       # confirm on the NB1 baseline that the loss has flattened
EARLY_EPOCH = 1                  # secondary metric: training loss after this epoch (speed)
EPOCHS_SENSITIVITY = 12          # sensitivity check (32 blocks, selected lr, final seeds)

# ---- Nuisance hyperparameter tuned per condition ---------------------------
LR_GRID = (3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2)
# Rule: if the best lr, or either edge of the usable range, sits on the grid
# edge, extend the grid until both ends contain a failing lr.

# ---- Seeds and fixed data choices -------------------------------------------
TUNE_SEED = 0
FINAL_SEEDS = (1, 2, 3)
MATCHED_PAIR_SEEDS = (1, 2, 3, 4, 5)   # the eight matched-pair conditions get two extra seeds
SPLIT_SEED = 0                   # fixes the 50k / 10k train / validation split
CALIB_SEED = 1                   # fixes the 512-image calibration batch
N_VAL = 10_000
N_CALIB = 512

# ---- Analysis -------------------------------------------------------------------
BOOTSTRAP_SAMPLES = 2000         # resampling seeds within each condition
BOOTSTRAP_SEED = 0


def final_seeds(depth: int, alpha: float) -> tuple:
    """Seeds for the final runs of a condition (more for matched-pair conditions)."""
    in_pair = any(abs(a - alpha) < 1e-9 and d == depth
                  for pair in MATCHED_PAIRS for d, a in pair)
    return MATCHED_PAIR_SEEDS if in_pair else FINAL_SEEDS


# ---- Paths --------------------------------------------------------------------
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
