"""Plot discounted-return learning curves for the 2M-step DB Acrobot runs.

Primary panel: TRAINER "Mean return" -- the gamma=0.99 discounted rollout return
the Direct Backprop objective actually optimizes, logged densely per evaluation.
Secondary panel: EVALUATOR undiscounted episode return (higher=better, ceiling +400).

Parses the training logs directly (no model needed).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("recurrent (memory)", "/tmp/db_rec_memory_2M.log", "C0"),
    ("recurrent (memoryless)", "/tmp/db_rec_memfree_2M.log", "C1"),
]
TOTAL_TIMESTEPS = 1_966_080  # actual, from the timestep-checker in the logs


def parse(path):
    disc, undisc = [], []
    for raw in open(path, errors="ignore"):
        for sub in raw.replace("\r", "\n").split("\n"):
            m = re.search(r"Mean return: ([-0-9.]+)", sub)
            if m and "TRAINER" in sub:
                disc.append(float(m.group(1)))
            m2 = re.search(r"Episode return mean: ([-0-9.]+)", sub)
            if m2 and "EVALUATOR" in sub:
                undisc.append(float(m2.group(1)))
    return disc, undisc


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for label, path, color in RUNS:
    disc, undisc = parse(path)
    # x-axis: evenly spaced env timesteps (one eval per equal chunk).
    xs_d = [(i + 1) / len(disc) * TOTAL_TIMESTEPS for i in range(len(disc))]
    xs_u = [(i + 1) / len(undisc) * TOTAL_TIMESTEPS for i in range(len(undisc))]
    ax1.plot(xs_d, disc, marker="o", ms=3, color=color, label=label)
    ax2.plot(xs_u, undisc, marker="o", ms=3, color=color, label=label)

ax1.set_title("Discounted return (γ=0.99) — DB objective")
ax1.set_xlabel("environment timesteps")
ax1.set_ylabel("discounted rollout return (TRAINER)")
ax1.grid(alpha=0.3)
ax1.legend()

ax2.set_title("Undiscounted episode return (eval)")
ax2.set_xlabel("environment timesteps")
ax2.set_ylabel("episode return (EVALUATOR)")
ax2.grid(alpha=0.3)
ax2.legend()

fig.suptitle("Direct Backprop on differentiable Acrobot (2M steps): recurrent vs memoryless GRU")
fig.tight_layout()
out = os.path.join(REPO, "analysis", "db_learning_curve_2M.png")
fig.savefig(out, dpi=130)
print(f"Wrote {out}")
