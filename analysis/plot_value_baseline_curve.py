"""Plot learning curves for the value-baseline comparison (2M-step, Acrobot).

Two PPO agents, both with a Markov reactive policy and Markov value TARGETS,
differing only in whether the advantage-computing critic is recurrent:
  * Markov value    -- critic hidden state zeroed every step (reset_critic=True)
  * recurrent value -- critic carries memory (reset_critic=False)

Left panel:  EVALUATOR undiscounted episode return (Acrobot: negative, higher=better).
Right panel: TRAINER value loss for the recurrent critic and the memory-free head
             (nr value loss), showing how well each critic fits its Markov targets.

Parses the training logs directly (no model needed).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("Markov value", "/tmp/markovval_train.log", "C1"),
    ("recurrent value (Markov target)", "/tmp/recval_train.log", "C0"),
]
TOTAL_TIMESTEPS = 1_986_560  # actual, from the timestep-checker in the logs


def parse(path):
    ret, vloss, nrvloss = [], [], []
    for raw in open(path, errors="ignore"):
        for sub in raw.replace("\r", "\n").split("\n"):
            m = re.search(r"Episode return mean: ([-0-9.]+)", sub)
            if m and "EVALUATOR" in sub:
                ret.append(float(m.group(1)))
            if "TRAINER" in sub:
                mv = re.search(r"(?<!Nr )Value loss: ([-0-9.]+)", sub)
                mn = re.search(r"Nr value loss: ([-0-9.]+)", sub)
                if mv:
                    vloss.append(float(mv.group(1)))
                if mn:
                    nrvloss.append(float(mn.group(1)))
    return ret, vloss, nrvloss


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for label, path, color in RUNS:
    if not os.path.exists(path):
        print(f"WARNING: missing {path}, skipping {label}")
        continue
    ret, vloss, nrvloss = parse(path)
    xs = [(i + 1) / len(ret) * TOTAL_TIMESTEPS for i in range(len(ret))]
    ax1.plot(xs, ret, marker="o", ms=3, color=color, label=label)
    if vloss:
        xv = [(i + 1) / len(vloss) * TOTAL_TIMESTEPS for i in range(len(vloss))]
        ax2.plot(xv, vloss, color=color, alpha=0.9, label=f"{label} (recurrent-critic loss)")

ax1.set_title("Eval episode return (Acrobot, higher=better)")
ax1.set_xlabel("environment timesteps")
ax1.set_ylabel("episode return (EVALUATOR)")
ax1.grid(alpha=0.3)
ax1.legend()

ax2.set_title("Critic value loss (TRAINER)")
ax2.set_xlabel("environment timesteps")
ax2.set_ylabel("value loss")
ax2.set_yscale("log")
ax2.grid(alpha=0.3)
ax2.legend()

fig.suptitle("PPO on Acrobot (2M steps): Markov vs recurrent value function "
             "(Markov policy + Markov targets both)")
fig.tight_layout()
out = os.path.join(REPO, "analysis", "value_baseline_curve_2M.png")
fig.savefig(out, dpi=130)
print(f"Wrote {out}")
