# Recurrent vs. Non-Recurrent Value Drift

This directory contains an experiment measuring **how off-policy a learned value
function becomes as the policy improves** — i.e. how much "value iteration" is
happening in a deep RL system — and whether a **recurrent** value function
behaves differently from an otherwise-identical **non-recurrent** one.

The pipeline is PPO + GRU, fully in JAX (Stoix's Anakin system), with checkpoints
saved throughout training. It has been run on two fully-observable tasks —
**CartPole-v1** (GRU hidden 32) and **Acrobot-v1** (GRU hidden 64) — plus a third
condition on Acrobot with a **forced non-recurrent (reactive) policy** but a
recurrent value head, to isolate cause from effect.

> **TL;DR of findings** (see [§6](#6-findings)):
> - On **CartPole** the recurrent value head learns to *ignore* its memory — the
>   task is effectively Markov, so recurrent and non-recurrent heads are identical.
> - On **Acrobot** the recurrent value head *strongly uses* history (ablation:
>   swapping the preceding history moves the value by ≈ its full across-state
>   spread).
> - **But that history-use is spurious.** It happens *even when the policy is
>   forced to be reactive* (so the true value is provably Markov), and it does
>   **not** improve accuracy: on-distribution, recurrent ties the non-recurrent
>   head (recurrent actor) or is slightly *worse* (reactive actor). The tempting
>   "recurrent value tracks a memoryful policy" story is **falsified** — a reactive
>   policy's recurrent critic uses history just as much.
> - A measurement bug was found and fixed along the way (the analysis drove the
>   reactive actor recurrently); see [§6e](#6e-a-measurement-bug-worth-recording).

---

## 1. The question

Consider a policy `pi_u` learnt after `u` updates and a much later policy
`pi_u'` (`u << u'`). We compare the **later** value function `V_hat^{u'}` against
the **ground-truth** values of both policies at states `S` drawn from `pi_u`'s
trajectories:

```
E_{tau ~ pi_u} [ V_hat^{u'}(tau) - V_{pi_u}(S)  ]^2      (drift vs the OLD policy)
E_{tau ~ pi_u} [ V_hat^{u'}(tau) - V_{pi_u'}(S) ]^2      (agreement w/ the NEW policy)
```

- `S` is the final state of a fixed-length prefix of a trajectory `tau ~ pi_u`.
- `V_{pi}(S)` is the true value of `S` under policy `pi`, estimated by Monte-Carlo
  rollouts branched from `S`.
- The gap between the two errors measures how far the value function has "moved
  on" from `pi_u`'s returns.

We do this for **two value heads trained side-by-side on the same data**:

| Head | Input | Notation |
|------|-------|----------|
| **Recurrent** | the whole trajectory `tau` (GRU carries history) | `V_hat^{u'}(tau)` |
| **Non-recurrent** | the single state `S` (GRU hidden state zeroed every step) | `V_hat_nr^{u'}(S)` |

Both heads have the **identical architecture**; the non-recurrent one simply has
its hidden state zeroed at every step, so its output is a pure function of the
current observation. This isolates the effect of recurrence on value drift.

---

## 2. What was added to the codebase

| File | Purpose |
|------|---------|
| `stoix/systems/ppo/anakin/rec_ppo_dual_value.py` | Recurrent PPO + an auxiliary non-recurrent value head, trained together. |
| `stoix/configs/system/ppo/rec_ppo_dual_value.yaml` | System hyperparameters (`system_name: rec_ppo_dual_value`). |
| `stoix/configs/default/anakin/default_rec_ppo_dual_value.yaml` | Hydra default composition for the dual-value system. |
| `bash_scripts/run_rec_ppo_dual_value_cartpole.sh` | One-command CartPole training run (GRU hidden 32) that produces the checkpoints. |
| `bash_scripts/run_rec_ppo_dual_value_acrobot.sh` | Same, on Acrobot (GRU hidden 64), stable UID `acrobot_h64`. |
| `stoix/systems/ppo/anakin/rec_ppo_nonrec_actor.py` | Control variant: **non-recurrent (reactive) actor** + recurrent value head. Isolates whether the policy drives value history-use (§6d). |
| `stoix/configs/system/ppo/rec_ppo_nonrec_actor.yaml` | System config (`system_name: rec_ppo_nonrec_actor`). |
| `stoix/configs/default/anakin/default_rec_ppo_nonrec_actor.yaml` | Hydra default composition for the control variant. |
| `bash_scripts/run_nonrec_actor_acrobot.sh` | Trains the control variant on Acrobot, UID `acrobot_nonrec_actor_h64`. |
| `bash_scripts/run_rec_ppo_cartpole.sh` | The plain recurrent-PPO run (no extra head), for reference. |
| `analysis/measure_value_drift.py` | The value-drift measurement script (§4). Auto-detects env/arch/actor-type from checkpoint metadata; `--model-name` selects the system. |
| `analysis/measure_history_ablation.py` | The history-ablation diagnostic: does the recurrent value/policy actually *use* memory? (§5). |

### How the non-recurrent head works
It is a second `RecurrentCritic` with the same architecture and its own
parameters and Adam optimizer. On every step it is called with the GRU **reset
flag = True**, which zeroes the hidden state (`ScannedRNN` overwrites the carry
with zeros on reset), making its output depend only on the current observation.

- **It never affects the policy.** The actor still trains on the recurrent
  critic's advantages, so policy learning is identical to plain recurrent PPO
  (same seed → same policy trajectory).
- It is trained with its **own independent GAE bootstrap** (its own value
  predictions → its own targets), so it is a genuine, self-consistent value
  estimator of the policy — differing from the recurrent critic only in
  recurrence.
- Both heads (`actor_params`, `critic_params`, `nr_critic_params`) are saved in
  every checkpoint.

---

## 3. Step 1 — Train the policy and generate checkpoints

Run the dual-value training (this is what produces the checkpoints the analysis
reads):

```bash
./bash_scripts/run_rec_ppo_dual_value_cartpole.sh
```

This runs recurrent PPO + the non-recurrent head on `CartPole-v1` with:

| Setting | Value |
|---------|-------|
| Environment | `gymnax/CartPole-v1` |
| Parallel environments | 4 |
| Total timesteps | 500,000 |
| GRU hidden size | 32 (actor + both critics) |
| Evaluations / checkpoints | 50 |
| `num_minibatches` | 1 (required: with 4 envs the recurrent batch has 4 sequences, and `num_minibatches` must divide 4) |

For the second task, run:

```bash
./bash_scripts/run_rec_ppo_dual_value_acrobot.sh
```

Identical setup on `gymnax/Acrobot-v1` but **GRU hidden size 64** and a stable
checkpoint UID `acrobot_h64` (rather than a timestamp), so its checkpoints land in
`checkpoints/rec_ppo_dual_value/acrobot_h64/`. Acrobot is fully observable but has
richer dynamics (6-D observation, momentum that accumulates), and its rewards are
negative (−1/step until the goal), so values are negative.

### Checkpointing details
- Checkpoints are written to `./checkpoints/rec_ppo_dual_value/<uid>/<step>/`.
  The `<uid>` is a timestamp for CartPole (e.g. `20260708154913`) and the stable
  string `acrobot_h64` for Acrobot.
- Stoix saves **one checkpoint per evaluation**, evenly spaced across training.
  With 500k steps and 50 evaluations, a checkpoint lands roughly every ~10k
  steps. Because each step count must be a multiple of
  `rollout_length * num_envs = 512`, the actual cadence is **9,728 steps**:

  ```
  checkpoint index k  ->  step = 9728 * k
  ckpt 1 = 9,728    ckpt 2 = 19,456   ckpt 5 = 48,640   ...   ckpt 45 = 437,760   ckpt 50 = 486,400
  ```

- 50 checkpoints total; `checkpoints/` is git-ignored.

### First-run note
The first launch spends a few minutes cloning `mujoco_menagerie` (a one-time
side effect of importing Stoix's environment registry). This is cached
afterwards. Training itself is fast on CPU.

### Plain recurrent PPO (no extra head), for reference
```bash
./bash_scripts/run_rec_ppo_cartpole.sh
```
Same setup, but only the standard recurrent critic. **Note:** its checkpoints do
*not* contain the non-recurrent head, so they cannot be used for the comparison
in Step 2. Use the dual-value run for the analysis.

---

## 4. Step 2 — Measure value drift

Once checkpoints exist:

```bash
# Compute the Monte-Carlo ground truth once (caches it), then score.
.venv/bin/python analysis/measure_value_drift.py

# Re-score from the cache instantly (e.g. to re-print, plot, or tweak output).
.venv/bin/python analysis/measure_value_drift.py --from-cache
```

By default this compares **u = checkpoint 3 (step 29,184)** — an early, weak
policy — against **u' = checkpoint 45 (step 437,760)** — a near-converged one,
and prints the four MSEs plus MC-noise-corrected versions.

For Acrobot, pass its run UID (required whenever more than one run exists):

```bash
.venv/bin/python analysis/measure_value_drift.py --run-uid acrobot_h64 \
  --ckpt-u 48640 --ckpt-uprime 437760 \
  --out analysis/value_drift_acrobot_ckpt5v45.json
```

**On-distribution variant.** Setting `--ckpt-u` equal to `--ckpt-uprime` scores a
checkpoint's value heads against *its own* states — removing distribution shift so
any recurrent-vs-non-recurrent MSE gap is purely representational. This is what
isolates "does recurrence make the value more *accurate*" from "value drift":

```bash
.venv/bin/python analysis/measure_value_drift.py --run-uid acrobot_h64 \
  --ckpt-u 437760 --ckpt-uprime 437760 \
  --out analysis/value_drift_acrobot_ondist45.json
```

### What the script does (faithful to the training code)
1. **Auto-matches env + architecture** — reads the environment, GRU hidden dim /
   cell type, and `gamma` from the checkpoint metadata, so the rebuilt env and
   networks exactly match the restored parameters. (This is why the same script
   handles both CartPole and Acrobot with no edits.)
2. **Samples `tau ~ pi_u`** on the eval env, carrying/resetting the recurrent
   hidden state exactly as in training. `S` = state after `--prefix-len` steps;
   only trajectories that survive the full prefix are kept (so `S` is genuinely
   non-terminal).
3. **Faithful policy memory.** Each policy's Monte-Carlo continuations from `S`
   start from the hidden state *that policy* would actually hold at `S`,
   reconstructed by replaying the prefix through *its own* actor. `pi_u` and
   `pi_u'` therefore each get their own memory — the GRU is **not** reset at the
   branch point.
4. **Value heads at `S`.** The recurrent head is read at `S = s_L` (the last
   prefix observation is appended before reading the value), matching the state
   the Monte-Carlo ground truth branches from and the observation the
   non-recurrent head uses.
5. **Monte-Carlo ground truth.** Branches the gymnax env at `S` and rolls out
   many gamma-discounted continuations under `pi_u` and `pi_u'`, masking any
   reward earned after termination (gymnax auto-resets).
6. Prints the 2×2 MSE table (raw + MC-noise-corrected) and writes JSON.

### Cost, caching, and accuracy
The expensive part is the Monte-Carlo:
`num_traj * mc_rollouts * max_horizon` env steps, for **both** policies. Two
mechanisms keep this manageable:

- **Caching.** The ground truth (state set + MC values + per-state variances) is
  independent of which value head you score, so it is computed once and saved to
  an auto-named `.npz` under `analysis/`. `--from-cache` re-scores instantly and
  never re-pays the rollout cost.
- **MC-noise correction.** The script records per-state rollout variance and
  reports an **unbiased estimate of the MSE against the *true* value** by
  subtracting the Monte-Carlo estimator's mean squared standard error. This lets
  a modest `--mc-rollouts` give the same expected answer as a huge one.
- **Short horizon.** With `gamma = 0.99`, weights beyond ~300 steps contribute
  <5% of the discounted mass (`0.99^300 ≈ 0.05`), so `--max-horizon 300` (the
  default) captures nearly all of it while cutting rollout length by ~40% vs the
  500-step episode cap.

### Key options
| Flag | Default | Meaning |
|------|---------|---------|
| `--run-uid` | auto-detected | Checkpoint run folder (needed if more than one run exists). |
| `--ckpt-u` | `29184` | Timestep of the early policy `u` (ckpt 3). |
| `--ckpt-uprime` | `437760` | Timestep of the late policy `u'` (ckpt 45). |
| `--num-traj` | `256` | Number of trajectories / states `S`. |
| `--prefix-len` | `64` | Prefix length `L`; `S` is the state after `L` steps. |
| `--mc-rollouts` | `64` | Monte-Carlo continuations per state, per policy. |
| `--max-horizon` | `300` | Max steps per continuation. |
| `--seed` | `0` | RNG seed for sampling + rollouts. |
| `--from-cache` | off | Skip Monte-Carlo; score from an existing cache. |
| `--recompute` | off | Force recomputation even if a cache exists. |
| `--out` | `analysis/value_drift_results.json` | Output JSON path. |
| `--overrides` | — | Extra Hydra overrides (rarely needed; architecture is auto-matched). |

### Reading the output
```
                                vs OLD policy V_pi_u   vs NEW policy V_pi_u'
Recurrent  V_hat(tau)                    <MSE>                  <MSE>
Non-recur. V_hat_nr(S)                   <MSE>                  <MSE>
```
A large "vs OLD" and small "vs NEW" error means the value function has iterated
away from `pi_u`'s returns toward `pi_u'`'s — the "value iteration" signal. The
MC-noise-corrected block below it is the unbiased estimate vs the *true* value;
the reported `MC standard-error^2` is what was subtracted.

Results are also saved to the `--out` JSON, and the ground-truth cache to a
`.npz` in `analysis/` (both git-ignored).

---

## 5. History ablation — does recurrence actually get used?

The value-drift MSEs can't tell you *why* a recurrent and non-recurrent head
agree or differ. `analysis/measure_history_ablation.py` answers the mechanistic
question directly: **does the recurrent value (and policy) actually depend on
history, or has it collapsed to a function of the current state?**

```bash
.venv/bin/python analysis/measure_history_ablation.py --run-uid acrobot_h64
.venv/bin/python analysis/measure_history_ablation.py --run-uid 20260708154913  # CartPole
```

It sweeps a set of checkpoints (default indices `1 3 5 10 20 30 45`) and, for
each, samples states `S` from that policy and reads the network **at `S` under
different preceding histories that all end at the same `S`**:

- **real** — the true prefix the policy experienced.
- **swap** — a *different real trajectory's* prefix (a batch derangement), then
  `S`. This is the key comparison: the GRU is warmed on a real,
  dynamically-coherent history (in-distribution), so any change reflects genuine
  dependence on *which* history preceded `S`.
- **scrambled** — the same prefix with its time-order permuted (secondary).

> **Why not compare against a zeroed hidden state?** The recurrent net is never
> trained on a zero hidden state mid-episode — it always acts warmed-up. Feeding
> a zero state at `S` is out-of-distribution, so `|V_real − V_zero|` measures that
> artifact, not history use. (An earlier version of this script did exactly that
> and produced a spurious 30+ value-unit "sensitivity"; the swap test fixes it.)

Two quantities are reported per checkpoint:

- **Value sensitivity** — `mean_S |V_real(S) − V_swap(S)| / std_S V_real(S)`.
  Read the raw `|real−swap|` alongside `spread = std_S V_real` (the number is
  unstable when the critic is near-constant across states). `~0` ⇒ the value
  ignores history.
- **Policy KL** — `mean_S KL( pi(·|S, real) ‖ pi(·|S, swap) )` in nats, with the
  policy entropy for context. `~0` ⇒ a reactive (Markov) policy; `>0` ⇒ the
  policy uses memory. **This is the load-bearing check**: for a *recurrent*
  policy, the true value legitimately depends on the hidden state (because future
  actions do), so a history-dependent policy is what *licenses* a
  history-dependent value.

It's forward-passes only (no Monte-Carlo), so it runs in well under a minute.
Output is written to `analysis/history_ablation_<run_uid>.json`.

---

## 6. Findings

### 6a. Value drift is real and large (both tasks)
Comparing an early policy `u` to a late `u'`, the later value head is a *much*
better fit to the **old** policy's returns than the **new** one's, at the states
the old policy visits — the value function has "moved on". Example (Acrobot,
ckpt 5 vs 45; MSE):

| | vs OLD `V_pi_u` | vs NEW `V_pi_u'` |
|---|---|---|
| Recurrent `V_hat(tau)` | 48.6 | 93.6 |
| Non-recurrent `V_hat_nr(S)` | 44.4 | 95.8 |

This is the sample-based "value iteration is local / forgets old values" effect:
`V_hat^{u'}` was trained on `pi_u'`'s state distribution and extrapolates poorly
onto `pi_u`'s off-distribution states.

### 6b. On a Markov task (CartPole), recurrence is unused
History ablation, CartPole (GRU 32):

| ckpt | `\|real−swap\|` | spread | sens/swap | policy KL | V_rec |
|---|---|---|---|---|---|
| 5 | 0.035 | 0.117 | 0.30 | 0.30 | 29.1 |
| 30 | 0.011 | 0.126 | 0.09 | 0.54 | 92.6 |
| 45 | 0.003 | 0.025 | 0.11 | 0.36 | 91.6 |

As training converges the value's history sensitivity **decays toward zero**
(`|real−swap|` → 0.003). CartPole is fully observable *and Markov* — the current
observation is a sufficient statistic — so the optimal value is a function of the
state alone and the recurrent critic correctly learns to ignore its memory. The
recurrent and non-recurrent heads become numerically identical (their means
matched to 2 decimals in the drift experiment).

### 6c. On Acrobot, the recurrent value genuinely uses history
History ablation, Acrobot (GRU 64):

| ckpt | `\|real−swap\|` | spread | sens/swap | policy KL | V_rec |
|---|---|---|---|---|---|
| 5 | 1.55 | 2.22 | 0.70 | 0.46 | −37.2 |
| 20 | 1.45 | 1.62 | 0.89 | 0.20 | −17.6 |
| 45 | 2.34 | 2.20 | 1.06 | 0.12 | −17.3 |

Here the value's history sensitivity **grows and persists**: by convergence,
swapping the preceding history moves `V_rec(S)` by ≈ the entire across-state
spread (`sens/swap ≈ 1.0`). The recurrent critic is *not* ignoring memory — the
opposite of CartPole. The *why* is answered by the controlled experiment in §6d.

### 6d. The controlled test: it is NOT the policy driving value history-use
The tempting explanation was: "the value uses history *because the policy does*."
For a recurrent policy the system is Markov in `(env_state, hidden_state)`, so
two trajectories reaching the same physical state with different memory act
differently thereafter and genuinely have different values — which a recurrent
critic must track.

To test this we trained a third condition: **a forced non-recurrent (reactive)
actor with a still-recurrent value head** (`rec_ppo_nonrec_actor`, Acrobot). The
actor's hidden state is zeroed every step, so the policy is provably Markov and
`V^pi(S)` cannot legitimately depend on history. Ablation at ckpt 45, all three
conditions:

| condition | `\|real−swap\|` | spread | sens/swap | policy KL |
|---|---|---|---|---|
| CartPole, recurrent actor | 0.003 | 0.025 | 0.11 | 0.355 |
| Acrobot, recurrent actor | 2.335 | 2.198 | 1.06 | 0.122 |
| **Acrobot, reactive actor** | **2.119** | 1.959 | **1.08** | **0.000** |

The reactive actor has **policy KL = 0** (verified: history literally cannot
change its output) — yet its recurrent value head uses history **just as much**
as the recurrent-actor case (`sens/swap` 1.08 vs 1.06). So the "policy drives
value history-use" hypothesis is **falsified**: the recurrent critic develops
strong history-dependence *even when the true value is Markov*.

> This also corrects the two earlier tables' `policy KL` readings: CartPole's
> non-zero policy KL (0.3–0.5) is real (its policy carries behaviourally-redundant
> memory), but it was never the cause of value history-use — the reactive-actor
> control shows value history-use appears with *zero* policy memory.

### 6e. …and the history-use is spurious: it does not improve accuracy
The clean accuracy test is **on-distribution** (`u = u' = ckpt 45`, so each head
is scored on its own states — no distribution shift to confound it). MC-corrected
MSE vs the true value:

| condition | recurrent `V_hat` | non-recurrent `V_hat_nr` | recurrent vs nonrec |
|---|---|---|---|
| Acrobot, recurrent actor | 99.38 | 99.41 | tied (−0.0%) |
| Acrobot, reactive actor | 93.10 | 90.74 | **+2.6% (worse)** |

Recurrence **never helps** once distribution shift is removed: it ties the
non-recurrent head with a recurrent policy, and is slightly *worse* with a
reactive one. Since the reactive policy's true value is Markov, the ~2-unit
history signal the critic learned is pure added variance — it can only hurt, and
it does.

> **Watch out for the distribution-shift trap.** In the off-policy comparison
> (`u`=ckpt5, `u'`=ckpt45), the reactive-actor recurrent head looked ~12% *better*
> than the non-recurrent one. That was an artifact of scoring on `pi_u`'s
> off-distribution states, not a real accuracy gain — the on-distribution test
> above (which the same script produces by setting `--ckpt-u == --ckpt-uprime`)
> removes it and reverses the sign. Always read the on-distribution number for an
> accuracy claim.

**Bottom line for "recurrence helps in fully-observable environments":** not
supported by these experiments. Recurrence is genuinely *engaged* on Acrobot (a
GRU value head builds up and uses history), but that engagement is **spurious** —
it is decoupled from whether the policy needs memory, and it does not improve (and
can slightly degrade) value accuracy. A likely mechanism is bootstrapping: TD/GAE
targets come from the critic's own history-conditioned predictions, so any initial
history-use is self-reinforcing regardless of whether the true value needs it.
Caveats: single seed, small network (hidden 64, `[64]` torsos), short training —
the *magnitude* of the effect is not robustly estimated, but the *direction*
(no accuracy benefit) is consistent across both actor conditions.

### 6f. A measurement bug worth recording
The first reactive-actor results were **invalid** and were re-run. The training
system zeroes the actor's hidden state every step, but the analysis scripts drove
the actor *recurrently* (letting its hidden state accumulate) — so they evaluated
a *different policy* than was trained. The tell was `policy KL = 0.46` for a
policy that is reactive by construction (KL must be exactly 0). The fix: the
analysis auto-detects a non-recurrent actor from the checkpoint's `system_name`
and resets the actor's hidden state every step during sampling, Monte-Carlo
rollouts, and the policy-KL ablation (`--model-name rec_ppo_nonrec_actor`).
Lesson: when an experiment forces an architectural constraint, assert the
invariant it implies (here, `policy KL == 0`) as a correctness check.

---

## 7. Reproduce end-to-end

```bash
# 1a. Train + checkpoint on CartPole (-> checkpoints/rec_ppo_dual_value/<timestamp>/).
./bash_scripts/run_rec_ppo_dual_value_cartpole.sh
# 1b. Train + checkpoint on Acrobot (-> checkpoints/rec_ppo_dual_value/acrobot_h64/).
./bash_scripts/run_rec_ppo_dual_value_acrobot.sh
# 1c. Control: reactive actor + recurrent value on Acrobot
#     (-> checkpoints/rec_ppo_nonrec_actor/acrobot_nonrec_actor_h64/).
./bash_scripts/run_nonrec_actor_acrobot.sh

# 2. Value drift (CartPole default u=ckpt3/u'=ckpt45; Acrobot needs --run-uid).
.venv/bin/python analysis/measure_value_drift.py
.venv/bin/python analysis/measure_value_drift.py --run-uid acrobot_h64 \
  --ckpt-u 48640 --ckpt-uprime 437760 --out analysis/value_drift_acrobot_ckpt5v45.json

# 2b. On-distribution accuracy (removes distribution shift). THIS is the number to
#     read for an accuracy claim; the off-policy MSE above is confounded by shift.
.venv/bin/python analysis/measure_value_drift.py --run-uid acrobot_h64 \
  --ckpt-u 437760 --ckpt-uprime 437760 --out analysis/value_drift_acrobot_ondist45.json

# 2c. The control variant (note --model-name). Auto-detects the reactive actor and
#     drives it correctly (hidden state zeroed every step).
.venv/bin/python analysis/measure_value_drift.py \
  --run-uid acrobot_nonrec_actor_h64 --model-name rec_ppo_nonrec_actor \
  --ckpt-u 437760 --ckpt-uprime 437760 \
  --out analysis/value_drift_nonrec_actor_ondist45.json

# 3. History ablation (does recurrence get used?).
.venv/bin/python analysis/measure_history_ablation.py --run-uid acrobot_h64
.venv/bin/python analysis/measure_history_ablation.py --run-uid 20260708154913
.venv/bin/python analysis/measure_history_ablation.py \
  --run-uid acrobot_nonrec_actor_h64 --model-name rec_ppo_nonrec_actor
```

To compare different checkpoints, pass their **step values** (index × 9,728),
e.g. `--ckpt-u 48640 --ckpt-uprime 486400` for ckpt 5 vs ckpt 50.
