# Learning Focus: Offline RL (IQL & AWAC) Without Full Environment Setup

HW3 is about **offline reinforcement learning**: learning from a fixed dataset without online exploration. The core skills are PyTorch (double Q-networks, expectile regression, advantage-weighted policy gradients) — not fighting `mujoco_py` or Modal on your laptop.

## What you implement

| File | Functions | Algorithm idea |
|------|-----------|----------------|
| `policies/MLP_policy.py` | `MLPPolicyAWAC.update` | AWR weights: `exp(λ · A(s,a))` |
| `critics/awac_critic.py` | `update` | Bellman backup with **policy** actions `a' ~ π(s')` |
| `agents/awac_agent.py` | `estimate_advantage`, `train` | `A = Q(s,a) − Q(s,a')`, `a' ~ π(s)` |
| `critics/iql_critic.py` | `v_net`, `expectile_loss`, `update_v`, `update_q` | V via expectile of Q; Q backup uses **V(s')** |
| `agents/iql_agent.py` | `estimate_advantage`, `train` | `A = Q(s,a) − V(s)`; train order: V → actor → Q |

Both agents reuse the same **advantage-weighted actor** (`MLPPolicyAWAC`).

## What you can skip locally

- **Modal** — cloud training for D4RL antmaze runs
- **D4RL / mujoco_py** — only needed for full `run_algo.py` rollouts
- **Pointmass gym env** — optional; dataset is in `offline_datasets/` if you later want plots

## Local workflow

```bash
cd hw3
conda activate cs224r          # or any env with torch + numpy
pip install -e .
pip install -r requirements-practice.txt   # optional; torch/numpy often enough
python test_without_env.py
```

The test script uses **synthetic batches** and a **mock replay buffer** — same tensor shapes as real training, no simulator.

## Conceptual map

**AWAC critic target:**  
`y = r + γ (1 − done) min(Q₁', Q₂')(s', a')` with `a' ~ π(·|s')`.

**IQL V-update:** regress `V(s)` toward `Q(s,a)` with **expectile** loss (over-estimate good actions).

**IQL Q-update:**  
`y = r + γ (1 − done) V(s')` (no max over actions — avoids OOD action bootstrap).

**Actor (both):** maximize `E[ exp(λ A) log π(a|s) ]` on **dataset** actions `a`.

## Suggested implementation order

1. `MLPPolicyAWAC.update` — one line (`exp_weights`)
2. `AWACCritic.update` — same pattern as HW2 double-Q TD
3. `AWACAgent` — sample `next_actions`, wire actor update
4. `IQLCritic.__init__` + `expectile_loss` + `update_v` + `update_q`
5. `IQLAgent` — advantage + actor (V and Q already in critic)

See `QUICK_START.md` for pseudocode and `GETTING_STARTED.md` for a longer walkthrough.

## When you need the full stack

See **`CS224R_2026_Homework_3.pdf`** for experiment commands and report tables. Submit runs via **Modal** (`modal_train.py` / `modal_train_para.py`) as in `README.md`. Your local implementations should match the same math; only data and env come from D4RL there.

**Experiments (from handout):**

- P1: AWAC on `antmaze-umaze-v0`, then `antmaze-medium-diverse-v0` (3 seeds each)
- P2.1: IQL with `--iql-expectile` 0.2 and 0.9 on umaze; submit best ζ to autograder
- P2.2: IQL medium maze with best ζ; compare vs AWAC in writeup
- P2.3: IQL vs filtered BC on `PointmassMedium-v0` + `offline_datasets/pointmass_stitching_dataset.npz`
