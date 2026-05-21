# Getting Started with HW3: Offline RL (IQL & AWAC)

Practice **PyTorch offline RL** on your Mac without Modal, D4RL, or MuJoCo. Full experiment runs stay on Modal when you are ready to submit.

## Prerequisites

```bash
cd hw3
conda activate cs224r    # existing HW1 env is fine if torch works
pip install -e .
python test_without_env.py
```

Optional minimal deps: `pip install -r requirements-practice.txt`

---

## Phase 1: Read the codebase (30–45 min)

### Algorithms


| Algo     | Critic backup                        | Advantage                  | Actor                     |
| -------- | ------------------------------------ | -------------------------- | ------------------------- |
| **AWAC** | `Q(s,a)` → `r + γ Q(s', a')`, `a'~π` | `Q(s,a) − Q(s,a')`, `a'~π` | AWR: `exp(λA) log π(a|s)` |
| **IQL**  | `Q` → `r + γ V(s')`                  | `Q(s,a) − V(s)`            | Same AWR actor            |


Both use **double Q** (min of two nets + target EMA) and the same `MLPPolicyAWAC` policy class.

### File tour

1. `**critics/awac_critic.py`** — `AWACCritic.update`: policy-action Bellman target.
2. `**critics/iql_critic.py**` — `v_net`, expectile V-fit, Q backup with V.
3. `**agents/awac_agent.py**` — wires critic + actor; samples `next_actions` for critic.
4. `**agents/iql_agent.py**` — V → actor → Q each step.
5. `**policies/MLP_policy.py**` — `MLPPolicyAWAC.update`: exponential advantage weights.
6. `**infrastructure/rl_trainer_awac.py**` — full loop (needs gym + D4RL); skip for now.

### Training flow (for reference)

**AWAC `train()`:**  
sample `a'~π(s')` → `critic.update(..., next_actions)` → `estimate_advantage` → `actor.update` → soft target update.

**IQL `train()`:**  
`critic.update_v` → advantage → `actor.update` → `critic.update_q` → soft target update.

---

## Phase 2: Implement in this order

See `**QUICK_START.md`** for copy-paste pseudocode.

1. `MLPPolicyAWAC.update` (easiest)
2. `AWACCritic.update`
3. `AWACAgent.estimate_advantage` + `train`
4. `IQLCritic` (`v_net`, `expectile_loss`, `update_v`, `update_q`)
5. `IQLAgent.estimate_advantage` + `train`

After each step:

```bash
python test_without_env.py
```

---

## Phase 3: What the test script checks

`test_without_env.py` builds mock agents with the same network factories as Pointmass (discrete) and AntMaze (continuous):

- Finite losses for critic / actor updates
- `expectile_loss` numeric check at ζ=0.9
- 10-step AWAC and IQL `train()` loops

It does **not** check learning quality — only that your code runs and backprops.

---

## Phase 4: Full experiments (optional)

When implementations pass locally:

```bash
# Course Modal setup — see README.md
modal run --detach modal_train.py --algo awac --env-name antmaze-umaze-v0 --exp-name my_run
```

Pointmass (discrete) can use a local `.npz`:

```bash
python cs224r/scripts/run_algo.py \
  --algo awac --env_name PointmassEasy-v0 \
  --offline_dataset offline_datasets/pointmass_stitching_dataset.npz \
  --exp_name local_pointmass --no_gpu
```

That path still needs `gym` and dependencies from `requirements.txt`.

---

## Handout

`**CS224R_2026_Homework_3.pdf**` (in this `hw3/` folder; due **5/8/2026**).


| Handout section                            | Code TODOs                                         | Local test                                    |
| ------------------------------------------ | -------------------------------------------------- | --------------------------------------------- |
| **Problem 1** — AWAC (AntMaze, continuous) | `MLP_policy.py`, `awac_critic.py`, `awac_agent.py` | `test_without_env.py` (discrete + continuous) |
| **Problem 2.1** — IQL + expectile ζ sweep  | `iql_critic.py`, `iql_agent.py`                    | same                                          |
| **Problem 2.3** — stitching (PointMass)    | same IQL code; run on Modal with `.npz`            | optional later                                |


**Deliverables (Modal + wandb):** Tables 1–6 in the PDF; autograder CSVs from **Eval_AverageReturn** only (`csv_data/P1/`, `csv_data/P2/`). See PDF “Autograder” section for exact filenames.

---

## Troubleshooting


| Issue                                      | Fix                                                    |
| ------------------------------------------ | ------------------------------------------------------ |
| `loss is None` / `TypeError` on `backward` | Fill in `### YOUR CODE ###` blocks                     |
| `v_net` AttributeError                     | Implement `IQLCritic.__init__` before other IQL tests  |
| `adv_n must be provided`                   | Call `actor.update(..., adv_n=...)` from agent `train` |
| Import errors for `gym`                    | Only needed for `run_algo.py`; tests don't import gym  |
| `expectile_loss` wrong                     | Use `                                                  |


More detail: `**LEARNING_FOCUS.md`**.