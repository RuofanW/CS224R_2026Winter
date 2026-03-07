# Getting Started with HW2: Actor-Critic Implementation

## Prerequisites

You only need **PyTorch** and **NumPy** for the coding exercise (no MuJoCo or MetaWorld).

```bash
conda activate cs224r   # or your env
python -c "import torch; print(torch.__version__)"
# If needed: pip install torch torchvision
```

Test without environment: from `hw2/ac`, run `python test_without_env.py`.

---

## Phase 1: Understand the Codebase (30–45 min)

### 1. Read the architecture — `ac.py`

1. **Actor (lines 11–30)**  
   Policy network; `forward()` returns a `TruncatedNormal` distribution (similar to HW1’s `MLPPolicySL`).

2. **Critic (lines 33–48)**  
   Q(s,a); input is concatenated `(obs, action)`; `num_critics` gives multiple Q-networks for stability.

3. **ACAgent (lines 51–89)**  
   Combines `actor`, `critic`, `critic_target`, and optimizers. You will implement three methods.

### 2. Training loop — `train.py` (method `train()`, ~lines 137–208)

- Pretrain with BC: `agent.bc()`
- Main loop: sample action → `update_critic()` (multiple times) → `update_actor()` → periodic `agent.bc()`

### 3. Helpers — `utils.py`

- `soft_update_params(net, target, tau)` — target = τ·source + (1−τ)·target
- `to_torch(batch, device)` — numpy → torch
- `TruncatedNormal` — bounded action distribution (like Normal but clamped to [-1, 1])

### 4. What you implement — `ac.py`

1. **`bc()`** (easiest) — behavior cloning  
2. **`update_critic()`** — Q-learning with target network  
3. **`update_actor()`** — policy gradient

---

## Phase 2: Implementation Order

### Task 1: Implement `bc()` (easiest)

Behavior cloning: match expert actions with negative log-likelihood.

**Full implementation:**

```python
def bc(self, replay_iter):
    metrics = dict()
    batch = next(replay_iter)
    obs, action, _, _, _ = utils.to_torch(batch, self.device)

    dist = self.actor(obs)
    log_probs = dist.log_prob(action)  # (batch, action_dim)
    loss = -log_probs.mean()

    self.actor_opt.zero_grad()
    loss.backward()
    self.actor_opt.step()

    metrics['bc_loss'] = loss.item()
    return metrics
```

**Test:** `cd hw2/ac && python test_without_env.py` — look for “✅ BC function implemented!”

---

### Task 2: Implement `update_critic()`

- **TD target:** r + γ · Q_target(s′, a′)  
- Use **critic_target** and **torch.no_grad()** for targets.  
- If `num_critics` > 1, use **min** over target Q-values.

**Full implementation:**

```python
def update_critic(self, replay_iter):
    metrics = dict()
    batch = next(replay_iter)
    obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

    current_qs = self.critic(obs, action)  # list of [batch, 1]

    with torch.no_grad():
        next_dist = self.actor(next_obs)
        next_actions = next_dist.sample()
        target_qs = self.critic_target(next_obs, next_actions)

    target_q = min(target_qs)
    target = reward + discount * target_q.squeeze()  # (batch,)

    loss = 0
    for current_q in current_qs:
        loss += F.mse_loss(current_q.squeeze(), target)
    loss = loss / len(current_qs)

    self.critic_opt.zero_grad()
    loss.backward()
    self.critic_opt.step()

    utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    metrics['critic_loss'] = loss.item()
    metrics['target_q_mean'] = target.mean().item()
    return metrics
```

**Test:** same script — “✅ update_critic function implemented!”

---

### Task 3: Implement `update_actor()`

Maximize Q(s, π(s)); critic gives Q-values.

**Full implementation:**

```python
def update_actor(self, replay_iter):
    metrics = dict()
    batch = next(replay_iter)
    obs, _, _, _, _ = utils.to_torch(batch, self.device)

    actions = self.actor(obs).sample()  # (batch, action_dim)
    q_values = self.critic(obs, actions)  # list of [batch, 1]
    q_value = min(q_values)

    loss = -q_value.mean()

    self.actor_opt.zero_grad()
    loss.backward()
    self.actor_opt.step()

    metrics['actor_loss'] = loss.item()
    metrics['q_value_mean'] = q_value.mean().item()
    return metrics
```

**Test:** `python test_without_env.py` — all tests should pass.

---

## Phase 3: Testing (no environment)

```bash
cd hw2/ac
python test_without_env.py
```

- ✅ = implemented and working  
- ❌ = not implemented or error  
Uses synthetic data only; no MetaWorld/MuJoCo needed.

---

## Phase 4: Debugging & reference

### Common pitfalls

1. Use **torch.no_grad()** when computing critic targets.  
2. Fix **shape mismatches** with `squeeze()` / `unsqueeze()`.  
3. With multiple critics, take **min** over Q-values.  
4. Use **critic_target** for targets, not `critic`.  
5. Call **soft_update_params()** after each critic update.

### Quick checks

```python
print(f"obs shape: {obs.shape}, action shape: {action.shape}")
# Gradients:
for name, param in self.actor.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### Quick reference

| utils.py | ac.py |
|----------|--------|
| `to_torch(batch, device)` | `self.actor(obs)` → TruncatedNormal |
| `soft_update_params(net, target, tau)` | `self.critic(obs, action)` → list of Qs |
| `TruncatedNormal` | `dist.sample()`, `dist.log_prob(a)`, `dist.mean` |

---

## Checklist & time

- [ ] Read `ac.py`, `train.py`, `utils.py`  
- [ ] Implement `bc()` and test  
- [ ] Implement `update_critic()` and test  
- [ ] Implement `update_actor()` and test  
- [ ] All tests in `test_without_env.py` pass  

**Rough total:** 2–4 hours (read 30–45 min, implement 2–3 h).

Once all tests pass, you’re done with the core implementation. To run in a real environment later, see `CLOUD_ALTERNATIVES.md` for setup.
