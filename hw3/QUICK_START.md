# HW3 Quick Start — implement & test locally

## One-time setup (~2 min)

```bash
cd /Users/ruofanwang/course/CS224R_2026Winter/hw3
conda activate cs224r          # or: conda create -n cs224r-hw3-practice python=3.10
pip install -e .
python -c "import torch; print('torch', torch.__version__)"
```

Only **torch** and **numpy** are required for `test_without_env.py`.

## Run tests after each change

```bash
python test_without_env.py
```

## Implementation order + pseudocode

### 1. `MLPPolicyAWAC.update` — `policies/MLP_policy.py`

```python
exp_weights = torch.exp(self.lambda_awac * adv_n)
```

### 2. `AWACCritic.update` — `critics/awac_critic.py`

```python
q1 = self._get_q_value(self.q_net, ob_no, ac_na)
q2 = self._get_q_value(self.q_net2, ob_no, ac_na)
with torch.no_grad():
    target_q = self.get_target_q(next_ob_no, next_actions)
    y = reward_n + self.gamma * (1.0 - terminal_n) * target_q
loss = self.mse_loss(q1, y)
loss2 = self.mse_loss(q2, y)
```

### 3. `AWACAgent` — `agents/awac_agent.py`

**`estimate_advantage`:**

```python
q_sa = self.critic.get_q(ob_no, ac_na)
dist = self.actor(ob_no)
a_pi = dist.sample()
if self.discrete_action:
    a_pi = a_pi.long()
v_s = self.critic.get_q(ob_no, a_pi)
adv = q_sa - v_s
```

**`train` — sample next actions + actor:**

```python
with torch.no_grad():
    dist = self.actor(ptu.from_numpy(next_ob_no))
    next_actions = dist.sample()
    if self.discrete_action:
        next_actions = next_actions.long()

adv = self.estimate_advantage(ob_no, ac_na)
actor_loss = self.actor.update(ob_no, ac_na, adv_n=ptu.to_numpy(adv))
```

### 4. `IQLCritic` — `critics/iql_critic.py`

**`__init__`:**

```python
self.v_net = v_network_initializer(self.ob_dim)
self.v_net.to(ptu.device)
```

**`expectile_loss(diff)`** with `ζ = self.iql_expectile`:

```python
weight = torch.where(diff > 0, zeta, 1.0 - zeta)
return weight * (diff ** 2)
```

**`update_v`:**

```python
value_loss = self.expectile_loss(q_t_values - v_t).mean()
```

**`update_q`:**

```python
q1 = self._get_q_value(self.q_net, ob_no, ac_na)
q2 = self._get_q_value(self.q_net2, ob_no, ac_na)
with torch.no_grad():
    v_next = self.v_net(next_ob_no).squeeze(-1)
    y = reward_n + self.gamma * (1.0 - terminal_n) * v_next
loss = self.mse_loss(q1, y)
loss2 = self.mse_loss(q2, y)
```

### 5. `IQLAgent` — `agents/iql_agent.py`

**`estimate_advantage`:**

```python
adv = q_sa - v_pi   # v_pi and q_sa already computed above
```

**`train` — actor block (after `update_v`):**

```python
adv = self.estimate_advantage(ob_no, ac_na)
actor_loss = self.actor.update(ob_no, ac_na, adv_n=ptu.to_numpy(adv))
```

## Files with `### YOUR CODE ###`

- `cs224r/policies/MLP_policy.py`
- `cs224r/critics/awac_critic.py`
- `cs224r/critics/iql_critic.py`
- `cs224r/agents/awac_agent.py`
- `cs224r/agents/iql_agent.py`

## Full training (later)

Commands and tables are in **`CS224R_2026_Homework_3.pdf`**. Example (P1.1, single seed):

```bash
modal run --detach modal_train.py --algo awac \
  --env-name antmaze-umaze-v0 \
  --exp-name awac_antmaze_umaze --use-wandb --seed 1
```

Not needed to learn the PyTorch pieces locally.
