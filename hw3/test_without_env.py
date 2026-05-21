"""
Test HW3 offline RL implementations WITHOUT gym, D4RL, or Modal.

Covers:
  - MLPPolicyAWAC.update (exp_weights)
  - AWACCritic.update
  - IQLCritic (v_net, expectile_loss, update_v, update_q)
  - AWACAgent / IQLAgent (estimate_advantage, train)

Usage (from hw3/):
    pip install -e .
    python test_without_env.py
    python test_without_env.py --test awac_critic
    python test_without_env.py --test awac_critic_continuous
"""

import argparse
import sys
import traceback

import numpy as np
import torch

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure.utils import get_env_kwargs
from cs224r.agents.awac_agent import AWACAgent
from cs224r.agents.iql_agent import IQLAgent
from cs224r.critics.awac_critic import AWACCritic
from cs224r.critics.iql_critic import IQLCritic
from cs224r.policies.MLP_policy import MLPPolicyAWAC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockSpace:
    def __init__(self, shape):
        self.shape = shape


class _MockDiscrete:
    def __init__(self, n):
        self.n = n


class MockEnv:
    """Minimal env stub (agents only need it in __init__)."""

    def __init__(self, ob_dim, ac_dim, discrete=True):
        self.observation_space = _MockSpace((ob_dim,))
        if discrete:
            self.action_space = _MockDiscrete(ac_dim)
        else:
            self.action_space = _MockSpace((ac_dim,))


def _finite_scalar(x):
    if x is None:
        return False
    if hasattr(x, 'item'):
        x = x.item()
    return isinstance(x, (int, float)) and np.isfinite(x)


def _make_discrete_params(ob_dim=2, ac_dim=4, batch_size=32):
    base = get_env_kwargs('PointmassEasy-v0')
    return {
        **base,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': True,
        'batch_size': batch_size,
        'learning_freq': 1,
        'n_layers': 2,
        'size': 64,
        'learning_rate': 1e-3,
        'awac_lambda': 0.1,
        'iql_expectile': 0.9,
        'rew_shift': 0.0,
        'rew_scale': 1.0,
        'num_timesteps': 1000,
    }


def _make_continuous_params(ob_dim=8, ac_dim=4, batch_size=32):
    base = get_env_kwargs('antmaze-umaze-v0')
    return {
        **base,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': False,
        'continuous': True,
        'batch_size': batch_size,
        'learning_freq': 1,
        'n_layers': 2,
        'size': 64,
        'learning_rate': 1e-3,
        'awac_lambda': 0.1,
        'iql_expectile': 0.9,
        'rew_shift': 0.0,
        'rew_scale': 1.0,
        'num_timesteps': 1000,
    }


def _fill_replay_buffer(agent, n, ob_dim, ac_dim, discrete=True):
    buf = agent.replay_buffer
    obs = np.random.randn(n, ob_dim).astype(np.float32)
    if discrete:
        actions = np.random.randint(0, ac_dim, size=n, dtype=np.int32)
    else:
        actions = np.clip(np.random.randn(n, ac_dim), -1, 1).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    dones = np.zeros(n, dtype=bool)
    dones[np.arange(49, n, 50)] = True

    if buf.obs is None:
        buf.obs = np.empty([buf.size, ob_dim], dtype=np.float32)
        if discrete:
            buf.action = np.empty([buf.size], dtype=np.int32)
        else:
            buf.action = np.empty([buf.size, ac_dim], dtype=np.float32)
        buf.reward = np.empty([buf.size], dtype=np.float32)
        buf.done = np.empty([buf.size], dtype=bool)

    buf.obs[:n] = obs
    buf.action[:n] = actions
    buf.reward[:n] = rewards
    buf.done[:n] = dones
    buf.num_in_buffer = n
    buf.next_idx = n % buf.size


def _random_batch(ob_dim, ac_dim, batch_size, discrete=True):
    ob_no = np.random.randn(batch_size, ob_dim).astype(np.float32)
    if discrete:
        ac_na = np.random.randint(0, ac_dim, size=batch_size, dtype=np.int32)
    else:
        ac_na = np.clip(np.random.randn(batch_size, ac_dim), -1, 1).astype(np.float32)
    re_n = np.random.randn(batch_size).astype(np.float32)
    next_ob_no = np.random.randn(batch_size, ob_dim).astype(np.float32)
    terminal_n = (np.random.rand(batch_size) < 0.1).astype(np.float32)
    return ob_no, ac_na, re_n, next_ob_no, terminal_n


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_awac_policy_update():
    print("=" * 60)
    print("Testing MLPPolicyAWAC.update (exp_weights)")
    print("=" * 60)
    try:
        policy = MLPPolicyAWAC(
            ac_dim=4, ob_dim=2, n_layers=2, size=32,
            discrete=True, learning_rate=1e-3, lambda_awac=0.1,
        )
        obs = np.random.randn(16, 2).astype(np.float32)
        actions = np.random.randint(0, 4, size=16, dtype=np.int32)
        adv = np.random.randn(16).astype(np.float32)
        loss = policy.update(obs, actions, adv_n=adv)
        if not _finite_scalar(loss):
            print("❌ Actor loss must be a finite scalar, got {}".format(loss))
            return False
        print("✅ MLPPolicyAWAC.update works (loss={:.4f})".format(loss))
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_awac_critic_update(discrete=True):
    label = "discrete" if discrete else "continuous"
    print("\n" + "=" * 60)
    print("Testing AWACCritic.update ({})".format(label))
    print("=" * 60)
    ob_dim, ac_dim, batch_size = (2, 4, 32) if discrete else (8, 4, 32)
    try:
        params = _make_discrete_params(ob_dim, ac_dim) if discrete else _make_continuous_params(ob_dim, ac_dim)
        critic = AWACCritic(params, params['optimizer_spec'])
        ob_no, ac_na, re_n, next_ob_no, terminal_n = _random_batch(ob_dim, ac_dim, batch_size, discrete)

        env = MockEnv(ob_dim, ac_dim, discrete)
        agent = AWACAgent(env, {**params, 'discrete': discrete})
        with torch.no_grad():
            dist = agent.actor(ptu.from_numpy(next_ob_no))
            next_actions = dist.sample()
            if discrete:
                next_actions = next_actions.long()

        metrics = critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n, next_actions)
        for key in ('Training Loss', 'Training Loss2'):
            if key not in metrics or not _finite_scalar(metrics[key]):
                print("❌ Missing or invalid '{}': {}".format(key, metrics))
                return False
        print("✅ AWACCritic.update ({}) — {}".format(label, metrics))
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_iql_expectile_loss():
    print("\n" + "=" * 60)
    print("Testing IQLCritic.expectile_loss")
    print("=" * 60)
    try:
        params = _make_discrete_params()
        critic = IQLCritic(params, params['optimizer_spec'])
        if not hasattr(critic, 'v_net'):
            print("❌ self.v_net not defined in IQLCritic.__init__")
            return False

        diff = torch.tensor([1.0, -1.0], device=ptu.device)
        loss = critic.expectile_loss(diff)
        if loss is None or (hasattr(loss, 'shape') and loss.shape != diff.shape):
            print("❌ expectile_loss must return per-sample tensor, got {}".format(loss))
            return False
        zeta = params['iql_expectile']
        expected = torch.tensor([zeta * 1.0, (1 - zeta) * 1.0], device=ptu.device)
        if not torch.allclose(loss, expected, atol=1e-5):
            print("❌ expectile_loss values wrong for ζ={}".format(zeta))
            print("   got {}, expected {}".format(loss, expected))
            return False
        print("✅ expectile_loss matches asymmetric formula")
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_iql_update_v():
    print("\n" + "=" * 60)
    print("Testing IQLCritic.update_v")
    print("=" * 60)
    try:
        params = _make_discrete_params()
        critic = IQLCritic(params, params['optimizer_spec'])
        ob_no, ac_na, _, _, _ = _random_batch(2, 4, 32, discrete=True)
        metrics = critic.update_v(ob_no, ac_na)
        if 'Training V Loss' not in metrics or not _finite_scalar(metrics['Training V Loss']):
            print("❌ Invalid V loss: {}".format(metrics))
            return False
        print("✅ IQLCritic.update_v — {}".format(metrics))
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_iql_update_q():
    print("\n" + "=" * 60)
    print("Testing IQLCritic.update_q")
    print("=" * 60)
    try:
        params = _make_discrete_params()
        critic = IQLCritic(params, params['optimizer_spec'])
        ob_no, ac_na, re_n, next_ob_no, terminal_n = _random_batch(2, 4, 32, discrete=True)
        metrics = critic.update_q(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        for key in ('Training Q Loss', 'Training Q Loss2'):
            if key not in metrics or not _finite_scalar(metrics[key]):
                print("❌ Missing or invalid '{}': {}".format(key, metrics))
                return False
        print("✅ IQLCritic.update_q — {}".format(metrics))
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_awac_agent_train(discrete=True):
    label = "discrete" if discrete else "continuous"
    print("\n" + "=" * 60)
    print("Testing AWACAgent.train ({})".format(label))
    print("=" * 60)
    ob_dim, ac_dim = (2, 4) if discrete else (8, 4)
    try:
        params = _make_discrete_params(ob_dim, ac_dim) if discrete else _make_continuous_params(ob_dim, ac_dim)
        env = MockEnv(ob_dim, ac_dim, discrete)
        agent = AWACAgent(env, {**params, 'discrete': discrete})
        _fill_replay_buffer(agent, 500, ob_dim, ac_dim, discrete)
        agent.t = 0

        ob_no, ac_na, re_n, next_ob_no, terminal_n = _random_batch(
            ob_dim, ac_dim, params['batch_size'], discrete)
        log = agent.train(ob_no, ac_na, re_n, next_ob_no, terminal_n)
        if not log:
            print("❌ train() returned empty log (buffer not ready or train body not implemented)")
            return False
        for key in ('Critic Loss', 'Actor Loss'):
            if key not in log or not _finite_scalar(log[key]):
                print("❌ Missing or invalid '{}': {}".format(key, log))
                return False
        print("✅ AWACAgent.train ({}) — {}".format(label, log))
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_iql_agent_train():
    print("\n" + "=" * 60)
    print("Testing IQLAgent.train (discrete)")
    print("=" * 60)
    ob_dim, ac_dim = 2, 4
    try:
        params = _make_discrete_params(ob_dim, ac_dim)
        env = MockEnv(ob_dim, ac_dim, discrete=True)
        agent = IQLAgent(env, params)
        _fill_replay_buffer(agent, 500, ob_dim, ac_dim, discrete=True)
        agent.t = 0

        ob_no, ac_na, re_n, next_ob_no, terminal_n = _random_batch(
            ob_dim, ac_dim, params['batch_size'], discrete=True)
        log = agent.train(ob_no, ac_na, re_n, next_ob_no, terminal_n)
        if not log:
            print("❌ train() returned empty log")
            return False
        for key in ('Critic V Loss', 'Critic Q Loss', 'Actor Loss'):
            if key not in log or not _finite_scalar(log[key]):
                print("❌ Missing or invalid '{}': {}".format(key, log))
                return False
        print("✅ IQLAgent.train — {}".format(log))
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


def test_multiple_train_steps():
    print("\n" + "=" * 60)
    print("Testing 10 AWAC + 10 IQL train steps")
    print("=" * 60)
    try:
        for AgentCls, name in [(AWACAgent, 'AWAC'), (IQLAgent, 'IQL')]:
            params = _make_discrete_params()
            env = MockEnv(2, 4, discrete=True)
            agent = AgentCls(env, params)
            _fill_replay_buffer(agent, 500, 2, 4, discrete=True)
            agent.t = 0
            for step in range(10):
                batch = _random_batch(2, 4, params['batch_size'], discrete=True)
                log = agent.train(*batch)
                if not log:
                    print("❌ {} step {}: empty log".format(name, step))
                    return False
            print("   {}: 10 steps OK".format(name))
        print("✅ Multiple train steps completed")
        return True
    except Exception as e:
        print("❌ {}".format(e))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TEST_REGISTRY = {
    'awac_policy': ('MLPPolicyAWAC.update', test_awac_policy_update),
    'awac_critic': ('AWACCritic.update (discrete)', lambda: test_awac_critic_update(True)),
    'awac_critic_continuous': (
        'AWACCritic.update (continuous)', lambda: test_awac_critic_update(False)),
    'iql_expectile': ('IQL expectile_loss', test_iql_expectile_loss),
    'iql_update_v': ('IQLCritic.update_v', test_iql_update_v),
    'iql_update_q': ('IQLCritic.update_q', test_iql_update_q),
    'awac_agent': ('AWACAgent.train (discrete)', lambda: test_awac_agent_train(True)),
    'awac_agent_continuous': (
        'AWACAgent.train (continuous)', lambda: test_awac_agent_train(False)),
    'iql_agent': ('IQLAgent.train', test_iql_agent_train),
    'multi_step': ('Multiple train steps', test_multiple_train_steps),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW3 local tests (no gym/D4RL)')
    parser.add_argument(
        '--test', '-t',
        choices=list(TEST_REGISTRY.keys()) + ['all'],
        default='all',
        help='Run one test (default: all). Use awac_critic for AWACCritic.update.',
    )
    args = parser.parse_args()

    ptu.init_gpu(use_gpu=False)

    print("\n" + "=" * 60)
    print("HW3 Offline RL — tests without gym / D4RL / Modal")
    print("=" * 60)
    print("\nInstall once:  cd hw3 && pip install -e .\n")

    if args.test == 'all':
        tests = list(TEST_REGISTRY.values())
    else:
        tests = [TEST_REGISTRY[args.test]]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print("❌ {} crashed: {}".format(name, e))
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        print("{:40s} {}".format(name, "✅ PASS" if passed else "❌ FAIL"))

    if all(p for _, p in results):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Implement the ### YOUR CODE ### blocks, then re-run.")
        sys.exit(1)
