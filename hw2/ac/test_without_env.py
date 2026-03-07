"""
Test script for Actor-Critic implementation WITHOUT needing MetaWorld environment.
This lets you implement and test update_critic, update_actor, and bc functions
using synthetic data.

Usage:
    python test_without_env.py
"""

import torch
import torch.nn as nn
import numpy as np
from ac import ACAgent

# Create a simple mock replay buffer iterator
class MockReplayIter:
    def __init__(self, batch_size=32, obs_dim=39, action_dim=4):
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Generate synthetic batch data
        obs = np.random.randn(self.batch_size, self.obs_dim).astype(np.float32)
        action = np.random.randn(self.batch_size, self.action_dim).astype(np.float32)
        reward = np.random.randn(self.batch_size).astype(np.float32)
        discount = np.ones(self.batch_size, dtype=np.float32) * 0.99
        next_obs = np.random.randn(self.batch_size, self.obs_dim).astype(np.float32)
        
        return (obs, action, reward, discount, next_obs)


def _metric_float(m, key):
    """Get metric as Python float (handles tensors)."""
    if key not in m:
        return None
    v = m[key]
    return v.item() if hasattr(v, 'item') else float(v)


def test_bc():
    """Test behavior cloning function"""
    print("=" * 60)
    print("Testing BC (Behavior Cloning)")
    print("=" * 60)
    
    # Create agent
    obs_shape = (39,)
    action_shape = (4,)
    agent = ACAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device='cpu',
        lr=1e-4,
        hidden_dim=256,
        num_critics=2,
        critic_target_tau=0.005,
        stddev_clip=0.3,
        use_tb=False
    )
    
    # Create mock data
    replay_iter = MockReplayIter(batch_size=32, obs_dim=39, action_dim=4)
    
    # Test BC function
    try:
        metrics = agent.bc(replay_iter)
        if 'bc_loss' not in metrics:
            print("❌ BC must return metrics with key 'bc_loss' (you returned {})".format(metrics))
            return False
        loss = metrics['bc_loss']
        if not isinstance(loss, (int, float)) or not np.isfinite(loss):
            print("❌ bc_loss must be a finite number, got {}".format(loss))
            return False
        print("✅ BC function implemented!")
        print(f"   Metrics: {metrics}")
        return True
    except NotImplementedError:
        print("❌ BC function not implemented yet")
        return False
    except Exception as e:
        print(f"❌ BC function error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_update_critic():
    """Test critic update function"""
    print("\n" + "=" * 60)
    print("Testing update_critic")
    print("=" * 60)
    
    # Create agent
    obs_shape = (39,)
    action_shape = (4,)
    agent = ACAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device='cpu',
        lr=1e-4,
        hidden_dim=256,
        num_critics=2,
        critic_target_tau=0.005,
        stddev_clip=0.3,
        use_tb=False
    )
    
    # Create mock data
    replay_iter = MockReplayIter(batch_size=32, obs_dim=39, action_dim=4)
    
    # Test critic update
    try:
        metrics = agent.update_critic(replay_iter)
        if 'critic_loss' not in metrics:
            print("❌ update_critic must return metrics with key 'critic_loss' (you returned {})".format(metrics))
            return False
        loss = metrics['critic_loss']
        if not isinstance(loss, (int, float)) or not np.isfinite(loss):
            print("❌ critic_loss must be a finite number, got {}".format(loss))
            return False
        print("✅ update_critic function implemented!")
        print(f"   Metrics: {metrics}")
        return True
    except NotImplementedError:
        print("❌ update_critic function not implemented yet")
        return False
    except Exception as e:
        print(f"❌ update_critic function error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_update_actor():
    """Test actor update function"""
    print("\n" + "=" * 60)
    print("Testing update_actor")
    print("=" * 60)
    
    # Create agent
    obs_shape = (39,)
    action_shape = (4,)
    agent = ACAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device='cpu',
        lr=1e-4,
        hidden_dim=256,
        num_critics=2,
        critic_target_tau=0.005,
        stddev_clip=0.3,
        use_tb=False
    )
    
    # Create mock data
    replay_iter = MockReplayIter(batch_size=32, obs_dim=39, action_dim=4)
    
    # Test actor update
    try:
        metrics = agent.update_actor(replay_iter)
        if 'actor_loss' not in metrics:
            print("❌ update_actor must return metrics with key 'actor_loss' (you returned {})".format(metrics))
            return False
        loss = metrics['actor_loss']
        if not isinstance(loss, (int, float)) or not np.isfinite(loss):
            print("❌ actor_loss must be a finite number, got {}".format(loss))
            return False
        print("✅ update_actor function implemented!")
        print(f"   Metrics: {metrics}")
        return True
    except NotImplementedError:
        print("❌ update_actor function not implemented yet")
        return False
    except Exception as e:
        print(f"❌ update_actor function error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_updates():
    """Test that multiple updates work correctly"""
    print("\n" + "=" * 60)
    print("Testing Multiple Updates")
    print("=" * 60)
    
    obs_shape = (39,)
    action_shape = (4,)
    agent = ACAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device='cpu',
        lr=1e-6,
        hidden_dim=256,
        num_critics=2,
        critic_target_tau=0.005,
        stddev_clip=0.3,
        use_tb=False
    )
    
    replay_iter = MockReplayIter(batch_size=32, obs_dim=39, action_dim=4)
    
    try:
        # Run multiple updates
        for i in range(500):
            critic_metrics = agent.update_critic(replay_iter)
            actor_metrics = agent.update_actor(replay_iter)
            bc_metrics = agent.bc(replay_iter)
            # Require expected metrics (same as individual tests)
            for name, key in [('critic', 'critic_loss'), ('actor', 'actor_loss'), ('BC', 'bc_loss')]:
                m = critic_metrics if name == 'critic' else (actor_metrics if name == 'actor' else bc_metrics)
                val = _metric_float(m, key)
                if val is None or not np.isfinite(val):
                    print("❌ Multiple updates: {} metric '{}' missing or invalid".format(name, key))
                    return False
            if i % 10 == 0:
                print(f"   Step {i+1}: Critic loss: {_metric_float(critic_metrics, 'critic_loss'):.4f}, "
                      f"Actor loss: {_metric_float(actor_metrics, 'actor_loss'):.4f}, "
                      f"BC loss: {_metric_float(bc_metrics, 'bc_loss'):.4f}")
        print("✅ Multiple updates completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error during multiple updates: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_steps(num_steps=20):
    """Run a few steps with mock data, record metrics, check all finite."""
    print("\n" + "=" * 60)
    print("Testing Training ({} steps, mock data)".format(num_steps))
    print("=" * 60)
    
    agent = ACAgent(
        obs_shape=(39,),
        action_shape=(4,),
        device='cpu',
        lr=1e-3,
        hidden_dim=256,
        num_critics=2,
        critic_target_tau=0.005,
        stddev_clip=0.3,
        use_tb=False,
    )
    replay = MockReplayIter(batch_size=32, obs_dim=39, action_dim=4)
    recorded = []
    
    try:
        for step in range(num_steps):
            cm = agent.update_critic(replay)
            am = agent.update_actor(replay)
            bm = agent.bc(replay)
            c = _metric_float(cm, 'critic_loss')
            a = _metric_float(am, 'actor_loss')
            b = _metric_float(bm, 'bc_loss')
            if c is None or a is None or b is None:
                print("❌ Missing metrics at step {}".format(step))
                return False
            if not (np.isfinite(c) and np.isfinite(a) and np.isfinite(b)):
                print("❌ Non-finite loss at step {}: critic={}, actor={}, bc={}".format(step, c, a, b))
                return False
            recorded.append((c, a, b))
        print("   Step 0:  critic={:.4f}, actor={:.4f}, bc={:.4f}".format(*recorded[0]))
        print("   Step {}: critic={:.4f}, actor={:.4f}, bc={:.4f}".format(num_steps - 1, *recorded[-1]))
        print("✅ Training ({} steps): all metrics finite.".format(num_steps))
        return True
    except Exception as e:
        print("❌ Error: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Actor-Critic Implementation Test (No Environment Required)")
    print("=" * 60)
    print("\nThis script tests your implementations using synthetic data.")
    print("You don't need MetaWorld or mujoco_py to run this!\n")
    
    results = []
    results.append(("BC", test_bc()))
    results.append(("update_critic", test_update_critic()))
    results.append(("update_actor", test_update_actor()))
    results.append(("Multiple Updates", test_multiple_updates()))
    results.append(("Training (20 steps)", test_training_steps()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Your implementation looks good!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

