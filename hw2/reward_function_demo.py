"""
Demonstration of Reward Function Impact
========================================

This script demonstrates how different reward functions affect learning,
even without running actual RL experiments. It simulates the learning
process to show the conceptual differences.

Run: python reward_function_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable


class SimpleEnvironment:
    """
    Simple 1D environment: agent starts at position 0, goal is at position 10.
    Agent can move left (-1) or right (+1).
    """
    def __init__(self):
        self.position = 0
        self.goal = 10
        self.max_steps = 100
        
    def reset(self):
        self.position = 0
        return self.position
    
    def step(self, action):
        """Action: -1 (left) or +1 (right)"""
        self.position = max(0, min(20, self.position + action))
        done = (self.position >= self.goal)
        return self.position, done


def sparse_reward(env: SimpleEnvironment) -> float:
    """Sparse: reward only when goal reached"""
    if env.position >= env.goal:
        return 1.0
    return 0.0


def dense_reward(env: SimpleEnvironment) -> float:
    """Dense: reward based on distance to goal"""
    distance = abs(env.position - env.goal)
    max_distance = env.goal
    return 1.0 - (distance / max_distance)  # Closer = higher reward


def shaped_reward(env: SimpleEnvironment) -> float:
    """Shaped: combines distance with progress bonus"""
    distance = abs(env.position - env.goal)
    max_distance = env.goal
    
    # Base reward from distance
    base_reward = 1.0 - (distance / max_distance)
    
    # Bonus for making progress (if moving toward goal)
    progress_bonus = 0.1 if env.position > 0 else 0.0
    
    # Milestone bonus (closer to goal)
    if env.position >= env.goal * 0.8:
        milestone_bonus = 0.2
    elif env.position >= env.goal * 0.5:
        milestone_bonus = 0.1
    else:
        milestone_bonus = 0.0
    
    return base_reward + progress_bonus + milestone_bonus


def simulate_learning(
    reward_fn: Callable,
    num_episodes: int = 100,
    exploration_rate: float = 0.3
) -> Tuple[List[float], List[int]]:
    """
    Simulate learning with a given reward function.
    
    Uses a simple policy: 
    - With probability exploration_rate: random action
    - Otherwise: move toward goal (if we know where it is)
    
    Returns: (rewards_per_episode, steps_to_goal_per_episode)
    """
    env = SimpleEnvironment()
    rewards = []
    steps_to_goal = []
    
    # Simple Q-learning style: learn which direction is better
    q_values = {0: {'left': 0.0, 'right': 0.0}}  # Q(state, action)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(env.max_steps):
            # Choose action (epsilon-greedy)
            if np.random.random() < exploration_rate:
                action = np.random.choice([-1, 1])  # Random
            else:
                # Greedy: choose action with higher Q-value
                if state not in q_values:
                    q_values[state] = {'left': 0.0, 'right': 0.0}
                
                if q_values[state]['right'] > q_values[state]['left']:
                    action = 1
                else:
                    action = -1
            
            # Take step
            next_state, done = env.step(action)
            reward = reward_fn(env)
            episode_reward += reward
            
            # Update Q-value (simple Q-learning update)
            if state not in q_values:
                q_values[state] = {'left': 0.0, 'right': 0.0}
            if next_state not in q_values:
                q_values[next_state] = {'left': 0.0, 'right': 0.0}
            
            action_key = 'right' if action == 1 else 'left'
            learning_rate = 0.1
            discount = 0.9
            
            # Q-learning update
            best_next_q = max(q_values[next_state]['left'], 
                             q_values[next_state]['right'])
            q_values[state][action_key] = (
                (1 - learning_rate) * q_values[state][action_key] +
                learning_rate * (reward + discount * best_next_q)
            )
            
            state = next_state
            steps += 1
            
            if done:
                steps_to_goal.append(steps)
                break
        else:
            # Episode didn't finish
            steps_to_goal.append(env.max_steps)
        
        rewards.append(episode_reward)
        exploration_rate *= 0.995  # Decay exploration
    
    return rewards, steps_to_goal


def plot_comparison():
    """Compare learning with different reward functions"""
    print("Simulating learning with different reward functions...")
    print("This may take a moment...\n")
    
    # Run simulations
    sparse_rewards, sparse_steps = simulate_learning(sparse_reward, num_episodes=200)
    dense_rewards, dense_steps = simulate_learning(dense_reward, num_episodes=200)
    shaped_rewards, shaped_steps = simulate_learning(shaped_reward, num_episodes=200)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Cumulative rewards
    axes[0].plot(np.cumsum(sparse_rewards), label='Sparse Reward', alpha=0.7)
    axes[0].plot(np.cumsum(dense_rewards), label='Dense Reward', alpha=0.7)
    axes[0].plot(np.cumsum(shaped_rewards), label='Shaped Reward', alpha=0.7)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Reward Over Episodes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Steps to goal (smoothed)
    window = 10
    sparse_smooth = np.convolve(sparse_steps, np.ones(window)/window, mode='valid')
    dense_smooth = np.convolve(dense_steps, np.ones(window)/window, mode='valid')
    shaped_smooth = np.convolve(shaped_steps, np.ones(window)/window, mode='valid')
    
    axes[1].plot(sparse_smooth, label='Sparse Reward', alpha=0.7)
    axes[1].plot(dense_smooth, label='Dense Reward', alpha=0.7)
    axes[1].plot(shaped_smooth, label='Shaped Reward', alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal (smoothed)')
    axes[1].set_title('Learning Progress: Steps to Reach Goal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()  # Lower is better
    
    plt.tight_layout()
    plt.savefig('reward_function_comparison.png', dpi=150)
    print("✅ Plot saved as 'reward_function_comparison.png'")
    
    # Print statistics
    print("\n" + "="*60)
    print("LEARNING STATISTICS")
    print("="*60)
    
    print(f"\n📊 Sparse Reward:")
    print(f"   Average steps to goal (last 50 episodes): {np.mean(sparse_steps[-50:]):.1f}")
    print(f"   Episodes to first success: {next((i for i, s in enumerate(sparse_steps) if s < 100), 'Never'):}")
    print(f"   Total reward (first 50 episodes): {np.sum(sparse_rewards[:50]):.1f}")
    
    print(f"\n📊 Dense Reward:")
    print(f"   Average steps to goal (last 50 episodes): {np.mean(dense_steps[-50:]):.1f}")
    print(f"   Episodes to first success: {next((i for i, s in enumerate(dense_steps) if s < 100), 'Never'):}")
    print(f"   Total reward (first 50 episodes): {np.sum(dense_rewards[:50]):.1f}")
    
    print(f"\n📊 Shaped Reward:")
    print(f"   Average steps to goal (last 50 episodes): {np.mean(shaped_steps[-50:]):.1f}")
    print(f"   Episodes to first success: {next((i for i, s in enumerate(shaped_steps) if s < 100), 'Never'):}")
    print(f"   Total reward (first 50 episodes): {np.sum(shaped_rewards[:50]):.1f}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("1. Sparse rewards: Slow to learn, but may eventually find optimal solution")
    print("2. Dense rewards: Faster learning, continuous feedback")
    print("3. Shaped rewards: Fastest learning, guided by domain knowledge")
    print("\n💡 This demonstrates why reward function design is crucial in RL!")


if __name__ == "__main__":
    print("="*60)
    print("Reward Function Impact Demonstration")
    print("="*60)
    print("\nThis script simulates how different reward functions")
    print("affect learning in a simple 1D navigation task.")
    print("\nTask: Agent starts at position 0, goal is at position 10.")
    print("Agent can move left (-1) or right (+1).\n")
    
    try:
        plot_comparison()
    except ImportError:
        print("❌ Error: matplotlib not installed.")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nYou can still read REWARD_FUNCTIONS_GUIDE.md for conceptual understanding!")

