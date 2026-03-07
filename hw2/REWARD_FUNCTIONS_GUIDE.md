# Understanding Reward Functions: Impact on RL Learning

## Overview

Problem 1 explores how different reward function designs affect reinforcement learning performance. This is a fundamental concept in RL: **the reward function shapes what the agent learns**.

---

## 1. Types of Reward Functions

### Sparse Rewards

**Definition**: Reward is only given at specific milestones (usually at task completion).

**Example**:
```python
# Sparse reward for reaching goal
if agent_reached_goal:
    reward = 1.0
else:
    reward = 0.0
```

**Characteristics**:
- ✅ Simple to design
- ✅ Directly encodes task objective
- ❌ **Very hard to learn from** (credit assignment problem)
- ❌ Agent gets no feedback until it accidentally succeeds
- ❌ Requires many random explorations before learning

**When to use**: When the task is well-defined and you have enough exploration budget.

---

### Dense Rewards

**Definition**: Reward is given at every step based on progress toward the goal.

**Example**:
```python
# Dense reward: distance to goal
distance_to_goal = np.linalg.norm(agent_pos - goal_pos)
reward = -distance_to_goal  # Closer = higher reward
```

**Characteristics**:
- ✅ Provides learning signal at every step
- ✅ Faster learning (better credit assignment)
- ✅ Easier for agent to understand what to do
- ❌ Can lead to **reward hacking** (agent finds shortcuts)
- ❌ Requires careful design to align with true objective

**When to use**: When you can design meaningful intermediate rewards.

---

### Shaped Rewards

**Definition**: Dense rewards that guide the agent toward the goal using domain knowledge.

**Example**:
```python
# Shaped reward for robot reaching object
distance_to_object = np.linalg.norm(robot_pos - object_pos)
gripper_open = 1.0 if gripper_open else 0.0
object_grasped = 1.0 if object_in_gripper else 0.0

# Reward shaping: guide agent through stages
reward = (
    -0.1 * distance_to_object +      # Get closer to object
    0.5 * gripper_open +              # Open gripper when near
    1.0 * object_grasped              # Bonus for grasping
)
```

**Characteristics**:
- ✅ Combines benefits of dense rewards
- ✅ Can encode expert knowledge
- ✅ Often leads to faster learning
- ❌ Can bias agent away from optimal solution
- ❌ Requires domain expertise

---

## 2. Common Reward Function Designs

### Distance-Based Rewards

```python
# Continuous distance reward
distance = np.linalg.norm(current_state - goal_state)
reward = -distance  # Minimize distance
```

**Pros**: Natural, smooth gradient  
**Cons**: May not capture task complexity (e.g., obstacles)

---

### Stage-Based Rewards

```python
# Reward for completing stages
if stage == "approach":
    reward = -distance_to_object
elif stage == "grasp":
    reward = 1.0 if object_grasped else -0.1
elif stage == "lift":
    reward = 2.0 if object_lifted else -0.1
```

**Pros**: Guides agent through task phases  
**Cons**: Requires defining stages manually

---

### Penalty-Based Rewards

```python
# Penalize unwanted behaviors
reward = (
    1.0 if task_complete else 0.0
    - 0.01 * energy_consumed      # Penalize high energy
    - 0.1 * collisions             # Penalize collisions
    - 0.05 * joint_limits_exceeded  # Penalize unsafe actions
)
```

**Pros**: Discourages bad behaviors  
**Cons**: Can create local minima if penalties are too large

---

## 3. Impact on Learning

### Learning Speed

| Reward Type | Learning Speed | Why |
|------------|----------------|-----|
| **Sparse** | Very Slow | No feedback until success |
| **Dense** | Fast | Continuous learning signal |
| **Shaped** | Very Fast | Expert guidance |

### Sample Efficiency

- **Sparse**: Requires millions of samples
- **Dense**: Requires thousands to hundreds of thousands
- **Shaped**: Can require only thousands

### Final Performance

- **Sparse**: Often finds optimal solution (if it learns)
- **Dense**: May converge to suboptimal local minima
- **Shaped**: Depends on quality of shaping

---

## 4. Common Problems with Reward Functions

### Problem 1: Reward Hacking

**What it is**: Agent finds a way to maximize reward that doesn't solve the actual task.

**Example**:
```python
# Intended: robot should pick up object
reward = 1.0 if object_in_gripper else 0.0

# Hacked: robot learns to "trick" sensor
# (e.g., by moving gripper to always trigger sensor)
```

**Solution**: Add constraints, use multiple reward components, or use sparse rewards.

---

### Problem 2: Reward Scale Issues

**What it is**: Different reward components have vastly different scales.

**Example**:
```python
reward = (
    1000.0 * task_complete +    # Huge reward
    0.001 * distance_to_goal   # Tiny reward (ignored!)
)
```

**Solution**: Normalize or scale rewards to similar magnitudes.

---

### Problem 3: Sparse Reward Credit Assignment

**What it is**: Agent doesn't know which actions led to success.

**Example**:
```python
# Agent takes 100 steps, then gets reward=1.0
# Which of the 100 actions was important?
```

**Solution**: Use reward shaping or dense rewards.

---

### Problem 4: Conflicting Objectives

**What it is**: Different reward components push agent in opposite directions.

**Example**:
```python
reward = (
    1.0 * speed +        # Go fast!
    -1.0 * collisions   # But don't crash!
)
# Agent is confused: should it go fast or slow?
```

**Solution**: Carefully balance weights or use hierarchical rewards.

---

## 5. Reward Function Design Principles

### Principle 1: Align with True Objective

The reward should reflect what you actually want the agent to do.

```python
# ❌ Bad: Reward doesn't match goal
reward = -distance_to_goal  # But task is to avoid obstacles!

# ✅ Good: Reward matches goal
reward = 1.0 if reached_goal_safely else -0.1 * collisions
```

---

### Principle 2: Provide Learning Signal

Every step should provide some information (if possible).

```python
# ❌ Bad: No signal until end
reward = 1.0 if task_done else 0.0

# ✅ Good: Continuous signal
reward = -distance_to_goal + 0.1 * progress_made
```

---

### Principle 3: Scale Appropriately

All reward components should be on similar scales.

```python
# ❌ Bad: Unbalanced scales
reward = 1000.0 * task_done + 0.001 * smoothness

# ✅ Good: Balanced scales
reward = 1.0 * task_done + 0.1 * smoothness
```

---

### Principle 4: Avoid Reward Hacking

Design rewards that are hard to exploit.

```python
# ❌ Bad: Easy to hack
reward = 1.0 if sensor_triggered else 0.0

# ✅ Good: Harder to hack
reward = (
    1.0 if object_in_gripper and gripper_closed else 0.0
    - 0.1 * distance_to_object  # Also encourage approach
)
```

---

## 6. Experimental Insights (What Problem 1 Explores)

### Typical Experiment Setup

1. **Sparse Reward Baseline**
   - Reward = 1.0 only when task complete
   - Measure: Learning time, final performance

2. **Dense Reward Variant**
   - Reward = -distance_to_goal at every step
   - Measure: Learning speed, convergence

3. **Shaped Reward Variant**
   - Reward = combination of distance, progress, milestones
   - Measure: Sample efficiency, final performance

### Expected Results

| Metric | Sparse | Dense | Shaped |
|--------|--------|-------|--------|
| **Samples to Learn** | Millions | Thousands | Hundreds |
| **Learning Speed** | Very Slow | Fast | Very Fast |
| **Final Performance** | High (if learns) | Medium-High | High |
| **Robustness** | High | Medium | Depends |

---

## 7. Practical Tips for Reward Design

### Tip 1: Start Simple

Begin with sparse rewards to understand the task, then add shaping.

### Tip 2: Visualize Reward Distribution

Plot reward over time to see if agent is learning:
```python
# If reward stays flat → agent not learning
# If reward increases → agent learning
```

### Tip 3: Use Reward Normalization

Normalize rewards to [0, 1] or [-1, 1] for stability:
```python
reward_normalized = (reward - reward_min) / (reward_max - reward_min)
```

### Tip 4: Test Multiple Designs

Try different reward functions and compare:
- Learning curves
- Final performance
- Sample efficiency

### Tip 5: Consider Intrinsic Motivation

For sparse reward tasks, add curiosity or exploration bonuses:
```python
reward = (
    1.0 if task_done else 0.0
    + 0.01 * novelty_bonus  # Encourage exploration
)
```

---

## 8. Key Takeaways

1. **Sparse rewards** are simple but slow to learn from
2. **Dense rewards** speed up learning but require careful design
3. **Reward shaping** can dramatically improve sample efficiency
4. **Reward hacking** is a real problem - design defensively
5. **Scale matters** - balance different reward components
6. **There's no one-size-fits-all** - design for your specific task

---

## 9. Further Reading

- **Reward Shaping**: Ng et al. "Policy Invariance Under Reward Transformations"
- **Reward Hacking**: Amodei et al. "Concrete Problems in AI Safety"
- **Intrinsic Motivation**: Oudeyer & Kaplan "What is Intrinsic Motivation?"

---

## 10. Practice Questions

1. **Why do sparse rewards make learning difficult?**
   - Answer: Credit assignment problem - agent doesn't know which actions led to success.

2. **What is reward hacking? Give an example.**
   - Answer: Agent maximizes reward in unintended ways. Example: Robot learns to trigger sensor without actually grasping object.

3. **How can you prevent reward scale issues?**
   - Answer: Normalize rewards or ensure all components are on similar scales.

4. **When should you use dense vs sparse rewards?**
   - Answer: Dense when you can design meaningful intermediate rewards. Sparse when task is simple or you want to avoid reward hacking.

5. **What is the trade-off between learning speed and optimality?**
   - Answer: Dense/shaped rewards learn faster but may converge to suboptimal solutions. Sparse rewards are slower but often find optimal solutions.

---

## Summary

Reward function design is one of the most important aspects of RL. The choice between sparse, dense, and shaped rewards involves trade-offs between:
- **Learning speed** (dense > sparse)
- **Sample efficiency** (shaped > dense > sparse)
- **Optimality** (sparse often best, if it learns)
- **Design complexity** (sparse < dense < shaped)

Understanding these trade-offs helps you design better reward functions for your specific task!

