"""
Smart Grid AI Project - Module 4: AI Grid Optimization
------------------------------------------------------------
Covers:
  - Energy Distribution
  - Grid Load Balancing
  - Power Routing
  - Battery Charging Strategy
  - Renewable Energy Allocation

AI Components used: Optimization Algorithms (Linear Programming), Reinforcement Learning

Usage: python grid_optimization.py
"""

import pandas as pd
import numpy as np
import pulp

print("Loading data...")
df = pd.read_csv("data/cleaned_grid_data.csv", parse_dates=["timestamp"])

# ============================================================
# 1. ENERGY DISTRIBUTION + LOAD BALANCING + POWER ROUTING
# (Linear Programming Optimization)
# ============================================================
print("\n" + "=" * 55)
print("1. GRID OPTIMIZATION (Linear Programming)")
print("=" * 55)
print("Optimizing power distribution for a sample day (24 hours)...\n")

# Sample ek din ka data (first 24 hours)
sample = df.head(24).reset_index(drop=True)

# 3 regions banate hain jahan demand distribute karni hai (Grid Load Balancing / Power Routing)
np.random.seed(42)
region_weights = [0.5, 0.3, 0.2]  # region A, B, C - demand split
regions = ["Region_A", "Region_B", "Region_C"]

total_optimized_cost = 0
allocation_log = []

for idx, row in sample.iterrows():
    demand = row["total_demand_mw"]
    solar = row["solar_production_mw"]
    wind = row["wind_production_mw"]
    renewable_supply = solar + wind
    grid_supply_needed = max(0, demand - renewable_supply)

    # LP Problem: minimize cost of energy distribution across regions
    prob = pulp.LpProblem(f"GridOptimization_hour_{idx}", pulp.LpMinimize)

    # Decision variables: kitni renewable aur grid power har region ko milegi
    renewable_alloc = {r: pulp.LpVariable(f"renew_{r}_{idx}", lowBound=0) for r in regions}
    grid_alloc = {r: pulp.LpVariable(f"grid_{r}_{idx}", lowBound=0) for r in regions}

    # Cost: renewable sasti hai (cost=1), grid mehngi hai (cost=3)
    prob += pulp.lpSum([renewable_alloc[r] * 1 + grid_alloc[r] * 3 for r in regions])

    # Constraints: har region ki demand poori honi chahiye
    for i, r in enumerate(regions):
        region_demand = demand * region_weights[i]
        prob += renewable_alloc[r] + grid_alloc[r] >= region_demand

    # Total renewable allocation available renewable se zyada nahi ho sakti
    prob += pulp.lpSum([renewable_alloc[r] for r in regions]) <= renewable_supply

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    hour_cost = pulp.value(prob.objective)
    total_optimized_cost += hour_cost

    allocation_log.append({
        "hour": row["hour"],
        "demand_mw": round(demand, 1),
        "renewable_available_mw": round(renewable_supply, 1),
        "grid_power_needed_mw": round(grid_supply_needed, 1),
        "optimized_cost": round(hour_cost, 1)
    })

alloc_df = pd.DataFrame(allocation_log)
print(alloc_df.to_string(index=False))
print(f"\nTotal optimized distribution cost (24hr sample): {total_optimized_cost:.2f}")
alloc_df.to_csv("data/grid_optimization_schedule.csv", index=False)
print("Saved: data/grid_optimization_schedule.csv")

# ============================================================
# 2. RENEWABLE ENERGY ALLOCATION SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("2. RENEWABLE ENERGY ALLOCATION (Region-wise Split)")
print("=" * 55)
for i, r in enumerate(regions):
    print(f"{r}: {region_weights[i]*100:.0f}% of total demand allocated")

# ============================================================
# 3. BATTERY CHARGING STRATEGY (Reinforcement Learning)
# ============================================================
print("\n" + "=" * 55)
print("3. BATTERY CHARGING STRATEGY (Reinforcement Learning)")
print("=" * 55)

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class BatteryEnv(gym.Env):
        """
        Custom environment: battery ko kab charge/discharge karna hai decide karta hai.
        State: [demand, renewable_supply, battery_level]
        Action: 0 = discharge, 1 = hold, 2 = charge
        Reward: cost minimize karna (grid power kam use karna, battery overflow/empty na hone dena)
        """
        def __init__(self, data):
            super().__init__()
            self.data = data.reset_index(drop=True)
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
            self.battery_capacity = 800
            self.max_steps = len(self.data)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_idx = 0
            self.battery_level = self.battery_capacity * 0.5
            return self._get_obs(), {}

        def _get_obs(self):
            row = self.data.iloc[self.step_idx]
            return np.array([
                row["total_demand_mw"] / 15000,
                (row["solar_production_mw"] + row["wind_production_mw"]) / 1500,
                self.battery_level / self.battery_capacity
            ], dtype=np.float32)

        def step(self, action):
            row = self.data.iloc[self.step_idx]
            renewable = row["solar_production_mw"] + row["wind_production_mw"]
            demand = row["total_demand_mw"]

            if action == 0:  # discharge
                discharge_amt = min(self.battery_level, demand * 0.05)
                self.battery_level -= discharge_amt
                grid_needed = max(0, demand - renewable - discharge_amt)
            elif action == 2:  # charge
                charge_amt = min(self.battery_capacity - self.battery_level, renewable * 0.1)
                self.battery_level += charge_amt
                grid_needed = max(0, demand - (renewable - charge_amt))
            else:  # hold
                grid_needed = max(0, demand - renewable)

            reward = -grid_needed / 1000  # kam grid power use karo = better reward
            self.step_idx += 1
            terminated = self.step_idx >= self.max_steps - 1
            return self._get_obs(), reward, terminated, False, {}

    env = BatteryEnv(df.head(500))
    rl_model = PPO("MlpPolicy", env, verbose=0)
    rl_model.learn(total_timesteps=5000)

    # Evaluate
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated:
            break

    print(f"RL agent trained (PPO algorithm, 5000 timesteps)")
    print(f"Total reward over 100-step evaluation: {total_reward:.2f}")
    print("Agent learns WHEN to charge/discharge/hold the battery to minimize grid power usage.")

    rl_model.save("models/battery_rl_agent")
    print("Saved: models/battery_rl_agent.zip")

except ImportError as e:
    print(f"RL libraries not installed: {e}")
    print("Install with: pip install stable-baselines3 gymnasium")

print("\n" + "=" * 55)
print("Module 4 (AI Grid Optimization) complete.")
print("=" * 55)
