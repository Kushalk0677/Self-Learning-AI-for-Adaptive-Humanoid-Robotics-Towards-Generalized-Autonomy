import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import pickle
from stable_baselines3 import PPO
from transformers import BertModel

# ------------------ Load Trained Models ------------------

print("🔄 Loading models...")

# Load PPO Model (Motor Control)
ppo_model = PPO.load("humanoid_ppo.zip")
print("✅ PPO Model Loaded")

# Load High-Level HRL Policy
class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

env = gym.make("Humanoid-v5")
high_level_policy = HighLevelPolicy(env.observation_space.shape[0], 3)
high_level_policy.load_state_dict(torch.load("high_level_policy.pth"))
high_level_policy.eval()
print("✅ High-Level Policy Loaded")

# Load Decision Transformer Model
class DecisionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecisionTransformer, self).__init__()
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, output_dim)

    def forward(self, x):
        x = self.transformer(x)[0]
        return self.fc(x[:, 0])

dt_model = DecisionTransformer(env.observation_space.shape[0], env.action_space.shape[0])
dt_model.load_state_dict(torch.load("decision_transformer.pth"))
dt_model.eval()
print("✅ Decision Transformer Loaded")

# Load Meta-Learner (MAML)
class MetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

meta_model = MetaLearner(env.observation_space.shape[0], env.action_space.shape[0])
meta_model.load_state_dict(torch.load("meta_learner.pth"))
meta_model.eval()
print("✅ Meta-Learner Loaded")

# Load Sensor Fusion Data
with open("sensor_fusion.pkl", "rb") as f:
    sensor_fusion = pickle.load(f)
print("✅ Sensor Fusion Model Loaded")

print("✅ All models successfully loaded!")

# ------------------ Run the Merged AI Model in Simulation ------------------

test_env = gym.make("Humanoid-v5", render_mode="human")
obs, _ = test_env.reset()
total_reward = 0
total_steps = 1000

print("🏃 Running the AI in the MuJoCo simulation...")

for _ in range(total_steps):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    # Get actions from each model
    action_ppo, _ = ppo_model.predict(obs)  # PPO for motor control
    action_hrl = high_level_policy(obs_tensor).detach().numpy()  # HRL high-level planning
    action_dt = dt_model(obs_tensor).detach().numpy()  # Decision Transformer memory-based learning
    action_maml = meta_model(obs_tensor).detach().numpy()  # MAML fast adaptation

    # Get Sensor Fusion Data
    fused_obs = np.concatenate((sensor_fusion.get_fused_data(), obs), axis=0)

    # **Final Action Selection (Weighted Combination)**
    final_action = (
        (0.4 * action_ppo) +  # PPO controls locomotion
        (0.2 * action_hrl) +  # HRL guides high-level planning
        (0.2 * action_dt) +   # DT adds sequence-based decision-making
        (0.2 * action_maml)   # MAML enables fast adaptation
    )

    # Step in Environment
    obs, reward, done, truncated, _ = test_env.step(final_action)
    test_env.render()
    total_reward += reward
    if done:
        obs, _ = test_env.reset()

print(f"✅ Simulation complete! Total Reward: {total_reward:.2f}")

# ------------------ Performance Evaluation ------------------

def evaluate_model(env, episodes=10):
    print("📊 Evaluating AI Performance...")
    success_count = 0
    total_energy = 0
    adaptation_time = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Get actions from each model
            action_ppo, _ = ppo_model.predict(obs)
            action_hrl = high_level_policy(obs_tensor).detach().numpy()
            action_dt = dt_model(obs_tensor).detach().numpy()
            action_maml = meta_model(obs_tensor).detach().numpy()

            # Merge actions
            final_action = (0.4 * action_ppo) + (0.2 * action_hrl) + (0.2 * action_dt) + (0.2 * action_maml)

            obs, reward, done, truncated, info = env.step(final_action)
            total_energy += abs(final_action).sum()
            step_count += 1
            if done and reward > 0:
                success_count += 1

        adaptation_time += step_count

    success_rate = success_count / episodes
    avg_energy = total_energy / episodes
    avg_adaptation_time = adaptation_time / episodes

    print(f"✅ Task Success Rate: {success_rate * 100:.2f}%")
    print(f"⚡ Average Energy Consumption: {avg_energy:.2f}")
    print(f"⏳ Average Adaptation Time: {avg_adaptation_time:.2f} steps")
    return success_rate, avg_energy, avg_adaptation_time

# Run evaluation
success, energy, adaptation_time = evaluate_model(test_env)

# ------------------ Save Evaluation Metrics ------------------
evaluation_results = {
    "Success Rate": success,
    "Average Energy Consumption": energy,
    "Average Adaptation Time": adaptation_time
}

with open("evaluation_results.pkl", "wb") as f:
    pickle.dump(evaluation_results, f)

print("✅ Evaluation metrics saved successfully!")
