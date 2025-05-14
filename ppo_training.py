import gymnasium as gym
import os
from stable_baselines3 import PPO

# Initialize Environment
env = gym.make("Humanoid-v5")

# Load or Initialize PPO Model
ppo_model = PPO("MlpPolicy", env, verbose=1)

if os.path.exists("humanoid_ppo.zip"):
    ppo_model = PPO.load("humanoid_ppo.zip", env)
    print("✅ PPO Model Loaded from Checkpoint!")

# Train and Save Checkpoints
for step in range(10):  # 1M steps (100K each)
    print(f"Training PPO... Interval {step+1}/10")
    ppo_model.learn(total_timesteps=100_000)
    ppo_model.save("humanoid_ppo.zip")

    # Save to Google Drive
    !cp humanoid_ppo.zip "/content/drive/My Drive/humanoid_ppo.zip"
    print(f"✅ PPO Checkpoint Saved to Drive at {((step+1) * 100_000)} Steps")
