import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import mujoco

# ✅ Mount Google Drive for Checkpoint Saving
from google.colab import drive
drive.mount('/content/drive')

# ✅ Define Checkpoint Directory
checkpoint_dir = "/content/drive/My Drive/MAML_Checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)  # ✅ Ensure directory exists

# ✅ Initialize MuJoCo Humanoid Environment
env = gym.make("Humanoid-v5")
obs, _ = env.reset()

# ✅ Select Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define Meta-Learning Model
class MetaLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ✅ Initialize Model & Optimizer
meta_model = MetaLearner(env.observation_space.shape[0], env.action_space.shape[0])
meta_model.to(device)
optimizer = optim.Adam(meta_model.parameters(), lr=5e-6)

# ✅ Find Latest Checkpoint & Resume Training
existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("meta_learner_")]

start_interval = 1  # Default start if no checkpoint is found

if existing_checkpoints:
    latest_checkpoint = max(existing_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Get latest numbered file
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    meta_model.load_state_dict(torch.load(latest_checkpoint_path, map_location=device))

    # ✅ Extract last completed interval number
    start_interval = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
    print(f"✅ Loaded Checkpoint: {latest_checkpoint} (Resuming from Interval {start_interval})")

# ✅ Start Training from Last Interval
num_intervals = 500  # Total Intervals
iterations_per_task = 7500  # More iterations per task
tasks = ["walk", "run", "jump", "climb", "dodge", "balance", "crouch", "reach", "crawl", "sidestep", "step-up", "duck"]

start_time = time.time()

# ✅ Training Loop
for step in range(start_interval, num_intervals + 1):  # ✅ Start from correct interval
    print(f"Training MAML... Interval {step}/{num_intervals}")

    for task in tasks:
        obs, _ = env.reset(options={"random": True})

        for i in range(iterations_per_task):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action = meta_model(obs_tensor)
            obs, reward, done, truncated, info = env.step(action.detach().cpu().numpy())

            reward_tensor = torch.tensor(reward * 100, dtype=torch.float32, requires_grad=True).to(device)

            loss = -reward_tensor.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), max_norm=1.0)
            optimizer.step()

            if i % 500 == 0:
                obs, _ = env.reset(options={"random": True})

    # ✅ Save Model Checkpoint Every Interval
    if (step + 1) % 1 == 0:
        checkpoint_filename = f"meta_learner_{step}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(meta_model.state_dict(), checkpoint_path)
        print(f"✅ Checkpoint Saved: {checkpoint_filename}")

# ✅ Training Completion
elapsed_time = time.time() - start_time
print(f"Total Training Time: {elapsed_time / 3600:.2f} hours")


