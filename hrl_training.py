import gymnasium as gym
import torch
import torch.nn as nn

# Initialize Environment
env = gym.make("Humanoid-v5")
obs, _ = env.reset()

# Define High-Level Policy Network
class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# Initialize Policy
high_level_policy = HighLevelPolicy(env.observation_space.shape[0], 3)

# Save Model Locally
torch.save(high_level_policy.state_dict(), "high_level_policy.pth")

# Save to Google Drive
!cp high_level_policy.pth "/content/drive/My Drive/high_level_policy.pth"
print("✅ High-Level Policy Model Saved to Drive!")
