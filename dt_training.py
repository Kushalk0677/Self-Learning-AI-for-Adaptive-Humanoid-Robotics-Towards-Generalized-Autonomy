import gymnasium as gym
import torch
import torch.nn as nn
from transformers import BertModel

# Initialize Environment
env = gym.make("Humanoid-v5")

# Define Decision Transformer Model
class DecisionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecisionTransformer, self).__init__()
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, output_dim)

    def forward(self, x):
        x = self.transformer(x)[0]
        return self.fc(x[:, 0])

# Initialize Model
dt_model = DecisionTransformer(env.observation_space.shape[0], env.action_space.shape[0])

# Save Model Locally
torch.save(dt_model.state_dict(), "decision_transformer.pth")

# Save to Google Drive
!cp decision_transformer.pth "/content/drive/My Drive/decision_transformer.pth"
print("✅ Decision Transformer Model Saved to Drive!")
