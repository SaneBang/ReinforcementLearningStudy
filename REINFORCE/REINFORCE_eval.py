import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

save_dir = "./model"
os.makedirs(save_dir, exist_ok=True)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanh()
        )
        self.mean_head = nn.Linear(32, act_dim)
        self.std_head = nn.Linear(32, act_dim)

    def forward(self, x):
        x = self.shared(x)
        mean = self.mean_head(x)
        std = torch.log(1 + torch.exp(self.std_head(x)))  # Softplus
        return mean, std

class REINFORCE:
    def __init__(self, obs_dim, act_dim, lr=1e-4, gamma=0.99):
        self.gamma = gamma
        self.eps = 1e-6
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.log_probs, self.rewards = [], []

    def sample_action(self, state):
        state = torch.tensor([state], dtype=torch.float32)
        mean, std = self.policy(state)
        dist = Normal(mean, std + self.eps)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action).sum())
        return action.detach().numpy()[0]

    def update(self):
        R, returns = 0, []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs.clear(), self.rewards.clear()



# 평가 환경 설정 (동영상 저장 포함)
eval_env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
obs_dim = eval_env.observation_space.shape[0]
act_dim = eval_env.action_space.shape[0]
eval_env = gym.wrappers.RecordVideo(
    eval_env,
    video_folder=os.path.join(save_dir, "videos"),
    name_prefix="reinforce_eval",
    episode_trigger=lambda episode_id: True,  # 모든 에피소드 저장
    disable_logger=True
)

# 저장된 모델 불러오기
eval_policy = PolicyNetwork(obs_dim, act_dim)
model_path = os.path.join(save_dir, "reinforce_policy_seed1.pt")
eval_policy.load_state_dict(torch.load(model_path))
eval_policy.eval()

# 에피소드 평가
state, _ = eval_env.reset(seed=42)
done = False
total_reward = 0

while not done:
    state_tensor = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        mean, std = eval_policy(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
    state, reward, terminated, truncated, _ = eval_env.step(action.numpy()[0].reshape(-1))
    total_reward += reward
    done = terminated or truncated

eval_env.close()
print(f"Evaluation Total Reward: {total_reward}")
print(f"Saved evaluation video to: {os.path.join(save_dir, 'videos')}")
