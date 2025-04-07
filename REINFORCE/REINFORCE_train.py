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

# 저장 경로 설정
save_dir = "./model_halfcheetah"
os.makedirs(save_dir, exist_ok=True)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.mean_head = nn.Linear(64, act_dim)
        self.std_head = nn.Linear(64, act_dim)

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
        return action.detach().numpy()

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

# ✅ HalfCheetah-v5 환경 설정
env = gym.make("HalfCheetah-v5")
env = gym.wrappers.RecordEpisodeStatistics(env, 50)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

all_rewards = []
for seed in [1]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = REINFORCE(obs_dim, act_dim)
    seed_rewards = []
    for ep in range(1000):  # 보통 1,000~3,000 에피소드로도 충분
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            done = terminated or truncated
        seed_rewards.append(env.return_queue[-1])
        agent.update()
        if ep % 100 == 0:
            print(f"Seed {seed} Episode {ep} AvgReward (last 50): {int(np.mean(env.return_queue))}")

    # ✅ 모델 저장
    model_path = os.path.join(save_dir, f"reinforce_policy_seed{seed}.pt")
    torch.save(agent.policy.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    all_rewards.append(seed_rewards)

# ✅ 보상 그래프 저장
sns.set(style="darkgrid")
df = pd.DataFrame(all_rewards).T.melt(var_name="seed", value_name="reward")
sns.lineplot(data=df, x=df.index % len(seed_rewards), y="reward", hue="seed")
plt.title("REINFORCE on HalfCheetah-v5")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plot_path = os.path.join(save_dir, "reinforce_reward_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"Saved plot to: {plot_path}")
