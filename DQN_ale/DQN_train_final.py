import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ale_py
gym.register_envs(ale_py)

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import wandb

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4,32,8,stride=4), nn.ReLU(),
            nn.Conv2d(32,64,4,stride=2), nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136,512), nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )
    def forward(self, x):
        return self.network(x / 255.0)


def make_env(env_name, seed):
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
    env = ClipRewardEnv(env)   
    env = gym.wrappers.FrameStackObservation(env, 4)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def linear_epsilon_decay(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
    
env_name = "ALE/Breakout-v5"
seed = 1
learning_rate = 1e-4
buffer_size = 100000
gamma = 0.99
batch_size = 64
target_network_frequency = 1000
train_frequency = 4
total_timesteps = 5000000
start_e = 1.0
end_e = 0.05
exploration_fraction = 0.1
learning_starts = 80000
tau = 1.0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_env(env_name=env_name, seed=seed)
episode = 0

import os
os.makedirs("model", exist_ok=True)

behaveQ_net = QNetwork(env).to(device)
targetQ_net = QNetwork(env).to(device)
optimizer = optim.Adam(behaveQ_net.parameters(), lr=learning_rate)
targetQ_net.load_state_dict(behaveQ_net.state_dict())


memory =ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
        )

wandb.init(
    project="dqn-breakout",
    name=f"run-seed-{seed}",
    config={
        "env": env_name,
        "total_timesteps": total_timesteps,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "exploration_fraction": exploration_fraction,
        "start_e": start_e,
        "end_e": end_e,
        "train_freq": train_frequency,
        "target_update_freq": target_network_frequency,
        "learning_starts": learning_starts,
        "seed": seed,
    }
)

state, info = env.reset(seed=seed)
for current_step in range(total_timesteps):
    epsilon = linear_epsilon_decay(start_e, end_e,exploration_fraction * total_timesteps,current_step)
    wandb.log({"epsilon": epsilon}, step=current_step)
    
    if random.random() < epsilon:
        actions = env.action_space.sample()
        
    else:
        q_value = behaveQ_net(torch.Tensor(np.expand_dims(state, axis=0)).to(device))
        actions = torch.argmax(q_value, dim=1).cpu().numpy()[0]

    next_state, reward, terminated, truncated, infos =  env.step(actions)
    done = terminated or truncated
    if "episode" in infos:
        episodic_return = infos["episode"]["r"]
        print(f"global_step={current_step}, episodic_return={episodic_return}, current_episode={episode}")
        wandb.log({"episodic_return": episodic_return}, step=current_step)
        episode += 1

    memory.add(state, next_state, actions, reward, done, infos)
    
    if not done:
        state = next_state
    else:
        state, _ = env.reset()
    
    if current_step > learning_starts:
        if current_step % train_frequency == 0:
            data = memory.sample(batch_size)
            with torch.no_grad():
                target_max, _ = targetQ_net(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
            old_val = behaveQ_net(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)
            wandb.log({"loss": loss.item()}, step=current_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if current_step % target_network_frequency == 0:
            for target, behavior in zip(targetQ_net.parameters(), behaveQ_net.parameters()):
                target.data.copy_(
                    tau * behavior.data + (1.0  - tau) * target.data
                )

    if episode % 1000 == 0:
        torch.save(behaveQ_net.state_dict(), f"./model/Breakout_{episode}.pth")
        # print(f"{episode} model saved")
env.close()
wandb.finish()



