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

import wandb

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

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


class ReplayMemory:
    def __init__(self, capacity, state_dim):
        self.states = np.zeros((capacity, *state_dim), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.currentIdx = 0
        self.maxLen = capacity
        self.size = 0  # 실제 저장된 개수

    def add(self, state, action, reward, next_state, done):
        idx = self.currentIdx % self.maxLen  # 원형 버퍼 적용
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.currentIdx += 1
        self.size = min(self.size + 1, self.maxLen)  # 저장된 개수 업데이트

    def sample(self, batch_size):
        batch_indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states[batch_indices],
            self.dones[batch_indices],
        )

    
def make_env(env_name, seed):
    env = gym.make(env_name,frameskip=1 ,render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84,84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env,4)
    # env.action_space.seed(seed)
    return env

def linear_epsilon_decay(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
    
env_name = "ALE/Breakout-v5"
seed = 1
learning_rate = 1e-4
buffer_size = 200000
gamma = 0.99
batch_size = 64
target_network_frequency = 1000
train_frequency = 4
total_timesteps = 10000000
start_e = 1.0
end_e = 0.05
exploration_fraction = 0.1
learning_starts = 80000

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
targetQ_net.load_state_dict(targetQ_net.state_dict())
print("Env Information")
print(f"Observation:{env.observation_space}")
print(f"Action:{env.action_space}")

memory = ReplayMemory(buffer_size,env.observation_space.shape)

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
print(state.shape)
for current_step in range(total_timesteps):
    epsilon = linear_epsilon_decay(start_e, end_e,exploration_fraction * total_timesteps,current_step)
    wandb.log({"epsilon": epsilon}, step=current_step)
    if random.random() < epsilon:
        actions = random.sample([0,1,2,3],1)[0]
        
    else:
        obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # 이 부분 이해 안감
        # obs_tensor Gradient 추적 시작
        Q_value = behaveQ_net(obs_tensor)  
        # print(Q_value)
        actions = np.argmax(Q_value.detach().cpu().numpy()) # gradient 추적 종료 후 cpu 이동 및 넘파이 전달

    next_state, reward, terminated, truncated, infos =  env.step(actions)
    done = terminated or truncated
    
    if "episode" in infos:
        episodic_return = infos["episode"]["r"]
        episodic_steps = infos["episode"]["l"]
        print(f"global_step={current_step}, episodic_return={episodic_return}, current_episode={episode}")
        wandb.log({"episodic_return": episodic_return}, step=current_step)
        wandb.log({"episodic_steps": episodic_steps}, step=current_step)
        env.reset()
        episode += 1
    memory.add(state, actions, reward, next_state, done)
    state = next_state
    
    if current_step > learning_starts and current_step % train_frequency == 0:
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # Tensor 변환
        state_tensor = torch.from_numpy(states).float().to(device)
        next_state_tensor = torch.from_numpy(next_states).float().to(device)
        actions_tensor = torch.from_numpy(actions).long().to(device)
        rewards_tensor = torch.from_numpy(rewards).float().to(device)
        dones_tensor = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            maxtarget, _ = targetQ_net(next_state_tensor).max(dim=1)
            tdtarget = rewards_tensor + gamma * maxtarget * (1 - dones_tensor)

        behavior = behaveQ_net(state_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        loss = F.mse_loss(tdtarget,behavior)
        wandb.log({"loss": loss.item()}, step=current_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if current_step % target_network_frequency == 0:
        targetQ_net.load_state_dict(behaveQ_net.state_dict())

    if episode % 1000 == 0:
        torch.save(behaveQ_net.state_dict(), f"./model/Breakout_{episode}.pth")
        print(f"{episode} model saved")

env.close()
wandb.finish()



