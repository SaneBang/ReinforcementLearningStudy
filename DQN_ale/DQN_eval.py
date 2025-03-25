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

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

model_save_path = "/home/erp42/sangwoo/model/Breakout_5000.pth"
video_folder = "videos/"
os.makedirs(video_folder, exist_ok=True)  # 동영상 저장 폴더 생성
# print(os.curdir)
def make_eval_env(env_name, seed):
    env = gym.make(env_name,frameskip=1,render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)  # 모든 에피소드 녹화
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    # env.action_space.seed(seed)
    return env

# 환경 및 모델 불러오기
env_name = "ALE/Breakout-v5"
seed = random.randint(1, 100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_eval_env(env_name, seed=seed)

state, _ = env.reset(seed=seed)

# Q-Network 모델 정의 (훈련 시와 동일한 구조)
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )
    
    def forward(self, x):
        return self.network(x / 255.0)

# 모델 로드
q_network = QNetwork(env).to(device)
q_network.load_state_dict(torch.load(model_save_path, map_location=device))
q_network.eval()

# 평가 실행
num_episodes = 5  # 녹화할 에피소드 수
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.Tensor(state).unsqueeze(0).to(device)
        # print(state_tensor.shape)
        with torch.no_grad():
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            # print(action)
        
        state, reward, terminated, truncated, infos = env.step(action)
        done = terminated or truncated
    
    if "episode" in infos:
        print(f"total reward: {infos['episode']['r']}") 

    # print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
print(f"Videos saved in {video_folder}")
