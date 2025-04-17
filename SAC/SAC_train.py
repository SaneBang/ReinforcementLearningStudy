import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import os
import wandb
import random
import time

#test
class ReplayBuffer:
    def __init__(self, state_dim, max_len=100000, device='cpu'):
        self.states   = np.zeros((max_len, state_dim), dtype=np.float32)
        self.actions  = np.zeros((max_len, 1), dtype=np.float32)
        self.rewards  = np.zeros((max_len, 1), dtype=np.float32)
        self.nstates  = np.zeros((max_len, state_dim), dtype=np.float32)

        self.max_len = max_len
        self.device = device

        self.current_idx = 0
        self.current_size = 0

    def add(self, state, action, reward, next_state):
        idx = self.current_idx % self.max_len

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.nstates[idx] = next_state

        self.current_idx += 1
        self.current_size = min(self.current_size + 1, self.max_len)

    def sample(self, batch_size, as_tensor=True):
        indices = np.random.choice(self.current_size, size=batch_size, replace=False)

        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.nstates[indices]
        )

        if as_tensor:
            return tuple(torch.tensor(b, device=self.device) for b in batch)
        return batch

    def __len__(self):
        return self.current_size


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        obs_dim = int(np.prod(env.observation_space.shape))  
        act_dim = int(np.prod(env.action_space.shape))     
        input_dim = obs_dim + act_dim

        hidden_dim = 256

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value
    
LOG_STD_MIN = -5
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        hidden_dim = 256

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)

        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        # [-1, 1] → [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() #sample과 rsample의 차이점은 미분 가능과 불가능 차이인데.. 왜 어떤건 가능하고 어떤건 안되지? 그리고 이거 backpropagation을 어디서 함?
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # log_prob 계산
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action
    
# 저장 경로
save_dir = "./model_halfcheetah_SAC"
os.makedirs(save_dir, exist_ok=True)

# A3C 설정
GAMMA = 0.99
LR = 1e-4
MAX_EPISODES = 30000
ENV_NAME = "HalfCheetah-v5"

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(ENV_NAME)
    
    print(env.action_space)
    print(env.action_space.high)
    print(env.action_space.low)
    print(env.observation_space)

    SoftActor = Actor(env).to(device)

    Qfunction01 = SoftQNetwork(env).to(device)
    Qfunction02 = SoftQNetwork(env).to(device)

    TargetQfunction01 = SoftQNetwork(env).to(device)
    TargetQfunction02 = SoftQNetwork(env).to(device)

    TargetQfunction01.load_state_dict(Qfunction01.state_dict())
    TargetQfunction02.load_state_dict(Qfunction02.state_dict())

    QfuncOptim = optim.Adam(list(Qfunction01.parameters()) + list(Qfunction02.parameters()), lr = 1e-3)  #? 이거 뭐임?4
    SoftActorOptim = optim.Adam(list(SoftActor.parameters()), lr = 3e-4)


    memory = ReplayBuffer(env.observation_space.shape)

    start_time = time.time()
    
    state,_ = env.reset(seed=0)

    for current_step in range(100000):
        # if current_step < 10000:
        #     actions = 
        # else:
        action, _, _ = SoftActor.get_action(torch.Tensor(env).to(device))
        action = action.detach().cpu().numpy() 
        print(action)