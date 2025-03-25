import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id, seed, idx, capture_video):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        print(env.observation_space)

        env.action_space.seed(seed)
        return env

    return thunk


class CategoricalDQN(nn.Module):
    def __init__(self, env, atom_size=51, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size).to(env.device)  # buffer 등록 필요 시 register_buffer 사용

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, env.single_action_space.n * atom_size)
        )
        self.action_dim = env.single_action_space.n

    def forward(self, x):
        x = self.network(x / 255.0)
        x = x.view(-1, self.action_dim, self.atom_size)
        return F.softmax(x, dim=2)

    def q_values(self, x):
        prob = self.forward(x)
        support = self.support.expand_as(prob)
        return torch.sum(prob * support, dim=2)

def projection_distribution(next_dist, rewards, dones, gamma, support, v_min, v_max, atom_size):
    batch_size = rewards.size(0)
    delta_z = float(v_max - v_min) / (atom_size - 1)
    projected_dist = torch.zeros((batch_size, atom_size), device=rewards.device)

    for i in range(atom_size):
        tz_j = torch.clamp(rewards + gamma * support[i] * (1 - dones), v_min, v_max)
        b_j = (tz_j - v_min) / delta_z
        l = b_j.floor().long()
        u = b_j.ceil().long()

        eq_mask = (u == l).float()
        projected_dist.view(-1).index_add_(
            0,
            (l + torch.arange(batch_size, device=rewards.device) * atom_size).view(-1),
            (next_dist[:, i] * (u.float() - b_j + eq_mask)).view(-1)
        )
        projected_dist.view(-1).index_add_(
            0,
            (u + torch.arange(batch_size, device=rewards.device) * atom_size).view(-1),
            (next_dist[:, i] * (b_j - l.float())).view(-1)
        )

    return projected_dist

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.observations = np.zeros((buffer_size, *state_dim), dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size, *state_dim), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        total = self.buffer_size if self.full else self.pos
        indices = np.random.choice(total, batch_size, replace=False)

        obs_batch = self.observations[indices]
        next_obs_batch = self.next_observations[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        dones_batch = self.dones[indices]

        return (
            torch.tensor(obs_batch, dtype=torch.uint8, device=self.device),
            torch.tensor(actions_batch, dtype=torch.int64, device=self.device),
            torch.tensor(next_obs_batch, dtype=torch.uint8, device=self.device),
            torch.tensor(rewards_batch, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(dones_batch, dtype=torch.float32, device=self.device).unsqueeze(1),
        )




if __name__ == "__main__":
    model_path = "./model"
    os.makedirs(model_path, exist_ok=True)
    seed = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = "BreakoutNoFrameskip-v4"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    learning_rate = 1e-4
    buffer_size = 100000 
    total_timesteps = 10000000
    start_e = 1.0
    end_e = 0.1
    exploration_fraction = 0.1
    learning_starts = 80000
    train_frequency = 4
    batch_size = 32
    gamma = 0.99
    target_network_frequency = 1000
    tau = 1.0
    atom_size = 51
    v_min = -10.0
    v_max = 10.0


    episode = 0
    use_wandb = False
    if use_wandb:
        import wandb

        wandb.init(
            project="dqn-breakout",
            config={
                "env_name": env_name,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "gamma": gamma,
                "start_e": start_e,
                "end_e": end_e,
                "exploration_fraction": exploration_fraction,
                "train_frequency": train_frequency,
                "learning_starts": learning_starts,
                "target_network_frequency": target_network_frequency,
                "tau": tau,
                "seed": seed,
            },
        )

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, seed + i, i, False) for i in range(1)]
    )

    q_network = CategoricalDQN(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = CategoricalDQN(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    obs_shape = envs.single_observation_space.shape
    action_shape = (1,)  # Discrete 환경일 경우

    rb = ReplayBuffer(
    buffer_size=buffer_size,
    state_dim=obs_shape,
    action_dim=action_shape[0],
    device=device
    )



    state, _ = envs.reset(seed=seed)
    for global_step in range(total_timesteps):
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(state).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_state, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode += 1
                    print(f"steps:{global_step}, episode:{episode}, reward:{info['episode']['r']}, stepLength:{info['episode']['l']}")
                    if use_wandb:
                        wandb.log(
                            {
                                "episode": episode,
                                "episodic_return": info["episode"]["r"],
                                "episodic_length": info["episode"]["l"],
                                "epsilon": epsilon,
                                "global_step": global_step,
                            }
                        )

        real_next_state = next_state.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_state[idx] = infos["final_observation"][idx]
        rb.add(state, real_next_state, actions, rewards, terminations)


        state = next_state

        if global_step > learning_starts and global_step % train_frequency == 0:
            obs, actions, next_obs, rewards, dones = rb.sample(batch_size)

            with torch.no_grad():
                next_dist = target_network(next_obs)  # (B, A, atoms)
                next_q = torch.sum(next_dist * target_network.support, dim=2)  # (B, A)
                next_actions = torch.argmax(next_q, dim=1, keepdim=True)  # (B, 1)
                next_dist = next_dist[range(batch_size), next_actions.squeeze()]  # (B, atoms)

                target_dist = projection_distribution(
                    next_dist, rewards, dones, gamma,
                    target_network.support, target_network.v_min, target_network.v_max, target_network.atom_size
                )

            dist = q_network(obs)  # (B, A, atoms)
            action_dist = dist[range(batch_size), actions.squeeze()]  # (B, atoms)
            loss = -torch.sum(target_dist * torch.log(action_dist + 1e-8), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({
                        "loss": loss.item(),
                        "global_step": global_step
                        })


            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )

            if episode % 1000 == 0:
                model_file = os.path.join(model_path, f"Breakout_dqn_classic_{episode}.pth")
                torch.save(q_network.state_dict(), model_file)
                print(f"✅ Saved model at episode {episode} to {model_file}")

    envs.close()
