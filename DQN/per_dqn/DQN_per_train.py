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


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device="cpu", alpha=0.6, beta_start=0.4, beta_frames=1000000):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.observations = np.zeros((buffer_size, *state_dim), dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size, *state_dim), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

        self.pos = 0
        self.full = False

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def add(self, obs, next_obs, action, reward, done):
        max_prio = self.priorities.max() if self.full else 1.0

        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        if self.full:
            prios = self.priorities
            total = self.buffer_size
        else:
            prios = self.priorities[:self.pos]
            total = self.pos

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

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
            torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1),
            indices
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities.detach().cpu().numpy()



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

    episode = 0
    use_wandb = True
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

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    obs_shape = envs.single_observation_space.shape
    action_shape = (1,)  # Discrete 환경일 경우

    rb = PrioritizedReplayBuffer(
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

        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                obs, actions, next_obs, rewards, dones, weights, indices = rb.sample(batch_size)
                with torch.no_grad():
                    target_max = target_network(next_obs).max(1)[0]
                    td_target = rewards.flatten() + gamma * target_max * (1 - dones.flatten())

                q_values = q_network(obs)
                q_action = q_values.gather(1, actions).squeeze()

                td_error = td_target - q_action
                loss = (weights.flatten() * td_error ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                new_priorities = td_error.abs() + 1e-6
                rb.update_priorities(indices, new_priorities)

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
