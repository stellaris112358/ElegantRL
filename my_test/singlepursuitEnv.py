from typing import Tuple
import random

import numpy as np
import numpy.random as rd
import torch as th
import time
import gymnasium as gym
from gymnasium import spaces

ARY = np.ndarray
TEN = th.Tensor


class PointChasingVecEnv:
    def __init__(self, dim=2, env_num=32, sim_device=0,seed=42):
        self.dim = dim
        self.init_distance = 8.0

        # reset buffers
        self.p0s = None  # evader positions (num_envs, dim)
        self.v0s = None  # evader velocities
        self.p1s = None  # pursuer positions
        self.v1s = None  # pursuer velocities

        self.distances = None  # (num_envs,) distance between p0 and p1
        self.cur_steps = None  # (num_envs,) current step number

        """env info"""
        self.env_name = "PointChasingVecEnv"
        self.state_dim = self.dim * 4
        self.action_dim = self.dim
        self.max_step = 2 ** 10
        self.if_discrete = False

        # number of parallel envs and device
        self.num_envs = env_num
        self.device = th.device("cpu" if sim_device == -1 else f"cuda:{sim_device}")

        # gymnasium-compatible spaces (single-environment shapes)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        # reproducibility
        self._seed = seed

        self.seed(self._seed)

    def seed(self, seed: int):
        """设置随机种子"""
        self._seed = int(seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        th.manual_seed(self._seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(self._seed)

    def reset(self, **_kwargs) -> Tuple[TEN, dict]:
        self.p0s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.v0s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.p1s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.v1s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)

        self.cur_steps = th.zeros(self.num_envs, dtype=th.float32, device=self.device)

        for env_i in range(self.num_envs):
            self.reset_env_i(env_i)

        self.distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5

        state = self.get_state()
        return state, dict()

    def reset_env_i(self, i: int):
        # Use deterministic per-env init for reproducibility across runs if desired.
        th.manual_seed(i)
        self.p0s[i] = th.normal(0, 1, size=(self.dim,), device=self.device)
        self.v0s[i] = th.zeros((self.dim,), device=self.device)
        self.p1s[i] = th.normal(-self.init_distance, 1, size=(self.dim,), device=self.device)
        self.v1s[i] = th.zeros((self.dim,), device=self.device)

        self.cur_steps[i] = 0

    def step(self, actions: TEN) -> Tuple[TEN, TEN, TEN, TEN, dict]:
        """
        Vectorized step.
        actions: tensor shape (num_envs, action_dim)
        returns: next_states, rewards, terminated (float tensor), truncated (bool tensor), info dict
        """
        # normalize actions per env
        actions_l2 = (actions ** 2).sum(dim=1, keepdim=True) ** 0.5
        actions_l2 = actions_l2.clamp_min(1.0)
        actions = actions / actions_l2

        # update pursuer
        self.v1s *= 0.75
        self.v1s += actions
        self.p1s += self.v1s * 0.01

        # update evader (stochastic)
        self.v0s *= 0.50
        self.v0s += th.rand(size=(self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.p0s += self.v0s * 0.01

        """reward"""
        distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5
        rewards = self.distances - distances - actions_l2.squeeze(1) * 0.02
        self.distances = distances

        """done / terminated / truncated"""
        self.cur_steps += 1
        # terminated if captured, truncated if hit max steps
        terminated = distances < self.dim
        truncated = (self.cur_steps == self.max_step)

        # reset envs that finished
        for env_i in range(self.num_envs):
            if terminated[env_i] or truncated[env_i]:
                self.reset_env_i(env_i)

        terminated = terminated.type(th.float32)
        truncated = truncated.type(th.bool)

        """next_state"""
        next_states = self.get_state()
        return next_states, rewards, terminated, truncated, dict()

    def get_state(self) -> TEN:
        return th.cat((self.p0s, self.v0s, self.p1s, self.v1s), dim=1)

    @staticmethod
    def get_action(states: TEN) -> TEN:
        # heuristic: pursuer moves towards evader
        states_reshape = states.reshape((states.shape[0], 4, -1))
        p0s = states_reshape[:, 0]
        p1s = states_reshape[:, 2]
        return p0s - p1s


def check_chasing_vec_env():
    env = PointChasingVecEnv(dim=2, env_num=2, sim_device=0)

    reward_sums = [0.0] * env.num_envs
    reward_sums_list = [[] for _ in range(env.num_envs)]

    states, _ = env.reset()
    for _ in range(env.max_step * 4):
        actions = env.get_action(states)
        states, rewards, terminated, truncated, _ = env.step(actions)

        dones = th.logical_or(terminated.type(th.bool), truncated)
        for env_i in range(env.num_envs):
            reward_sums[env_i] += rewards[env_i].item()

            if dones[env_i]:
                print(f"{env.distances[env_i].item():8.4f}    {actions[env_i].detach().cpu().numpy().round(2)}")
                reward_sums_list[env_i].append(reward_sums[env_i])
                reward_sums[env_i] = 0.0

    reward_sums_list = np.array(reward_sums_list, dtype=object)
    print("shape:", [len(l) for l in reward_sums_list])
    print("mean: ", [np.mean(l) if len(l) > 0 else 0.0 for l in reward_sums_list])
    print("std:  ", [np.std(l) if len(l) > 0 else 0.0 for l in reward_sums_list])


if __name__ == "__main__":

    start_time = time.perf_counter()
    t2 = time.perf_counter()
    check_chasing_vec_env()
    t3 = time.perf_counter()
    print(f"check_chasing_vec_env elapsed: {t3 - t2:.4f} seconds")

