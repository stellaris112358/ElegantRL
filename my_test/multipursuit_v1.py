import torch
import numpy as np
import numpy.random as rd
import time
import random

TargetReturnDict = {
    2: 5.5,
    3: 3.5,
    4: 2.5,
    8: -1.5,  # -1.37
}

def set_seed(seed: int):
    """设置随机种子以便复现（尽量保证可重复，CUDA 下可能仍有少量不确定性）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ChasingVecEnv:
    def __init__(self, dim=2, env_num=32, n_chasers=3, device_id=0):
        self.dim = dim
        self.init_distance = 8.0
        self.n_chasers = n_chasers

        # reset
        self.p0s = None  # evader position, shape (env_num, dim)
        self.v0s = None  # evader velocity, shape (env_num, dim)
        self.p1s = None  # chasers positions, shape (env_num, n_chasers, dim)
        self.v1s = None  # chasers velocities, shape (env_num, n_chasers, dim)

        self.distances = None  # min distance between evader and any chaser, shape (env_num,)
        self.cur_steps = None  # a tensor of current step number

        """env info"""
        self.env_name = "ChasingVecEnv"
        # state contains: p0, v0, p1s(flat), v1s(flat)
        self.state_dim = self.dim * (2 + 2 * self.n_chasers)
        # action is flattened actions for all chasers: n_chasers * dim
        self.action_dim = self.dim * self.n_chasers
        self.max_step = 2**10
        self.if_discrete = False
        self.target_return = TargetReturnDict.get(dim, 0.0)

        self.env_num = env_num
        self.device = torch.device("cpu" if device_id == -1 else f"cuda:{device_id}")

    def reset(self):
        self.p0s = torch.zeros(
            (self.env_num, self.dim), dtype=torch.float32, device=self.device
        )
        self.v0s = torch.zeros(
            (self.env_num, self.dim), dtype=torch.float32, device=self.device
        )
        self.p1s = torch.zeros(
            (self.env_num, self.n_chasers, self.dim), dtype=torch.float32, device=self.device
        )
        self.v1s = torch.zeros(
            (self.env_num, self.n_chasers, self.dim), dtype=torch.float32, device=self.device
        )

        self.cur_steps = torch.zeros(
            self.env_num, dtype=torch.float32, device=self.device
        )

        for env_i in range(self.env_num):
            self.reset_env_i(env_i)

        # distances: min over chasers of euclidean distance to evader
        distances_all = ((self.p0s.unsqueeze(1) - self.p1s) ** 2).sum(dim=2) ** 0.5
        self.distances = distances_all.min(dim=1).values

        return self.get_state()

    def reset_env_i(self, i):
        # evader around origin
        self.p0s[i] = torch.normal(0, 1, size=(self.dim,), device=self.device)
        self.v0s[i] = torch.zeros((self.dim,), device=self.device)
        # multiple chasers around -init_distance
        self.p1s[i] = torch.normal(-self.init_distance, 1, size=(self.n_chasers, self.dim), device=self.device)
        self.v1s[i] = torch.zeros((self.n_chasers, self.dim), device=self.device)

        self.cur_steps[i] = 0

    def step(self, actions):
        """
        :param actions: [tensor] actions.shape == (env_num, action_dim) flattened OR (env_num, n_chasers, dim)
        :return: next_states [tensor] next_states.shape == (env_num, state_dim)
        :return: rewards [tensor] rewards == (env_num, )
        :return: terminal [tensor] terminal == (env_num, ), terminal = 1. if terminated (captured) else 0.
        :return: truncate [tensor] truncate == (env_num, ), True if truncated by max_step.
        :return: None [None or dict]
        """
        # accept flattened input or already shaped
        if actions.dim() == 2 and actions.shape[1] == self.action_dim:
            actions = actions.view(self.env_num, self.n_chasers, self.dim)
        elif actions.dim() == 3:
            pass
        else:
            raise ValueError("actions must have shape (env_num, action_dim) or (env_num, n_chasers, dim)")

        # normalize each chaser action vector
        actions_l2 = (actions**2).sum(dim=2) ** 0.5  # (env_num, n_chasers)
        actions_l2 = actions_l2.clamp_min(1.0)  # avoid div0
        actions = actions / actions_l2.unsqueeze(2)

        # update chasers
        self.v1s *= 0.75
        self.v1s += actions
        self.p1s += self.v1s * 0.01

        # update evader with stochastic velocity
        self.v0s *= 0.50
        self.v0s += torch.rand(
            size=(self.env_num, self.dim), dtype=torch.float32, device=self.device
        )
        self.p0s += self.v0s * 0.01

        """reward"""
        # distances between evader and each chaser
        distances_all = ((self.p0s.unsqueeze(1) - self.p1s) ** 2).sum(dim=2) ** 0.5  # (env_num, n_chasers)
        distances = distances_all.min(dim=1).values  # (env_num,)
        # action penalty: sum of norms of each chaser's raw action (before normalization)
        action_penalty = actions_l2.sum(dim=1)  # (env_num,)
        rewards = self.distances - distances - action_penalty * 0.02
        self.distances = distances

        """done / terminal / truncate"""
        self.cur_steps += 1  # array

        # terminated if any chaser is within threshold (capture)
        terminal = distances < self.dim  # bool tensor
        # truncated if reach max step
        truncate = (self.cur_steps == self.max_step)  # bool tensor

        # reset environments that are either terminated or truncated
        for env_i in range(self.env_num):
            if terminal[env_i] or truncate[env_i]:
                self.reset_env_i(env_i)

        # match PointChasingVecEnv style: terminal as float32, truncate as bool
        terminal = terminal.type(torch.float32)
        truncate = truncate.type(torch.bool)

        """next_state"""
        next_states = self.get_state()

        return next_states, rewards, terminal, truncate, None

    def get_state(self):
        # flatten chasers' positions/velocities
        p1s_flat = self.p1s.reshape(self.env_num, self.n_chasers * self.dim)
        v1s_flat = self.v1s.reshape(self.env_num, self.n_chasers * self.dim)
        return torch.cat((self.p0s, self.v0s, p1s_flat, v1s_flat), dim=1)

    @staticmethod
    def get_action(states, n_chasers=3, dim=2):
        """
        produce simple heuristic actions: chasers move towards evader.
        states shape: (env_num, state_dim)
        returns flattened actions shape: (env_num, n_chasers * dim)
        """
        env_num = states.shape[0]
        # indices in flattened state
        p0 = states[:, :dim]  # (env_num, dim)
        p1s_flat = states[:, 2 * dim : 2 * dim + n_chasers * dim]
        p1s = p1s_flat.reshape(env_num, n_chasers, dim)
        actions = p0.unsqueeze(1) - p1s  # (env_num, n_chasers, dim)
        return actions.reshape(env_num, n_chasers * dim)


def check_chasing_vec_env():
    env = ChasingVecEnv(dim=2, env_num=4, n_chasers=3, device_id=0)
    total_steps = env.max_step * 4  ## 4096步
    print("Env num:", env.env_num, "n_chasers:", env.n_chasers)

    reward_sums = [0.0] * env.env_num  # episode returns
    reward_sums_list = [[] for _ in range(env.env_num)]

    states = env.reset()
    for _ in range(total_steps):   
        actions = env.get_action(states, n_chasers=env.n_chasers, dim=env.dim)
        states, rewards, terminal, truncate, _ = env.step(actions)

        # combine terminal and truncate for bookkeeping
        dones = torch.logical_or(terminal.type(torch.bool), truncate)

        for env_i in range(env.env_num):
            reward_sums[env_i] += rewards[env_i].item()

            if dones[env_i]:
                # print min distance and actions for that env
                # act = actions[env_i].reshape(env.n_chasers, env.dim).detach().cpu().numpy().round(2)
                # print(f"{env.distances[env_i].item():8.4f}    {act}")
                reward_sums_list[env_i].append(reward_sums[env_i])
                reward_sums[env_i] = 0.0

    reward_sums_list = np.array([np.array(l) for l in reward_sums_list], dtype=object)
    print("shape per env (variable episodes):", [len(l) for l in reward_sums_list])
    print("mean per env: ", [np.mean(l) if len(l) > 0 else 0.0 for l in reward_sums_list])
    print("std per env:  ", [np.std(l) if len(l) > 0 else 0.0 for l in reward_sums_list])


if __name__ == "__main__":
    set_seed(42)
    start_time = time.time()
    check_chasing_vec_env()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")