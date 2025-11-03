from elegantrl.train.config import Config,get_gym_env_args
from elegantrl.train.run import train_agent
from my_test.singlepursuitEnv import PointChasingVecEnv
from elegantrl.agents.AgentPPO import AgentPPO
import os
import sys
import gymnasium as gym
import numpy as np


def train_ppo_for_singlepursuit(gpu_id=0):
    agent_class = AgentPPO  # DRL algorithm
    env_class = PointChasingVecEnv
    env_temp = PointChasingVecEnv(dim=2, env_num=8, sim_device=0, seed=42)

    # 从 env_temp 提取信息，兼容不同命名（num_envs / env_num）
    env_name = getattr(env_temp, "env_name")
    state_dim = int(getattr(env_temp, "state_dim"))
    action_dim = int(getattr(env_temp, "action_dim"))
    num_envs = int(getattr(env_temp, "num_envs"))

    env_args = {
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": False,
        "num_envs": num_envs,
    }

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e6)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.95  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args,if_single_process=True)  # train the agent

if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_ppo_for_singlepursuit(gpu_id=GPU_ID)
