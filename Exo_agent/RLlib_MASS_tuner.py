# Ensure we can still import pymss and Model
import sys
sys.path.append("/home/medicalrobotics/MASS_EXO/python")
# Environment building:
import gym
import pymss    # access to c++ libraries (envmanager.cpp)
from MASS_env import MASS_env
# RLlib related dependencies (DRL library)
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env
import ray
import json
# hyperparameter tuning dependencies
from ray import air, tune
# Neural Net dependencies (torch)
import torch
from torch import nn
import Model
# Other standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

class Exo_Trainer():
    def __init__(self) -> None:
        """
        Initialise the RLlib PPO agent, check cuda is connected to GPU,
        create directory for saving policies.
        """
        print('ben was here')
        ### check that cuda has initialised correctly ###
        use_cuda = torch.cuda.is_available()
        print(f"============cuda available: {use_cuda}================")

        ### Create directory for saving RLlib policy ###
        self.policy_path = "{}/policies/".format(os.path.dirname(__file__)) # Define folder path

        if not(os.path.isdir(self.policy_path)):
            os.mkdir(self.policy_path)

        ### Plotting setup ###
        self.min_rewards = []
        self.max_rewards = []
        self.mean_rewards = []
        self.epochs = 0
        ### Initialise the environment and agent ###
        register_env("MASS_env", MASS_env)
        self.metafile_path = "/home/medicalrobotics/MASS_EXO/data/metadata_crip.txt"
        self.config = {
            "env": "MASS_env",
            "env_config": {
                "meta_file": self.metafile_path,
            },
            "framework": "torch",
            "num_gpus": 1,
            "num_gpus_per_worker": 1,
            "num_workers": 1,
            "lr": 0.0003,
            "lambda": 0.1,
            "gamma": 0.95,
        }
        print(f"============config saved================")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(self.config)
        print(f"============config updated================")
        self.agent = ppo.PPO(config=ppo_config)
        print(f"============config applied ================")
    
    def Train_Exo(self, n: int):
        """
        Train the agent for n iterations
        """
        status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
        n_iter = n
        for n in range(n_iter):
            result = self.agent.train()
            chkpt_file = self.agent.save(self.policy_path)
            self.epochs += 1
            self.plot_reward(result["episode_reward_min"], result["episode_reward_mean"], result["episode_reward_max"])
            print(status.format(
                    n + 1,
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"],
                    chkpt_file
                    ))

    def Restore_Agent(self, path):
        """
        Restore the agent to one of the saved policy files
        """
        self.agent.restore(path)
        print(f"============agent restored================")
    
    def get_action(self, state):
        """
        Retrieves the action for the given state, by using the restored agent
        """
        return self.agent.compute_single_action(state) # right format for c++?
    
    def plot_reward(self, min, mean, max):
        """
        Plots the reward and saves the figure, overwriting previous one
        """
        self.min_rewards.append(min)
        self.max_rewards.append(max)
        self.mean_rewards.append(mean)
        epoch_space = np.linspace(1, self.epochs, self.epochs)

        plt.clf()
        plt.plot(epoch_space, self.min_rewards, label = "Min episode reward", color="blue")
        plt.plot(epoch_space, self.max_rewards, label = "Max episode reward", color="red")
        plt.plot(epoch_space, self.mean_rewards, label = "Mean episode reward", color="orange")
        plt.title("Episode Reward vs Number of Epochs (Torch)")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig("/home/medicalrobotics/MASS_EXO/Exo_agent/Plots/RewardPlot_torch.png")


def debug():
    ben = Exo_Trainer()
    ben.Train_Exo(1000)

debug()