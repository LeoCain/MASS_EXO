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
        ### check that cuda has initialised correctly ###
        use_cuda = torch.cuda.is_available()
        print(f"============cuda available: {use_cuda}================")

        ### Create directory for saving RLlib policy ###
        self.policy_path = "{}/policies_torch/".format(os.path.dirname(__file__)) # Define folder path

        if not(os.path.isdir(self.policy_path)):
            os.mkdir(self.policy_path)

        ### Initialise the environment and ray tuner ###
        self.metafile_path = "/home/medicalrobotics/MASS_EXO/data/metadata.txt"
        self.tuner = self.Initialise_Tuner()

        ### create lists to store previous rewards, and an epoch counter ###
        self.min_rewards = []
        self.max_rewards = []
        self.mean_rewards = []
        self.epochs = 0

        ### initialise ray? ###
    
    def Initialise_Tuner(self):
        """
        Sets up the tuner config: Defines hyperparameter space to tune,
        objective to optimise, environment, search algorithm and scheduler
        :return: tuner object
        """
        PBT = tune.PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations={
                "lambda": [0.9, 0.95, 1.0],
                "lr": tune.uniform(1e-3, 1e-5),
                "num_sgd_iter": tune.randint(1, 30),
                "sgd_minibatch_size": [64, 128, 256, 512],
                "train_batch_size": tune.randint(2000, 160000),
            },
        )

        tuner = tune.Tuner(
            "PPO",
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=PBT,
                num_samples=1,
            ),
            param_space = {
                "env": MASS_env,
                "env_config": {
                    "meta_file": self.metafile_path,
                },
                "framework": "torch",
                "num_gpus": 1,
                "num_workers": 5,
                "num_envs_per_worker": 1,
                "num_gpus_per_worker": 0.2,
                "lr": 0.0001,
                "gamma": 0.999,
                "train_batch_size": 4000,
                "lambda": 1.0,
            }
        )

        return tuner

    def Tune_Params(self):
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


# def debug():
#     ben = Exo_Trainer()
#     ben.Train_Exo(1000)

# debug()