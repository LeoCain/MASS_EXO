# Ensure we can still import pymss and Model
import sys
from tabnanny import verbose
sys.path.append("/home/medicalrobotics/Anton/MASS_EXO/python")
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
from ray.rllib.utils.schedules.polynomial_schedule import PolynomialSchedule
# hyperparameter tuning dependencies
from ray import air, tune
from ray.tune.analysis import experiment_analysis, ExperimentAnalysis
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.air.config import RunConfig
# Neural Net dependencies (torch)
import torch
from torch import nn
import Model
from Exo_model import Actor_NN
# Other standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import time

class Exo_Trainer():
    def __init__(self, mode='train', restore_agent=False) -> None:
        """
        Initialise the RLlib PPO agent, check cuda is connected to GPU,
        create directory for saving policies.
        :param mode: 'tune' or 'train' mode, tune for tuning hyperparams, train for 
                    training using specified set of hyperparams. 
        """
        ### check that cuda has initialised correctly ###
        use_cuda = torch.cuda.is_available()
        print(f"============cuda available: {use_cuda}================")

        ### Create directory for saving RLlib policy ###
        self.policy_path = "{}/policies/".format(os.path.dirname(__file__)) # Define folder path
        if not(os.path.isdir(self.policy_path)):
            os.mkdir(self.policy_path)

        ### Initialise the environment, agent, tuner, network ###
        # register custom env and NN
        ModelCatalog.register_custom_model("Actor_NN", Actor_NN)
        register_env("MASS_env", MASS_env)
        self.metafile_path = "/home/medicalrobotics/Anton/MASS_EXO/data/metadata_bws.txt"
        self.sim_NN_path = "/home/medicalrobotics/Anton/MASS_EXO/nn/max.pt"
        self.muscle_NN_path = "/home/medicalrobotics/Anton/MASS_EXO/nn/max_muscle.pt"
        if mode == 'tune':
            # tunes hyperparameters - not rigorously tested, but should only be used
            # once reward function and simulation are good and getting decent results.
            # i.e. the actor should work - the tuner is just for further optimisation
            self.tuner = self.Initialise_Tuner()
        elif mode == 'train':
            self.config = {
                "env": MASS_env,
                "env_config": {
                    "meta_file": self.metafile_path,
                    "sim_NN":self.sim_NN_path,
                    "muscle_NN":self.muscle_NN_path,
                },
                "model": {
                    "custom_model": "Actor_NN",
                },
                "framework": "torch",
                "num_gpus": 1.0/7.0,
                "num_workers": 6,
                "num_envs_per_worker": 1,
                "num_gpus_per_worker": 1.0/7.0,
                "lr": 0.00005,
                "lambda": 1.0,
                "gamma": 0.999,
                "horizon": 300,
                "rollout_fragment_length": 200,
                "train_batch_size": 9600,
                "log_level": 'ERROR',
                "clip_param": 0.15,
                "grad_clip": 4,
                # note that this MUST occur:
                # train_batch_size % (num_workers * rollout_fragment_length) == 0
            }
            print(f"============config saved================")
            self.ppo_config = ppo.DEFAULT_CONFIG.copy()
            self.ppo_config.update(self.config)
            print(f"============config updated================")
            self.agent = ppo.PPO(config=self.ppo_config)
            print(f"************{self.ppo_config}*********")
            print(f"============config applied ================")
            if restore_agent:
                pass
        else:
            print("invalid mode entered")

        ### create lists to store previous rewards, and an epoch counter ###
        self.min_rewards = []
        self.max_rewards = []
        self.mean_rewards = []
        self.epochs = 0

    def Initialise_Tuner(self):
        """
        Sets up the tuner config: Defines hyperparameter space to tune,
        objective to optimise, environment, search algorithm and scheduler
        :return: tuner object
        """
        ray.init()
        PopBasedBandit = PB2(
            time_attr="time_total_s",
            
            perturbation_interval=100,   # how often to re-consider chosen params
            quantile_fraction=0.25,     # probability of copying good params to runs with bad params
            
            hyperparam_bounds={  # Specifies bounds to search for best hyperparams within
                # SGD Momentum?
                # "lambda": [0.9, 0.95, 1.0],
                "lr": [1e-3, 1e-5],
                # "momentum":
                # "num_sgd_iter": tune.randint(1, 30),
                # "sgd_minibatch_size": [64, 128, 256, 512],
                "train_batch_size": [3000, 60000],
            },
        )

        tuner = tune.Tuner(
            # might need to add stop case for tuning
            "PPO",
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=PopBasedBandit,
                num_samples=6,  # increase this?
            ),
            run_config = RunConfig(
                name="PB2_nw6_1",
                local_dir = "{}/ray_tune_results".format(os.path.dirname(__file__)),
                verbose=3,
                # sync_config=tune.SyncConfig(upload_dir="s3://..."),
                # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=2),
            ),
            param_space = {
                "env": MASS_env,
                "env_config": {
                    "meta_file": self.metafile_path,
                },
                "framework": "torch",
                # "num_gpus": 1,
                "num_workers": 3,
                "num_envs_per_worker": 1,
                "num_gpus_per_worker": 0.15,
                # "lr": 0.0001,
                "gamma": 0.999,
                # "train_batch_size": 4000,
                "lambda": 1.0,
                "horizon": 300,
                # "log_level": 'INFO',
            }
        )

        return tuner

    def Tune_Params(self):
        """
        Tunes the specified hyperparameters. Then plots the process.
        """
        results = self.tuner.fit()

        print("best hyperparameters: ", results.get_best_result(metric="episode_reward_mean").config)
        # Plot by wall-clock time
        analysis = ExperimentAnalysis("/home/medicalrobotics/ray_results/PPO")
        dfs = analysis.fetch_trial_dataframes()
        # This plots everything on the same plot
        ax = None
        for d in dfs.values():
            ax = d.plot("training_iteration", "episode_reward_mean", ax=ax, legend=False)

        plt.xlabel("iterations")
        plt.ylabel("mean episode reward")

        print('best config:', analysis.get_best_config("episode_reward_mean"))

    def Train_Exo(self, n: int):
        """
        Train the agent for n iterations
        """
        status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
        n_iter = n
        for n in range(n_iter):
            # print(n)
            result = self.agent.train()
            chkpt_file = self.agent.save(self.policy_path)
            self.epochs += 1
            self.plot_reward(result["episode_reward_min"], result["episode_reward_mean"], result["episode_reward_max"])
            # agent is stopped and reloaded every 25 epochs because RLlib rollout workers have a memory leak (as of 2022)
            if n%25==0 and not n==0:
                print("========stopping...===========")
                self.agent.stop()
                time.sleep(1)
                print("========stopped, reloading...===========")
                self.agent = ppo.PPO(config=self.ppo_config)
                print("========Reloaded===========")
                self.Restore_Agent(chkpt_file)
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
        Retrieves the action for the given state, by using the restored agent.
        used mostly when it is called by render_exo.cpp
        """
        return self.agent.compute_single_action(state)
    
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
        plt.legend(loc='upper left')
        plt.savefig("/home/medicalrobotics/Anton/MASS_EXO/Exo_agent/Plots/RewardPlot_torch.png")


def debug():
    ben = Exo_Trainer('train')
    # ben.Tune_Params()
    ben.Train_Exo(5000)
    # ben.Restore_Agent("/home/medicalrobotics/MASS_EXO/Exo_agent/policies_torch/checkpoint_000120")

if __name__ == "__main__":
    debug()