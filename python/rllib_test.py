from ray.rllib.algorithms import ppo
import torch
from torch import nn
import os
from ray.tune.logger import pretty_print
import gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch

### create custom actor net ###
class Actor_NN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.hidden_layers = nn.Sequential(
            nn.Linear(obs_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.to_logits = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)
        self._output = None

    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]
        self._output = self.hidden_layers(inputs)
        logits = self.to_logits(self._output)
        return logits, state

    def get_action(self, s):
        """ Returns the action for the model to take in this state, s"""
        ts = torch.tensor(s.astype(np.float32))
        out = self.hidden_layers(ts)
        out = self.to_logits(out)
        return out

    def value_function(self):
        value_out = self.value_branch(self._output)
        return torch.reshape(value_out, [-1])

def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits)
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    return -train_batch[SampleBatch.REWARDS].dot(log_probs)
# env = gym.make("Pendulum-v1")
# actor = Actor_NN(env.observation_space, env.action_space, env.action_space.shape[0], {}, "Pendulum-v1")
# print(env.observation_space.shape)
# obs = torch.rand((1, 3))
# obs_dict = {"obs": obs}

# state = np.array([1.0])
# state = [torch.from_numpy(state)]

# seq_lens = np.ndarray((1,))
# seq_lens = torch.from_numpy(seq_lens)
# print(actor(obs_dict, state, seq_lens)[0]) 
# print(actor.get_action(np.array([0, 0, 0])))

### Create directory for saving RLlib policy ###
policy_path = "{}/policies/".format(os.path.dirname(__file__)) # Define folder path

if not(os.path.isdir(policy_path)):
    os.mkdir(policy_path)

### configure the environment and training ###
ModelCatalog.register_custom_model("Actor_NN", Actor_NN)
config = {
    "model": {
        "custom_model": "Actor_NN"
    },
    "env": "Pendulum-v1",
    "framework": "torch",
    "num_gpus": 1,
    "num_workers": 10,
    "lr": 0.0003,
    "lambda": 0.1,
    "gamma": 0.95,
}

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)

### Train ###

Trainer = ppo.PPO(config=ppo_config)
# status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
# n_iter = 50
# for n in range(n_iter):
#     result = Trainer.train()
#     chkpt_file = Trainer.save(policy_path)
#     print(status.format(
#             n + 1,
#             result["episode_reward_min"],
#             result["episode_reward_mean"],
#             result["episode_reward_max"],
#             result["episode_len_mean"],
#             chkpt_file
#             ))

### Test v1 ###
Trainer.restore("/home/medicalrobotics/MASS_EXO/python/policies/checkpoint_000050")
env = gym.make("Pendulum-v1")

# print("=================Loading from RLlib checkpoint===================")

# n_step = 10
# for n in range(n_step):
#     sum_reward = 0
#     state = env.reset()
#     done=False

#     while not done:
#         action = Trainer.compute_single_action(state)
#         state, reward, done, info = env.step(action)
#         sum_reward += reward
#         env.render()

#     print("cumulative reward", sum_reward)



### Test v2 ###
Tensor = torch.cuda.FloatTensor
print("=================Loading from torchscript===================")
# Export policy as a torch model
policy = Trainer.get_policy()
model_path = "/home/medicalrobotics/MASS_EXO/python/policies/torch_pol"
policy.export_model(model_path)
# load torch model
model = torch.jit.load(model_path + "/model.pt")
model.eval()
# actor = Actor_NN(env.observation_space, env.action_space, env.action_space.shape[0], {}, "Pendulum-v1")
# torch.save(model.state_dict(), model_path + "/statedict.pt")
# actor.load_state_dict(torch.load(model_path + "/statedict.pt"))
# actor.eval()

state = np.array([1.0])
state = [Tensor(state)]

seq_lens = np.ndarray((1,))
seq_lens = Tensor(seq_lens)

# run render loop
n_step = 10
for n in range(n_step):
    sum_reward = 0
    s= env.reset()
    ts = Tensor(s.astype(np.float32))
    print(s)
    done=False
    while not done:
        obs_dict = {"obs": ts}
        action = model(obs_dict, state, seq_lens)
        print(action)
        s, reward, done, info = env.step(action[0].cpu().detach().numpy())
        ts = Tensor(s.astype(np.float32))
        sum_reward += reward
        env.render()

    print("cumulative reward", sum_reward)