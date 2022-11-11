import torch
from torch import nn
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch

"""
File containing the custom neural net for the torque actor.
"""

class Actor_NN(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        # link to parent RLlib and torch classes
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # define hidden layers
        fc1 = nn.Linear(obs_space.shape[0], 256)
        fc2 = nn.Linear(256, 256)
        fc3 = nn.Linear(256, 256)
        fc4 = nn.Linear(256, 256)
        # configure hidden layers
        self.hidden_layers = nn.Sequential(
            fc1,
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            fc2,
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            fc3,
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            fc4,
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # output layers
        self.to_logits = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)

        # initialise weights with xavier uniform initialiser:
        torch.nn.init.xavier_uniform_(fc1.weight)
        torch.nn.init.xavier_uniform_(fc2.weight)
        torch.nn.init.xavier_uniform_(fc3.weight)
        torch.nn.init.xavier_uniform_(fc4.weight)
        torch.nn.init.xavier_uniform_(self.to_logits.weight)

        # set neuron bias to 0
        fc1.bias.data.zero_()
        fc2.bias.data.zero_()
        fc3.bias.data.zero_()
        fc4.bias.data.zero_()

        # I think this needs to be included to make RLlib happy
        self._output = None

    def forward(self, input_dict, state, seq_lens):
        """
        Evaluates the forward call of the net.
        """
        inputs = input_dict["obs"]
        self._output = self.hidden_layers(inputs)
        logits = self.to_logits(self._output)

        return logits, state

    def value_function(self):
        """
        Evaluates the value function of the call.
        """
        value_out = self.value_branch(self._output)
        return torch.reshape(value_out, [-1])