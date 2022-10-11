# Ensure we can still import pymss and Model
import sys
sys.path.append("/home/medicalrobotics/MASS_EXO/python")
# DRL-related imports:
from traceback import print_tb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import Model
from torchsummary import summary
from ray.rllib.env.env_context import EnvContext
# Standard mathmatical libraries
import numpy as np
import time
# Environment building:
from gym import Env
from gym.spaces import Discrete, Box
# utilisation of c++ functions (EnvManager.cpp):
import pymss

class MASS_env(Env):
    """
    Class used to convert c++ MASS and DART environment into a gym.Env environment
    """
    def __init__(self, config: EnvContext):
        """
        Initialises the elements relevant to the environment:
        State Space
        Action Space
        Initial State
        Episode Length
        """
        print("==================if you can see me i'M ALIVE==============")
        ### Setup env via EnvManager.cpp file ###
        meta_file = config["meta_file"]
        self.num_env_threads = 1
        self.sim_env = pymss.pymss(meta_file,self.num_env_threads)

        ### Setting Up Action Space ###
        max_T = 80  # Maximum possible joint torque, Nm

        T_limit = np.array(
            [
                max_T,      # L hip torque
                max_T,      # L knee torque
                max_T,      # R hip torque
                max_T,      # R knee torque
            ]
        )
        self.action_space = Box(-np.float32(T_limit), np.float32(T_limit))

        ### Setting Up Observation Space ###
        # (actual motion, gait stage)
        # == ((pos_links_COM<x,y,z>, vel_links_COM<x,y,z>, gait_cycle_progress))
        state_dims = self.sim_env.GetNumState() # Dimension of the human model state description (pos, vel, gait stage) 
        obs_low_limit = np.full(state_dims, -100)
        obs_high_limit = np.full(state_dims, 100)
        self.observation_space = Box(np.float32(obs_low_limit), np.float32(obs_high_limit))

        ### Load human control NNs ###
        self.sim_NN = self.load_sim_NN("/home/medicalrobotics/MASS_EXO/nn_crip/max.pt")
        self.muscle_NN = self.load_muscle_NN("/home/medicalrobotics/MASS_EXO/nn_crip/max_muscle.pt")
        
        ### Set environment-specific global vars ###
        self.num_simulation_Hz = self.sim_env.GetSimulationHz()
        self.num_control_Hz = self.sim_env.GetControlHz()
    
        use_cuda = torch.cuda.is_available()
        print(f"============{use_cuda}================")
        self.Tensor = torch.cuda.FloatTensor

        ### Call the environment reset, initialise the state ###
        self.state = self.reset()

    def step(self, action):
        """
        Applies the given action snd steps the simulation appropriately
        :param action: The action to apply, as defined by the action space
        :return: (resultant state, resultant reward, whether game is now terminal or not, some info[not used])
        """
        ### Apply action to environment (actuate exo) ###
        T_LHip, T_LKnee, T_RHip, T_RKnee = action
        self.set_joint_torques(T_LHip, T_LKnee, T_RHip, T_RKnee)
  
        ### Step to next state ###
        self.MASS_step()

        ### Record relevant information ###
        self.state = self.sim_env.GetStates()[0]
        done = bool(self.sim_env.IsEndOfEpisodes()[0])

        ### Define Reward ###
        r_T = ((abs(T_LHip) + abs(T_LKnee) + abs(T_RHip) + abs(T_RKnee))/4)
        reward = 0 - (1/(self.sim_env.GetRewards()[0])) - (r_T/30) # Theoretical max reward of 0 - still needs tuning

        return self.state, reward, done, dict()

    def render(self, mode=''):
        """
        Visually renders the state
        """
        # TODO: The plan is to make another .so file, with the contents of window.cpp
        # Then I will try calling window.draw() here  

    def reset(self):
        """
        Resets the simulation
        """
        # self.set_joint_torques(0, 0, 0, 0)
        self.sim_env.Resets(False)
        return self.sim_env.GetStates()[0]

    def MASS_step(self):
        """
        Steps the DART simulation using MASS functions and specifications
        """
        ### load pos target from simNN and apply to env ###
        num_steps_per_cntrl = self.num_simulation_Hz // self.num_control_Hz
        p_target = self.sim_NN.get_action(self.state)
        p_target = np.array([p_target]) # Change to nested list format for compatibility with SetActions function
        self.sim_env.SetActions(p_target)

        ### load activations from muscleNN and apply to env ###
        inference_per_sim = 2
        
        # mt = self.Tensor(self.sim_env.GetMuscleTorques())
        for i in range(num_steps_per_cntrl//inference_per_sim):
            mt = self.Tensor(self.sim_env.GetMuscleTorques())
            dt = self.Tensor(self.sim_env.GetDesiredTorques())

            activations = self.muscle_NN(mt,dt).cpu().detach().numpy()
            self.sim_env.SetActivationLevels(activations)
            self.sim_env.Steps(inference_per_sim)

    def set_joint_torques(self, T_LHip, T_LKnee, T_RHip, T_RKnee):
        """
        Sets the torques applied to each of the exo joints
        :param T_LHip: Left hip torque to be applied
        :param T_RHip: Right hip torque to be applied
        :param T_LKnee: Left knee torque to be applied
        :param T_RKnee: Right knee torque to be applied
        """
        self.sim_env.SetLHipTs(T_LHip); 
        self.sim_env.SetRHipTs(T_RHip); 
        self.sim_env.SetLKneeTs(T_LKnee); 
        self.sim_env.SetRKneeTs(T_RKnee); 

    def load_sim_NN(self, path):
        """
        Loads the position target NN model
        :param path: The path to the saved weights for this net
        :return: The NN with weights loaded
        """
        num_states = self.sim_env.GetNumState()
        num_actions = self.sim_env.GetNumAction()
        NN = Model.SimulationNN(num_states, num_actions)
        NN.load(path)
        return NN
    
    def load_muscle_NN(self, path):
        """
        loads muscle activation NN model
        :param path: The path to the saved weights for this net
        :return: The NN with weights loaded
        """
        muscle_related_dofs = self.sim_env.GetNumTotalMuscleRelatedDofs()
        num_actions = self.sim_env.GetNumAction()
        num_muscles = self.sim_env.GetNumMuscles()
        NN = Model.MuscleNN(muscle_related_dofs, num_actions, num_muscles)
        NN.load(path)
        return NN
        


def debug_test():
    """
    Function purely for debugging and checking that the env is working as desired
    """
    env = MASS_env("/home/medicalrobotics/MASS_EXO/data/metadata.txt")
    done = False
    i = 0
    sim_time = time.time()
    while (not done):
        start = time.time()
        state, reward, done, _ = env.step([0, 0, 0, 0])
        i += 1
        # while (time.time()-start < 0.030303):
        #     continue
        print(f"state {i}:\n    reward: {reward}\n    done: {done}\n    time: {state[-1]}")
    sim_time = time.time()-sim_time
    print(sim_time)
    # print(env.observation_space)
    # print(env.observation_space.shape)
    # print(env.observation_space.low)
    # print(env.observation_space.high)
    # print(env.action_space)
    # print(env.action_space.high)
    # print(env.action_space.low)
    # print(env.action_space.shape)   # (4,)
    # print('#######################')
    # print(env.state_dim)

# debug_test()
        