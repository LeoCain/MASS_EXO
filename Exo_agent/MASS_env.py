# Ensure we can still import pymss and Model
import sys
sys.path.append("/home/medrobotics/MASSExo/MASSMerge/MASS_EXO/python")
sys.path.append("/home/medrobotics/MASSExo/MASSMerge/MASS_EXO")
# DRL-related imports:
from traceback import print_tb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import Model
from ray.rllib.env.env_context import EnvContext
# Standard mathmatical libraries
import numpy as np
import time
import matplotlib.pyplot as plt
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
        # print("==================if you can see me I'M ALIVE==============")
        ### Setup env via EnvManager.cpp file ###
        meta_file = config["meta_file"]
        sim_NN = config["sim_NN"]
        muscle_NN = config["muscle_NN"]
        self.num_env_threads = 1
        self.sim_env = pymss.pymss(meta_file,self.num_env_threads)
        # Keep track of number of resets (= to num episodes)

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
        self.action_space = Box(-np.float64(T_limit), np.float64(T_limit), dtype=np.float64)

        ### Setting Up Observation Space ###
        # (actual motion, gait stage)
        # == ((pos_links_COM<x,y,z>, vel_links_COM<x,y,z>, gait_cycle_progress))
        # angle_dims = self.sim_env.GetLegJointAngles()[0].size
        state_dims = self.sim_env.GetNumState() # Dimension of the human model state description (pos, vel, gait stage) 
        obs_low_limit = np.full(state_dims, -1000.0)
        obs_high_limit = np.full(state_dims, 1000.0)
        self.observation_space = Box(np.float64(obs_low_limit), np.float64(obs_high_limit), dtype=np.float64)

        ### Load human control NNs ###
        self.sim_NN = self.load_sim_NN(sim_NN)
        self.muscle_NN = self.load_muscle_NN(muscle_NN)
        
        ### Set environment-specific global vars ###
        self.num_simulation_Hz = self.sim_env.GetSimulationHz()
        self.num_control_Hz = self.sim_env.GetControlHz()

        ### Declare trackables (variables I'm interested in tracking) ###
        self.prevT_LHip, self.prevT_LKnee, self.prevT_RHip, self.prevT_RKnee =\
            0, 0, 0, 0
        self.prev_traj_r = 0
        self.r_T_tot = 0
        self.r_dT_tot = 0

        self.step_num = 0
        self.num_eps = -1   # set to -1 because initial reset will iterate this

        # keep track of original benjaSIM reward
        self.orig_r_list = []
        self.orig_r = 0 

        ### Call the environment reset, initialise the state ###
        self.state = self.reset()

        ### Check GPU access, define GPU tensor ###
        use_cuda = torch.cuda.is_available()
        print(f"============{use_cuda}================")
        self.Tensor = torch.cuda.FloatTensor

    def step(self, action):
        """
        Applies the given action snd steps the simulation appropriately
        :param action: The action to apply, as defined by the action space
        :return: (resultant state, resultant reward, whether game is now terminal or not, some info[not used])
        """
        # if self.num_eps >= 75:
        #     if self.num_eps == 75:
        #         print("============= Switched =============")
        #     apply_action = True

        # else:
        #     apply_action = False

        ### Apply action to environment (actuate exo) ###
        T_LHip, T_LKnee, T_RHip, T_RKnee = action
        self.set_joint_torques(T_LHip, T_LKnee, T_RHip, T_RKnee)
  
        ### Step to next state ###
        self.MASS_step()
        self.state = self.sim_env.GetStates()[0]
        
        ### Handle terminal states ###
        # Check if NaN values have made it through
        done = np.any(np.isnan(self.state)) or bool(self.sim_env.IsEndOfEpisodes()[0])
        
        fall_cost = 0
        if done:    # Handle issue where NaN get through to RLlib worker
            self.state = np.zeros_like(self.state)
            if not(self.step_num >= 300):
                fall_cost = 0
            return self.state, fall_cost, done, {}
        
        ### Define Reward Components ###
        # reward due to torque magnitude
        r_T = abs(T_LHip/80) + abs(T_LKnee/80) + \
            abs(T_RHip/80) + abs(T_RKnee/80)
        # reward due to change in torque magnitude
        dL_hip = abs(self.prevT_LHip - T_LHip)/160
        dL_knee = abs(self.prevT_LKnee - T_LKnee)/160
        dR_hip = abs(self.prevT_RHip - T_RHip)/160
        dR_knee = abs(self.prevT_RKnee - T_RKnee)/160
        r_dT = dL_hip + dL_knee + dR_hip + dR_knee + \
            max(dL_hip, dL_knee, dR_hip, dR_knee)
        # map torque rewards from [0, 4] -> {smaller better} 
        # to [0, 1] -> {larger better}
        r_T_map = np.exp(-r_T)
        r_dT_map = np.exp(-2*r_dT)
        # print(r_dT, r_dT_map)
        # reward due to closesness to desired trajectory
        leg_traj_r = self.sim_env.GetGaitRewards()[0]
        traj_r = self.sim_env.GetRewards()[0]
        
        ### Define Full reward ###
        reward = 0.1*traj_r + 0.3*leg_traj_r + 0.25*r_T_map + 0.35*r_dT_map
       
        # original benjaSIM reward
        self.orig_r += traj_r

        # update tracked values
        self.prevT_LHip = T_LHip
        self.prevT_LKnee = T_LKnee
        self.prevT_RHip = T_RHip
        self.prevT_RKnee = T_RKnee
        self.prev_traj_r = traj_r
        self.step_num += 1
        self.r_T_tot += r_T
        self.r_dT_tot += r_dT

        return self.state, reward, done, {}

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
        ### Append original reward value for plotting ###
        self.orig_r_list.append(self.orig_r)

        ### print info regarding episode ###
        if not (self.step_num==0) and self.num_eps%40 == 0:
            avg_r_T = self.r_T_tot/self.step_num
            avg_r_dT = self.r_dT_tot/self.step_num
            print(f"avg r_T, r_dT = {avg_r_T}, {avg_r_dT}")

        ### iterate/reset trackables ###
        self.num_eps += 1
        self.orig_r = 0 
        self.step_num = 0
        self.r_T_tot = 0
        self.r_dT_tot = 0

        ### Reset the environment ###
        # True indicates that benjaSIM will start in a randomised pose
        self.sim_env.Resets(False)
        
        ### Retrieve new start state ###
        state = self.sim_env.GetStates()[0]
        return state

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
        
    def Plot_Original_Reward(self):
        """
        Live plots the agent's performance as according to the
        original reward so that it can be compared to benjaSIM 
        training runs.
        """
        ### average every group of 50 episode rewards ###
        avg_r = []
        curr_tot = 0
        i = 0
        while i<len(self.orig_r_list):
            curr_tot += self.orig_r_list[i]
            i += 1
            if i % 50 == 0 and not(i == 0):
                avg_r.append(curr_tot/50)
                curr_tot = 0
        ### plot average 'blocks' of episode rewards ###
        num_blocks = np.linspace(0, len(avg_r) - 1, len(avg_r))
        plt.clf()
        plt.plot(num_blocks, avg_r, color="blue")

        plt.title("Mean Episode [original] Reward")
        plt.xlabel("Number of Episodes {*1000}")
        plt.ylabel("Mean Reward")
        plt.savefig("/home/medicalrobotics/MASS_EXO/Exo_agent/Plots/orig_reward.png")


def debug_test():
    """
    Function purely for debugging and checking that the env is working as desired
    """
    env = MASS_env({
        "meta_file":"/home/medicalrobotics/MASS_EXO/data/metadata.txt",
        "sim_NN":"/home/medicalrobotics/MASS_EXO/nn_norm/max.pt",
        "muscle_NN":"/home/medicalrobotics/MASS_EXO/nn_norm/max_muscle.pt",
        })
    j = 0
    while j < 10:
        tot_r = 0
        state = env.reset()
        done = False
        i = 0
        sim_time = time.time()
        while (not done):
            start = time.time()
            state, reward, done, _ = env.step([80, 80, 80, 80])
            tot_r += reward
            i += 1
            # while (time.time()-start < 0.030303):
            #     continue
            # print(f"state {i}:\n    reward: {reward}\n    done: {done}\n    time: {state[-1]}")
        sim_time = time.time()-sim_time
        print(tot_r, i)
        # print(sim_time)
        j += 1
        
if __name__ == "__main__":
    debug_test()
        