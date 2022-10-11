from tkinter.tix import Select
import numpy as np
import matplotlib.pyplot as plt
import os

class Plotter():
    """ 
    A class used to plot benjaSIM exo torques, joint angles, and 
    muscle group activations
    """
    def __init__(self):
        """
        Initialises lists for storing plot information:
        - Exo torques
        - Joint angles
        - muscle group activations
        - Gait phase
        """
        ### Exo Torques ###
        self.t_l_hip = []; self.t_l_knee = []
        self.t_r_hip = []; self.t_r_knee = []

        ### Joint Angles ###
        # Actual:
        self.l_hip_angle = []; self.l_knee_angle = []
        self.r_hip_angle = []; self.r_knee_angle = []
        # Desired: TODO: update desired angles
        self.d_l_hip_angle = []; self.d_l_knee_angle = []
        self.d_r_hip_angle = []; self.d_r_knee_angle = []

        ### Muscle Group Activations ###
        # Left Leg
        self.l_h_flex = []; self.l_h_ext = []; self.l_h_abd = []
        self.l_h_add = []; self.l_h_extrot = []; self.l_h_introt = []
        self.l_k_flex = []; self.l_k_ext = []
        # Right Leg
        self.r_h_flex = []; self.r_h_ext = []; self.r_h_abd = []
        self.r_h_add = []; self.r_h_extrot = []; self.r_h_introt = [] 
        self.r_k_flex = []; self.r_k_ext = []

        ### Gait Phase ###
        self.num_wraparounds = 0
        self.gait_phase = []
        
        ### Initialise all plots (so can be cleared from funcs) ###
        # fig, self.torque_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )
        # fig, self.angle_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )
        # fig, self.angle_comp_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )
        # fig, self.knee_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )
        # fig, self.hip_flex_ext_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )
        # fig, self.hip_add_abd_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )
        # fig, self.hip_int_ext_rot_axs = plt.subplots(
        #     4, sharex=True, sharey=True, constrained_layout = True
        #     )

    def Update_Torques(self, exo_torques):
        """
        Appends the inputted list of torques to their corresponding
        lists.
        :param exo_torques: list of applied exo torques in the order:
                [L hip T, L knee T, R hip T, R knee T]
        """
        self.t_l_hip.append(exo_torques[0])
        self.t_l_knee.append(exo_torques[1])
        self.t_r_hip.append(exo_torques[2])
        self.t_r_knee.append(exo_torques[3])
            
    def Update_Angles(self, joint_angles_act, joint_angles_ref):  #TODO update actual angles
        """
        Appends the inputted list of joint angles to their corresponding
        lists.
        :param joint_angles_act: list of actual observed joint angles in the order:
                [L hip, L knee, R hip, R knee]
        :param joint_angles_ref: list of ref joint angles in the order:
                [L hip, L knee, R hip, R knee]
        """
        # Actual angles
        self.l_hip_angle.append(joint_angles_act[0]) 
        self.l_knee_angle.append(joint_angles_act[1]) 
        self.r_hip_angle.append(joint_angles_act[2]) 
        self.r_knee_angle.append(joint_angles_act[3]) 
        # reference angles
        self.d_l_hip_angle.append(joint_angles_ref[0])
        self.d_l_knee_angle.append(joint_angles_ref[1])
        self.d_r_hip_angle.append(joint_angles_ref[2])
        self.d_r_knee_angle.append(joint_angles_ref[3])
    
    def Update_Activations(self, activations):
        """
        Appends the inputted list of activations to their corresponding
        muscle group lists.
        :param activations: list of muscle group activations in the order:
            {LHFlex,LHExt,LHAbd,LHAdd,LHExtRot,LHIntRot,LKFlex,LKExt,
            RHFlex,RHExt,RHAbd,RHAdd,RHExtRot,RHIntRot,RKFlex,RKExt,} 
        """
        activation_grp_lst = [
            self.l_h_flex, self.l_h_ext, self.l_h_abd, self.l_h_add, 
            self.l_h_extrot, self.l_h_introt, self.l_k_flex, self.l_k_ext,
            self.r_h_flex, self.r_h_ext, self.r_h_abd, self.r_h_add, 
            self.r_h_extrot, self.r_h_introt, self.r_k_flex, self.r_k_ext
        ]

        i = 0
        while (i < len(activation_grp_lst)):
            activation_grp_lst[i].append(activations[i])
            i += 1
    
    def Update_Gait_Phase(self, phase):
        """
        Appends the current phase of gait to the gait phase list.
        Some computation is done to adjust the value, because the
        given phase wraps back around to 0.
        :param phase: the current phase of gait (mapped between
                            [0-1])
        """
        phase = phase[0]
        # Check if a wraparound has occurred
        if len(self.gait_phase) < 1:
            prev_phase = 0
        else:
            prev_phase = self.gait_phase[-1]
        if self.num_wraparounds > 0:
            prev_phase = prev_phase%self.num_wraparounds
        if prev_phase > phase:
            self.num_wraparounds += 1   # add to wraparounds if yes
        # Compute current phase as number of wraps + gait_phase
        # print(f"Summed phase: {curr_phase}, phase: {phase}")
        curr_phase = self.num_wraparounds + phase
        self.gait_phase.append(curr_phase)

    def Update_All_Exo(self, exo_torques, joint_angles_act, 
                        joint_angles_ref, activations, phase):
        """
        Appends all inputted lists to their corresponding lists
        by calling individual update functions. Also updates the gait_phase
        list, which is used as the x-axis for all the plots.
        :param exo_torques: list of applied exo torques in the order:
                [L hip T, L knee T, R hip T, R knee T]
        :param joint_angles_act: list of actual joint angles in the order:
                [L hip, L knee, R hip, R knee]
        :param joint_angles_ref: list of ref joint angles in the order:
                [L hip, L knee, R hip, R knee]
        :param activations: list of muscle group activations in the order:
            {LHFlex,LHExt,LHAbd,LHAdd,LHExtRot,LHIntRot,LKFlex,LKExt,
            RHFlex,RHExt,RHAbd,RHAdd,RHExtRot,RHIntRot,RKFlex,RKExt,} 
        :param phase: current progression through the gait phase
        """
        self.Update_Torques(exo_torques)
        self.Update_Angles(joint_angles_act, joint_angles_ref)
        self.Update_Activations(activations)
        self.Update_Gait_Phase(phase)

    def Update_All_SIM(self, joint_angles_act, joint_angles_ref, activations, phase):
        """
        Appends all inputted lists to their corresponding lists
        by calling individual update functions. Also updates the gait_phase
        list, which is used as the x-axis for all the plots.
        :param joint_angles: list of joint angles in the order:
                [L hip, L knee, R hip, R knee]
        :param activations: list of muscle group activations in the order:
            {LHFlex,LHExt,LHAbd,LHAdd,LHExtRot,LHIntRot,LKFlex,LKExt,
            RHFlex,RHExt,RHAbd,RHAdd,RHExtRot,RHIntRot,RKFlex,RKExt,} 
        :param phase: current progression through the gait phase
        """
        self.Update_Angles(joint_angles_act, joint_angles_ref)
        self.Update_Activations(activations)
        self.Update_Gait_Phase(phase)

    def Plot_Info_Vertical(self, titles: list, y_plot_lists: list, x_plot_list: list,
    xlabel: str, ylabel: str, plot_name: str):
        """
        Clears current supplied figure, and plots using the given titles
        and lists.
        :param titles: List of strings which represent the titles of each
                        subplot.
        :param y_plot_lists: List of lists of doubles/floats. The y data to
                            be plotted on each subplot.
        :param x_plot_list: List of double/floats. The x data for the y
                            data to be plotted against.
        :param xlabel: String representation of the xlabel
        :param ylabel: String representation of the ylabel
        :param plot_name: String, super title and/or filename
        """
        # Clear exisiting plot
        fig, axs = plt.subplots(
            4, sharex=True, sharey=True, constrained_layout = True
            )
        for ax in axs:
            ax.clear()

        # Plot all inputted data on vertical subplot
        i = 0
        while i < len(titles):
            axs[i].plot(x_plot_list, y_plot_lists[i])
            axs[i].set_title(titles[i])
            i += 1
        # Label only outer sides of plots
        for ax in axs.flat:
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.label_outer()

        # Define folder path relative to python file location, and save
        fig_path = "{}/Plots/{}.png".format(os.path.dirname(__file__), plot_name)
        plt.savefig(fig_path)

    def Plot_Angle_Comparison(self, titles: list, y1_plot_lists: list, y2_plot_lists: list, x_plot_list: list,
                                xlabel: str, ylabel: str, plot_name: str):
        """
        Clears current supplied figure, and plots using the given titles
        and lists - compares items y1 and y2.
        :param titles: List of strings which represent the titles of each
                        subplot.
        :param y1_plot_lists: List of lists of doubles/floats. The y1 data to
                            be plotted on each subplot.
        :param y2_plot_lists: List of lists of doubles/floats. The y2 data to
                            be plotted on each subplot.
        :param x_plot_list: List of double/floats. The x data for the y
                            data to be plotted against.
        :param xlabel: String representation of the xlabel
        :param ylabel: String representation of the ylabel
        :param plot_name: String, super title and/or filename
        """
        # Clear exisiting plot
        fig, axs = plt.subplots(
            4, sharex=True, sharey=True, constrained_layout = True
            )
        for ax in axs:
            ax.clear()

        # Plot all inputted data on vertical subplot
        i = 0
        while i < len(titles):
            axs[i].plot(x_plot_list, y1_plot_lists[i], color="blue")
            axs[i].plot(x_plot_list, y2_plot_lists[i], color="red")
            axs[i].set_title(titles[i])
            i += 1
        # Label only outer sides of plots
        for ax in axs.flat:
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.label_outer()

        # Define folder path relative to python file location, and save
        fig_path = "{}/Plots/{}.png".format(os.path.dirname(__file__), plot_name)
        plt.savefig(fig_path)

    def Plot_Select(self, mode: str, fname: str):
        """
        Uses the Plot_Info_Vertical function to plot an aspect
        which is indicated by the string input:
        "Torque": Joint torque plot
        "Angle": Joint angle plot
        "Angle Comp": TODO
        "Knee activation: knee activation plot
        "hip flex ext activation"
        "hip add abd activation"
        "hip int ext rot activation"
        :param mode: String representation of the desired plot
        :param fname: The filename of the saved plot
        """
        if mode == "Torque":
            titles = ["Left Hip Torque", "Left Knee Torque",
                        "Right Hip Torque", "Right Knee Torque"]
            y_plot_lists = [self.t_l_hip, self.t_l_knee,
                            self.t_r_hip, self.t_r_knee]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Joint Torque"
            plot_name = fname
            # axs = self.torque_axs
            self.Plot_Info_Vertical(titles, y_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)

        elif mode == "Angle":
            titles = ["Left Hip Angle", "Left Knee Angle",
                        "Right Hip Angle", "Right Knee Angle"]
            y_plot_lists = [self.l_hip_angle, self.l_knee_angle,
                            self.r_hip_angle, self.r_knee_angle]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Joint Angle"
            plot_name = fname
            # axs = self.angle_axs
            self.Plot_Info_Vertical(titles, y_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)

        elif mode == "Angle Comp":
            titles = ["Left Hip Angle", "Left Knee Angle",
                        "Right Hip Angle", "Right Knee Angle"]
            y1_plot_lists = [self.l_hip_angle, self.l_knee_angle,
                            self.r_hip_angle, self.r_knee_angle]
            y2_plot_lists = [self.d_l_hip_angle, self.d_l_knee_angle,
                            self.d_r_hip_angle, self.d_r_knee_angle]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Joint Angle"
            plot_name = fname
            # axs = self.angle_axs
            self.Plot_Angle_Comparison(titles, y1_plot_lists, y2_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)

        elif mode == "Knee activation":
            titles = ["Left Knee Flexion", "Left Knee Extension",
                        "Right Knee Flexion", "Right Knee Extension"]
            y_plot_lists = [self.l_k_flex, self.l_k_ext,
                            self.r_k_flex, self.r_k_ext]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Activation"
            plot_name = fname
            # axs = self.knee_axs
            self.Plot_Info_Vertical(titles, y_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)

        elif mode == "hip flex ext activation":
            titles = ["Left Hip Flexion", "Left Hip Extension",
                        "Right Hip Flexion", "Right Hip Extension"]
            y_plot_lists = [self.l_h_flex, self.l_h_ext,
                            self.r_h_flex, self.r_h_ext]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Activation"
            plot_name = fname
            # axs = self.hip_flex_ext_axs
            self.Plot_Info_Vertical(titles, y_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)

        elif mode == "hip add abd activation":
            titles = ["Left Hip Adduction", "Left Hip Abduction",
                        "Right Hip Adduction", "Right Hip Abduction"]
            y_plot_lists = [self.l_h_add, self.l_h_abd,
                            self.r_h_add, self.r_h_abd]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Activation"
            plot_name = fname
            # axs = self.hip_add_abd_axs
            self.Plot_Info_Vertical(titles, y_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)

        elif mode == "hip int ext rot activation":
            titles = ["Left Hip Int Rot", "Left Hip Ext Rot",
                        "Right Hip Int Rot", "Right Hip Ext Rot"]
            y_plot_lists = [self.l_h_introt, self.l_h_extrot,
                            self.r_h_introt, self.r_h_extrot]
            x_plot_list = self.gait_phase
            xlabel = "Gait Phase"
            ylabel = "Activation"
            plot_name = fname
            # axs = self.hip_int_ext_rot_axs
            self.Plot_Info_Vertical(titles, y_plot_lists, x_plot_list,
                                    xlabel, ylabel, plot_name)
        else:
            print("Invalid Entry")

    def Plot_All_Exo(self):
        """
        Plots all collected data for the exo-applied environment
        """
        self.Plot_Select("Torque", "Exo_Torque")
        # print("============ Torque plotted ===========")
        self.Plot_Select("Angle", "Exo_Angle")
        # print("============ Angle plotted ===========")
        self.Plot_Select("Angle Comp", "Exo_Angle_Comp")
        self.Plot_Select("Knee activation", "Exo_Knee_Act")
        # print("============ Knee plotted ===========")
        self.Plot_Select("hip flex ext activation", "Exo_Hip_FE_act")
        # print("============ hip1 plotted ===========")
        self.Plot_Select("hip add abd activation", "Exo_Hip_AA_Act")
        # print("============ hip2 plotted ===========")
        self.Plot_Select("hip int ext rot activation", "Exo_Hip_IER_Act")
        # print("============ hip3 plotted ===========")
    
    def Plot_All_SIM(self):
        """
        Plots all collected data for the original environment
        (no exo -> no exo torques plotted)
        """
        self.Plot_Select("Angle", "SIM_Angle")
        # print("============ Angle plotted ===========")
        self.Plot_Select("Angle Comp", "SIM_Angle_Comp")
        self.Plot_Select("Knee activation", "SIM_Knee_Act")
        # print("============ Knee plotted ===========")
        self.Plot_Select("hip flex ext activation", "SIM_Hip_FE_act")
        # print("============ hip1 plotted ===========")
        self.Plot_Select("hip add abd activation", "SIM_Hip_AA_Act")
        # print("============ hip2 plotted ===========")
        self.Plot_Select("hip int ext rot activation", "SIM_Hip_IER_Act")
        # print("============ hip3 plotted ===========")

    def Update_Plot_All(self, exo_torques, #TODO update actual angles
                        joint_angles, activations, phase):
        """
        Updates all the lists to plot, then plots them, by calling 
        Update_All and Plot_All functions
        :param exo_torques: list of applied exo torques in the order:
                [L hip T, L knee T, R hip T, R knee T]
        :param joint_angles: list of joint angles in the order:
                [L hip, L knee, R hip, R knee]
        :param activations: list of muscle group activations in the order:
            {LHFlex,LHExt,LHAbd,LHAdd,LHExtRot,LHIntRot,LKFlex,LKExt,
            RHFlex,RHExt,RHAbd,RHAdd,RHExtRot,RHIntRot,RKFlex,RKExt,} 
        """
        #TODO update actual angles
        self.Update_All(exo_torques, joint_angles, activations, phase)
        self.Plot_All()

    def Save_All_To_CSV_Exo(self):
        """
        Saves all the data lists to CSV for exo env.
        """
        data_lists = np.array([
            self.t_l_hip, self.t_l_knee, self.t_r_hip, 
            self.t_r_knee, self.l_hip_angle, self.l_knee_angle,
            self.r_hip_angle, self.r_knee_angle, self.l_h_flex,
            self.l_h_ext, self.l_h_abd, self.l_h_add, self.l_h_extrot,
            self.l_h_introt, self.l_k_flex, self.l_k_ext, self.r_h_flex,
            self.r_h_ext, self.r_h_abd, self.r_h_add, self.r_h_extrot,
            self.r_h_introt, self.r_k_flex, self.r_k_ext, self.gait_phase
        ])
        path = "{}/Data/All_Data_EXO.csv".format(os.path.dirname(__file__))
        np.savetxt(path, data_lists.T, delimiter=",")

    def Save_All_To_CSV_SIM(self):
        """
        Saves all the data lists to CSV for MASS env.
        """
        data_lists = np.array([
            self.l_hip_angle, self.l_knee_angle,
            self.r_hip_angle, self.r_knee_angle, self.l_h_flex,
            self.l_h_ext, self.l_h_abd, self.l_h_add, self.l_h_extrot,
            self.l_h_introt, self.l_k_flex, self.l_k_ext, self.r_h_flex,
            self.r_h_ext, self.r_h_abd, self.r_h_add, self.r_h_extrot,
            self.r_h_introt, self.r_k_flex, self.r_k_ext, self.gait_phase
        ])
        path = "{}/Data/All_Data_SIM.csv".format(os.path.dirname(__file__))
        np.savetxt(path, data_lists.T, delimiter=",")
