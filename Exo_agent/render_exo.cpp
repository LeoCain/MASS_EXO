#include "render_exo.h"
#include "Environment.h"
#include "Character.h"
using namespace MASS;

/**
 * @brief executes all parent class constructors -> sets up muscle and sim
 * NNs for normal benjaSIM operation
 * 
 * @param env The normal benjaSIM environment, exo modifications to env
 *            are applied in the overridden step function.
 * @param nn_path The path to the trajectory target net weights (std benjaSIM)
 * @param muscle_nn_path The path to the muscle activation net weights (std benjaSIM)
 * @param RLlib_agent_path The path to the RLlib exo policy checkpoint
 */
exo_Window::
exo_Window(Environment* env, const std::string& nn_path, const std::string& muscle_nn_path, const std::string& RLlib_agent_path) 
    : Window(env, nn_path, muscle_nn_path)
{
    /** setup pybind and add relevant directory/s to its path **/
	mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = py::module::import("sys");
	py::str module_dir = "/home/medicalrobotics/Anton/MASS_EXO/Exo_agent";
    sys_module.attr("path").attr("insert")(1, module_dir);

	/** Execute relevant imports **/
	py::exec("from RLlib_MASS import Exo_Trainer", mns);
	py::exec("from MASS_env import MASS_env", mns);
	py::exec("from plotter import Plotter", mns);

	/** instantiate the agent object using pybind **/
    exo_agent = py::eval("Exo_Trainer()", mns);
	std::cout << "\n============agent initialised================\n\n";

	/** restore agent to the command-line supplied checkpoint * **/
    py::object restore_agent = exo_agent.attr("Restore_Agent");
    restore_agent(RLlib_agent_path);

	/** instantiate the python plotter object **/
	plotter = py::eval("Plotter()", mns);

	/** Define muscle groups **/
	define_muscle_groups();
}

/**
 * @brief Same as the MASS::Window step function, but incorperates
 * Exo torques by calling exo net and applying the output
 */
void
exo_Window::
Step()
{   
	Eigen::VectorXd exo_torques = GetExoTorquesFromNN();
	mEnv->SetExoTorques(exo_torques);
	// std::cout << "torques:" << mEnv->GetLHipT() << ", " << mEnv->GetLKneeT() << ", " << mEnv->GetRHipT() << ", " << mEnv->GetRKneeT() << "\n";

    int num = mEnv->GetSimulationHz()/mEnv->GetControlHz();
	Eigen::VectorXd action;
	if(mNNLoaded)
		action = GetActionFromNN();		// Some vector from which muscle torques can be calculated?
	else
		action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
	mEnv->SetAction(action);			// Action sent to environment

	record_data();	// Plot angles, torques, activations

	if(mEnv->GetUseMuscle())
	{
		int inference_per_sim = 2;
		for(int i=0;i<num;i+=inference_per_sim){
			Eigen::VectorXd mt = mEnv->GetMuscleTorques();		
			mEnv->SetActivationLevels(GetActivationFromNN(mt));
			for(int j=0;j<inference_per_sim;j++)
				mEnv->Step();
		}	
	}
	else
	{
		for(int i=0;i<num;i++)
			mEnv->Step();	
	}
	std::cout << "step reward:" << mEnv->GetReward() << "state:" << mEnv->GetState() << '\n';
}

/**
 * @brief Get the Exo Torques from the RLlib library
 * 
 * @return Eigen::VectorXd representation of the torques: [T_LHip, T_LKnee, T_RHip, T_RKnee]
 */
Eigen::VectorXd
exo_Window::
GetExoTorquesFromNN()
{
    Eigen::VectorXd exo_Ts;
    exo_Ts = exo_agent.attr("get_action")(mEnv->GetState()).cast<Eigen::VectorXd>();
	return exo_Ts;
}

/**
 * @brief records applied exo torques, benjaSIM joint angles, benjaSIM muscle group activation,
 * and sends to python script to be graphed
 */
void
exo_Window::
record_data()
{
	/** Read and save benjaSIM activations, group into:
	 * Hip flexion/extension groups
	 * Hip abduction/adduction groups
	 * Hip external/internal rotation groups
	 * Knee flexion/extension groups
	 */
	// Initialise activation total to 0 for all groups
	for (MuscleGroup& muscle_group : muscle_groups)
	{
		muscle_group.total_activation = 0.0;
	}

	// Loop through all muscles in benjaSIM
	for(auto muscle : mEnv->GetCharacter()->GetMuscles())
	{
		// Loop through all the defined muscles groups of interest
		for (MuscleGroup& muscle_group : muscle_groups)
		{
			// if we are interested in the current muscle
			// i.e. it is in one of the defined groups,
			// add the activation to the group's total.
			if (muscle_group.group.count(muscle->name))
			{
				muscle_group.add(muscle->activation);
			}
		}
	}
	
	// initialise vector to store avg activations of each group:
	// {LHFlex,LHExt,LHAbd,LHAdd,LHExtRot,LHIntRot,LKFlex,LKExt,
    //  RHFlex,RHExt,RHAbd,RHAdd,RHExtRot,RHIntRot,RKFlex,RKExt,} 
	Eigen::VectorXd avg_activations(16);
	// Find the average activation for each muscle group,
	// append to vector for sending to python
	for (int i=0; i<16; i++)
	{
		avg_activations[i] =  muscle_groups[i].get_avg_activation();
	}

	/** Read and save benjaSIM hip/knee joint angles **/
	// dummy angle vector for initialisation
	Eigen::VectorXd joint_angles_act(4);
	Eigen::VectorXd joint_angles_ref(4);
	// Actual positions
	double l_hip_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getParentJoint()->getPositions()[0];
	double r_hip_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getParentJoint()->getPositions()[0];
	double l_knee_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TibiaL")->getParentJoint()->getPositions()[0];
	double r_knee_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TibiaR")->getParentJoint()->getPositions()[0];
	// Reference positions
	auto l_hip_joint_idx = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	auto r_hip_joint_idx = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	auto l_knee_joint_idx = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	auto r_knee_joint_idx = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);	
	double l_hip_ref =  mEnv->GetTargetPositions()[l_hip_joint_idx];
	double r_hip_ref = mEnv->GetTargetPositions()[r_hip_joint_idx];
	double l_knee_ref = mEnv->GetTargetPositions()[l_knee_joint_idx];
	double r_knee_ref = mEnv->GetTargetPositions()[r_knee_joint_idx];

	joint_angles_act << l_hip_act, l_knee_act, r_hip_act, r_knee_act;
	joint_angles_ref << l_hip_ref, l_knee_ref, r_hip_ref, r_knee_ref;

	/** Send (torques, angles, activations, gait ref) to python for plotting **/
	py::object Update_All_Exo = plotter.attr("Update_All_Exo");
    Update_All_Exo(mEnv->GetExoTorques(), joint_angles_act, joint_angles_ref,
					avg_activations, mEnv->GetState().tail(1));
}

/**
 * @brief Save all the data to a CSV (All_Data.csv) with features as columns
 * 		  Also plots all the data.
 */
void
exo_Window::
Plot_And_Save()
{	
	/** Save data to CSV **/
	py::object Save_All_To_CSV_Exo = plotter.attr("Save_All_To_CSV_Exo");
    Save_All_To_CSV_Exo();
	/** Plot the data **/
	py::object Plot_All_Exo = plotter.attr("Plot_All_Exo");
    Plot_All_Exo();
}

void
exo_Window::
define_muscle_groups()
{
	// L Hip Flexion
	muscle_groups[LHFlex].group = {
		"L_Rectus_Femoris", "L_Rectus_Femoris1", "L_iliacus", "L_iliacus1",
		"L_iliacus2", "L_Psoas_Major", "L_Psoas_Major1", "L_Psoas_Major2", 
		"L_Psoas_Minor", "L_Sartorius"
		};
	// L Hip Extension
	muscle_groups[LHExt].group = {
		"L_Gluteus_Maximus", "L_Gluteus_Maximus1", "L_Gluteus_Maximus2", 
		"L_Gluteus_Maximus3", "L_Gluteus_Maximus4", "L_Adductor_Magnus", 
		"L_Adductor_Magnus1", "L_Adductor_Magnus2", "L_Adductor_Magnus3",
		"L_Adductor_Magnus4", "L_Bicep_Femoris_Longus", "L_Semitendinosus",
		"L_Semimembranosus", "L_Semimembranosus1"
	};
	// L Hip Abduction
	muscle_groups[LHAbd].group = {
		"L_Gluteus_Medius", "L_Gluteus_Medius1", "L_Gluteus_Medius2", 
		"L_Gluteus_Medius3", "L_Gluteus_Minimus", "L_Gluteus_Minimus1", 
		"L_Gluteus_Minimus2", "L_Tensor_Fascia_Lata", "L_Tensor_Fascia_Lata1", 
		"L_Tensor_Fascia_Lata2"
	};
	// L Hip Adduction
	muscle_groups[LHAdd].group = {
		"L_Pectineus", "L_Adductor_Longus", "L_Adductor_Longus1",
		"L_Gracilis", "L_Adductor_Brevis", "L_Adductor_Brevis1",
		"L_Adductor_Magnus", "L_Adductor_Magnus1", "L_Adductor_Magnus2",
		"L_Adductor_Magnus3", "L_Adductor_Magnus4"
	};
	// L Hip External rotation
	muscle_groups[LHExtRot].group = {
		"L_Gluteus_Maximus", "L_Gluteus_Maximus1", "L_Gluteus_Maximus2", 
		"L_Gluteus_Maximus3", "L_Gluteus_Maximus4", "L_Piriformis", 
		"L_Piriformis1", "L_Quadratus_Femoris", "L_Obturator_Externus",
		 "L_Obturator_Internus", "L_Superior_Gemellus", "L_Inferior_Gemellus"
	};
	// L hip Internal rotation
	muscle_groups[LHIntRot].group = { //SUS - exact same muscles as abductor grp
		"L_Gluteus_Medius", "L_Gluteus_Medius1", "L_Gluteus_Medius2", 
		"L_Gluteus_Medius3", "L_Gluteus_Minimus", "L_Gluteus_Minimus1", 
		"L_Gluteus_Minimus2", "L_Tensor_Fascia_Lata", "L_Tensor_Fascia_Lata1", 
		"L_Tensor_Fascia_Lata2"
	};
	// L Knee Flexion
	muscle_groups[LKFlex].group = {
		"L_Semimembranosus", "L_Semimembranosus1", "L_Semitendinosus",
		"L_Bicep_Femoris_Longus", "L_Bicep_Femoris_Short", "L_Bicep_Femoris_Short1",
		"L_Gracilis", "L_Sartorius", "L_Gastrocnemius_Lateral_Head", 
		"L_Gastrocnemius_Medial_Head", "L_Plantaris", "L_Popliteus"
	};
	// L Knee Extension
	muscle_groups[LKExt].group = {
		"L_Rectus_Femoris", "L_Rectus_Femoris1", "L_Vastus_Lateralis", 
		"L_Vastus_Lateralis1", "L_Vastus_Medialis", "L_Vastus_Medialis1",
		"L_Vastus_Medialis2", "L_Vastus_Intermedius", "L_Vastus_Intermedius1"
	};

	// R Hip Flexion
	muscle_groups[RHFlex].group = {
		"R_Rectus_Femoris", "R_Rectus_Femoris1", "R_iliacus", "R_iliacus1",
		"R_iliacus2", "R_Psoas_Major", "R_Psoas_Major1", "R_Psoas_Major2", 
		"R_Psoas_Minor", "R_Sartorius"
	};
	// R Hip Extension
	muscle_groups[RHExt].group = {
		"R_Gluteus_Maximus", "R_Gluteus_Maximus1", "R_Gluteus_Maximus2", 
		"R_Gluteus_Maximus3", "R_Gluteus_Maximus4", "R_Adductor_Magnus", 
		"R_Adductor_Magnus1", "R_Adductor_Magnus2", "R_Adductor_Magnus3",
		"R_Adductor_Magnus4", "R_Bicep_Femoris_Longus", "R_Semitendinosus",
		"R_Semimembranosus", "R_Semimembranosus1"
	};
	// R Hip Abduction
	muscle_groups[RHAbd].group = {
		"R_Gluteus_Medius", "R_Gluteus_Medius1", "R_Gluteus_Medius2", 
		"R_Gluteus_Medius3", "R_Gluteus_Minimus", "R_Gluteus_Minimus1", 
		"R_Gluteus_Minimus2", "R_Tensor_Fascia_Lata", "R_Tensor_Fascia_Lata1", 
		"R_Tensor_Fascia_Lata2"
	};
	// R Hip Adduction
	muscle_groups[RHAdd].group = {
		"R_Pectineus", "R_Adductor_Longus", "R_Adductor_Longus1",
		"R_Gracilis", "R_Adductor_Brevis", "R_Adductor_Brevis1",
		"R_Adductor_Magnus", "R_Adductor_Magnus1", "R_Adductor_Magnus2",
		"R_Adductor_Magnus3", "R_Adductor_Magnus4"
	};
	// R Hip External rotation
	muscle_groups[RHExtRot].group = {
		"R_Gluteus_Maximus", "R_Gluteus_Maximus1", "R_Gluteus_Maximus2", 
		"R_Gluteus_Maximus3", "R_Gluteus_Maximus4", "R_Piriformis", 
		"R_Piriformis1", "R_Quadratus_Femoris", "R_Obturator_Externus",
		 "R_Obturator_Internus", "R_Superior_Gemellus", "R_Inferior_Gemellus"
	};
	// R hip Internal rotation
	muscle_groups[RHIntRot].group = {
		"R_Gluteus_Medius", "R_Gluteus_Medius1", "R_Gluteus_Medius2", 
		"R_Gluteus_Medius3", "R_Gluteus_Minimus", "R_Gluteus_Minimus1", 
		"R_Gluteus_Minimus2", "R_Tensor_Fascia_Lata", "R_Tensor_Fascia_Lata1", 
		"R_Tensor_Fascia_Lata2"
	};
	// R Knee Flexion
	muscle_groups[RKFlex].group = {
		"R_Semimembranosus", "R_Semimembranosus1", "R_Semitendinosus",
		"R_Bicep_Femoris_Longus", "R_Bicep_Femoris_Short", "R_Bicep_Femoris_Short1",
		"R_Gracilis", "R_Sartorius", "R_Gastrocnemius_Lateral_Head", 
		"R_Gastrocnemius_MediaR_Head", "R_Plantaris", "R_Popliteus"
	};
	// R Knee Extension
	muscle_groups[RKExt].group = {
		"R_Rectus_Femoris", "R_Rectus_Femoris1", "R_Vastus_Lateralis", 
		"R_Vastus_Lateralis1", "R_Vastus_Medialis", "R_Vastus_Medialis1",
		"R_Vastus_Medialis2", "R_Vastus_Intermedius", "R_Vastus_Intermedius1"
	};
}

