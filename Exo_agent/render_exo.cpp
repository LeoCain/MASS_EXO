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
    /** instantiate the agent object using pybind **/
	mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = py::module::import("sys");

	py::str module_dir = (std::string(MASS_ROOT_DIR)+"/Exo_agent").c_str();
    sys_module.attr("path").attr("insert")(1, module_dir);
	py::exec("from RLlib_MASS import Exo_Trainer", mns);
	py::exec("from MASS_env import MASS_env", mns);
    exo_agent = py::eval("Exo_Trainer()", mns);
	std::cout << "\n============agent initialised================\n\n";
    py::object restore_agent = exo_agent.attr("Restore_Agent");
    restore_agent(RLlib_agent_path);
}

/**
 * @brief Same as the MASS::Window step function, but incorperates
 *          Exo torques by calling exo net and applying the output
 */
void
exo_Window::
Step()
{   
	Eigen::VectorXd exo_torques = GetExoTorquesFromNN();
	// Eigen::VectorXd exo_torques;
	// exo_torques << 10, 10, 10, 10;
	mEnv->SetExoTorques(exo_torques);
	std::cout << "torques:" << mEnv->GetLHipT() << ", " << mEnv->GetLKneeT() << ", " << mEnv->GetRHipT() << ", " << mEnv->GetRKneeT() << "\n";

    int num = mEnv->GetSimulationHz()/mEnv->GetControlHz();
	Eigen::VectorXd action;
	if(mNNLoaded)
		action = GetActionFromNN();		// Some vector from which muscle torques can be calculated?
	else
		action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
	mEnv->SetAction(action);			// Action sent to environment

	if(mEnv->GetUseMuscle())
	{
		int inference_per_sim = 2;
		for(int i=0;i<num;i+=inference_per_sim){
			Eigen::VectorXd mt = mEnv->GetMuscleTorques();		// Muscle torques can be ??
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
	// std::cout << "step reward:" << mEnv->GetReward() << "state:" << mEnv->GetState() << '\n';
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


