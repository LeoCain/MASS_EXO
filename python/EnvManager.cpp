#include "EnvManager.h"
#include "DARTHelper.h"
#include <omp.h>

/**
 * This file contains all the C++ functions that have been ported to 
 * python via pybind. most of them are just call functions from
 * environment.cpp. when you see calls to pymss in python,
 * this file is being called. i.e. it enables you to setup and
 * interact with the MASS environment from python
 */

EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs)
{
	// mMetafile = meta_file;
	dart::math::seedRand();
	omp_set_num_threads(mNumEnvs);
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new MASS::Environment());
		MASS::Environment* env = mEnvs.back();

		env->Initialize(meta_file,false);
	}
	muscle_torque_cols = mEnvs[0]->GetMuscleTorques().rows();
	tau_des_cols = mEnvs[0]->GetDesiredTorques().rows();
	mEoe.resize(mNumEnvs);
	mRewards.resize(mNumEnvs);
	mStates.resize(mNumEnvs, GetNumState());
	mMuscleTorques.resize(mNumEnvs, muscle_torque_cols);
	mDesiredTorques.resize(mNumEnvs, tau_des_cols);
	
	// win = new MASS::Window(mEnvs[0]);
}
int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}
int
EnvManager::
GetNumAction()
{
	return mEnvs[0]->GetNumAction();
}
int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}
int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}
int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}
bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}
void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step();
}
void
EnvManager::
Reset(bool RSI,int id)
{
	mEnvs[id]->Reset(RSI);
}
bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}

double 
EnvManager::
GetReward(int id)
{
	
	return mEnvs[id]->GetReward();
}

void
EnvManager::
Steps(int num)
{
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->Step();
	}
}
void
EnvManager::
StepsAtOnce()
{
	int num = this->GetNumSteps();
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->Step();
	}
}
void
EnvManager::
Resets(bool RSI)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}
const Eigen::VectorXd&
EnvManager::
IsEndOfEpisodes()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEoe[id] = (double)mEnvs[id]->IsEndOfEpisode();
	}

	return mEoe;
}
const Eigen::MatrixXd&
EnvManager::
GetStates()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mStates.row(id) = mEnvs[id]->GetState().transpose();
	}

	return mStates;
}
void
EnvManager::
SetActions(const Eigen::MatrixXd& actions)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction(actions.row(id).transpose());
	}
}
const Eigen::VectorXd&
EnvManager::
GetRewards()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mRewards[id] = mEnvs[id]->GetReward();
	}
	return mRewards;
}
const Eigen::VectorXd&
EnvManager::
GetGaitRewards()
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mRewards[id] = mEnvs[id]->GetGaitReward();
	}
	return mRewards;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTorques()
{
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mMuscleTorques.row(id) = mEnvs[id]->GetMuscleTorques();
	}
	return mMuscleTorques;
}
const Eigen::MatrixXd&
EnvManager::
GetDesiredTorques()
{
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mDesiredTorques.row(id) = mEnvs[id]->GetDesiredTorques();
	}
	return mDesiredTorques;
}

void
EnvManager::
SetActivationLevels(const Eigen::MatrixXd& activations)
{
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->SetActivationLevels(activations.row(id));	// why are there multiple envs
}

void
EnvManager::
ComputeMuscleTuples()
{
	int n = 0;
	int rows_JtA;
	int rows_tau_des;
	int rows_L;
	int rows_b;

	for(int id=0;id<mNumEnvs;id++)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		n += tps.size();
		if(tps.size()!=0)
		{
			rows_JtA = tps[0].JtA.rows();
			rows_tau_des = tps[0].tau_des.rows();
			rows_L = tps[0].L.rows();
			rows_b = tps[0].b.rows();
		}
	}
	
	mMuscleTuplesJtA.resize(n, rows_JtA);
	mMuscleTuplesTauDes.resize(n, rows_tau_des);
	mMuscleTuplesL.resize(n, rows_L);
	mMuscleTuplesb.resize(n, rows_b);

	int o = 0;
	for(int id=0;id<mNumEnvs;id++)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			mMuscleTuplesJtA.row(o) = tps[j].JtA;
			mMuscleTuplesTauDes.row(o) = tps[j].tau_des;
			mMuscleTuplesL.row(o) = tps[j].L;
			mMuscleTuplesb.row(o) = tps[j].b;
			o++;
		}
		tps.clear();
	}
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesJtA()
{
	return mMuscleTuplesJtA;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesTauDes()
{
	return mMuscleTuplesTauDes;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesL()
{
	return mMuscleTuplesL;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesb()
{
	return mMuscleTuplesb;
}

// Added by XS
/**
 * @brief calls MASS::Environment function which sets a new value for the left hip torque vector
 * @param vec Eigen::Vector3d& representation of the new T vector
 */
void 
EnvManager::
SetLHipTs(float T){
#pragma omp parallel for				// Multithreading?
	for (int id = 0;id<mNumEnvs;++id){
		mEnvs[id]->SetLHipT(T);
	}
}

/**
 * @brief calls MASS::Environment function which sets a new value for the right hip torque vector
 * @param vec Eigen::Vector3d& representation of the new T vector
 */
void 
EnvManager::
SetRHipTs(float T){
#pragma omp parallel for				// Multithreading?
	for (int id = 0;id<mNumEnvs;++id){
		mEnvs[id]->SetRHipT(T);
	}
}

/**
 * @brief calls MASS::Environment function which sets a new value for the left knee torque vector
 * @param vec Eigen::VectorXd& representation of the new T vector
 */
void 
EnvManager::
SetLKneeTs(float T){
#pragma omp parallel for				// Multithreading?
	for (int id = 0;id<mNumEnvs;++id){
		mEnvs[id]->SetLKneeT(T);
	}
}

/**
 * @brief calls MASS::Environment function which sets a new value for the right knee torque vector
 * @param vec Eigen::VectorXd& representation of the new T vector
 */
void 
EnvManager::
SetRKneeTs(float T){
#pragma omp parallel for				// Multithreading?
	for (int id = 0;id<mNumEnvs;++id){
		mEnvs[id]->SetRKneeT(T);
	}

}

// void 
// EnvManager::
// MakeWindow(std::string simNN_path, std::string muscleNN_path){
// 	MASS::Environment* env;
// 	env = new MASS::Environment();
// 	env->Initialize(mMetafile, false);
// 	mWindow = new MASS::Window(env, simNN_path, muscleNN_path);

// 	// for(int i = 0;i<mNumEnvs;i++){
// 	// 	mWindows.push_back(new MASS::Window(mEnvs[i], simNN_path, muscleNN_path));
// 	// 	MASS::Window* win = mWindows.back();
// 	// }
// }

// void
// EnvManager::
// DrawWindow(){
// 	mWindow->draw();
// }

PYBIND11_MODULE(pymss, m)
{
	py::class_<EnvManager>(m, "pymss")
		.def(py::init<std::string,int>())
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumAction",&EnvManager::GetNumAction)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("UseMuscle",&EnvManager::UseMuscle)
		.def("Step",&EnvManager::Step)
		.def("Reset",&EnvManager::Reset)
		.def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
		.def("GetReward",&EnvManager::GetReward)
		.def("Steps",&EnvManager::Steps)
		.def("StepsAtOnce",&EnvManager::StepsAtOnce)
		.def("Resets",&EnvManager::Resets)
		.def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
		.def("GetStates",&EnvManager::GetStates)
		.def("SetActions",&EnvManager::SetActions)
		.def("GetRewards",&EnvManager::GetRewards)
		.def("GetGaitRewards",&EnvManager::GetGaitRewards)
		//.def("GetLegJointAngles",&EnvManager::GetLegJointAngles)
		.def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
		.def("GetNumMuscles",&EnvManager::GetNumMuscles)
		.def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
		.def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
		.def("SetActivationLevels",&EnvManager::SetActivationLevels)
		.def("ComputeMuscleTuples",&EnvManager::ComputeMuscleTuples)
		.def("GetMuscleTuplesJtA",&EnvManager::GetMuscleTuplesJtA)
		.def("GetMuscleTuplesTauDes",&EnvManager::GetMuscleTuplesTauDes)
		.def("GetMuscleTuplesL",&EnvManager::GetMuscleTuplesL)
		.def("GetMuscleTuplesb",&EnvManager::GetMuscleTuplesb)
		.def("SetLHipTs", &EnvManager::SetLHipTs)
		.def("SetRHipTs", &EnvManager::SetRHipTs)
		.def("SetLKneeTs", &EnvManager::SetLKneeTs)
		.def("SetRKneeTs", &EnvManager::SetRKneeTs);
		// .def("MakeWindow", &EnvManager::MakeWindow)
		// .def("DrawWindow", &EnvManager::DrawWindow);
}