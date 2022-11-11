#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
namespace MASS
{

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::VectorXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};
class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);

	dart::dynamics::SkeletonPtr exo_model;
public:
	void Step();
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& a);
	double GetReward();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};
	int GetNumState(){return mNumState;}
	int GetNumAction(){return mNumActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	bool GetUseMuscle(){return mUseMuscle;}

	// Added by XS:
	Eigen::VectorXd GetExoTorques();
	double GetGaitReward();

	// Setters and getters for the hip/knee joint torque vectors:
	/**
	 * @return Eigen::Vector3d representation of Left Hip Torque
	 */
	float GetLHipT(){return T_Hip_L;}	

	/**
	 * @return Eigen::Vector3d representation of Right Hip Torque
	 */		
	float GetRHipT(){return T_Hip_R;}

	/**
	 * @return Eigen::VectorXd representation of Left Knee Torque
	 */
	float GetLKneeT(){return T_Knee_L;}	

	/**
	 * @return Eigen::VectorXd representation of Right Knee Torque
	 */
	float GetRKneeT(){return T_Knee_R;}

	/**
	 * @brief Sets a new value for the left hip torque vector
	 * @param vec Eigen::Vector3d& representation of the new T vector
	 */
	void SetLHipT(float T){T_Hip_L = T;}

	/**
	 * @brief Sets a new value for the right hip torque vector
	 * @param vec Eigen::Vector3d& representation of the new T vector
	 */
	void SetRHipT(float T){T_Hip_R = T;}

	/**
	 * @brief Sets a new value for the left knee torque vector
	 * @param vec Eigen::VectorXd& representation of the new T vector
	 */
	void SetLKneeT(float T){T_Knee_L = T;}

	/**
	 * @brief Sets a new value for the right knee torque vector
	 * @param vec Eigen::VectorXd& representation of the new T vector
	 */
	void SetRKneeT(float T){T_Knee_R = T;}

	void SetExoTorques(Eigen::VectorXd Ts);

	Eigen::VectorXd& GetTargetPositions(){return mTargetPositions;}

private:
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mTargetPositions,mTargetVelocities;

	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;	// The number of DOFs of the joint with no parent in the skeleton - the "first" joint? - XS

	Eigen::VectorXd mActivationLevels;
	Eigen::VectorXd mAverageActivationLevels;
	Eigen::VectorXd mDesiredTorque;
	std::vector<MuscleTuple> mMuscleTuples;
	MuscleTuple mCurrentMuscleTuple;
	int mSimCount;
	int mRandomSampleIndex;

	double w_q,w_v,w_ee,w_com;

	// Added by XS:
	// Variables added to represent exo torques to be applied to knee/hip joints
	float T_Hip_L = 0;						// Left hip
	float T_Hip_R = 0;						// Right hip
	// Knee torque is 1D vector as it is revolute 
	float T_Knee_L = 0;						// Left knee
	float T_Knee_R = 0; 					// Right knee
	// Eigen::VectorXd::Zero(1);	

};
};

#endif