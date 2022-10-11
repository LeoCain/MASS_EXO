#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "dart/collision/bullet/bullet.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

Environment::
Environment()
	:mControlHz(30),mSimulationHz(900),mWorld(std::make_shared<World>()),mUseMuscle(true),w_q(0.65),w_v(0.1),w_ee(0.15),w_com(0.1)	// numbers in brackets??
{

}

/**
 * @brief: Loads the parameter file, and configures the environment accordingly
 * 
 * @param meta_file: Parameter file containing configuration information
 * @param load_obj: ????
 */
void
Environment::
Initialize(const std::string& meta_file,bool load_obj)
{
	std::ifstream ifs(meta_file);	// Input stream class is created so meta_file can be read/operated on
	
	if(!(ifs.is_open()))			// Self-exlanatory - triggered if ifstream object cannot open file
	{
		std::cout<<"Can't read file "<<meta_file<<std::endl;
		return;
	}
	std::string str;		// String initialised to store each line of file.
	std::string index;		// String initialised to store the first word of each line.
	std::stringstream ss;	// stringstream object used as a buffer such that each word of the current line can be examined separately.
	MASS::Character* character = new MASS::Character();	// create a new MASS character object pointer.
	while(!ifs.eof())	// While not at the end of file:
	{
		// Clear all veriables from previous loop:
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs,str);	// Assign entire current line to str String.
		ss.str(str);			// Save current line to stringstream ss, so each word can be indexed.
		ss>>index;				// Assign the first word on this line to the index String.
		// Use index.compare() to check which parameter is being set on this line (if index.compare("str") returns 0, current word matches "str").
		// A switch case might be better here
		if(!index.compare("use_muscle"))
		{	
			// Use the stringstream ss to check the next word, and initialise this parameter appropriately
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseMuscle(true);	// sets mUseMuscle to true
			else
				this->SetUseMuscle(false);	// sets mUseMuscle to false
		}
		else if(!index.compare("con_hz")){
			int hz;
			ss>>hz;
			this->SetControlHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("skel_file")){
			std::string str2;
			ss>>str2;

			character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);
		}
		else if(!index.compare("muscle_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseMuscle())
				character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("exo_file")){
			std::string str2;
			printf("In\n");
			ss>>str2;
			exo_model = character->LoadExo(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("bvh_file")){	// This is the reference motion file.
			std::string str2,str3;

			ss>>str2>>str3;
			bool cyclic = false;
			if(!str3.compare("true"))
				cyclic = true;
			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}
		else if(!index.compare("reward_param")){
			double a,b,c,d;
			ss>>a>>b>>c>>d;
			this->SetRewardParameters(a,b,c,d);

		}


	}
	ifs.close();
	
	
	double kp = 300.0;
	character->SetPDParameters(kp,sqrt(2*kp));
	this->SetCharacter(character);
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	this->Initialize();
}

void
Environment::
Initialize()
{
	if(mCharacter->GetSkeleton()==nullptr){
		std::cout<<"Initialize character First"<<std::endl;
		exit(0);
	}
	if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")
		mRootJointDof = 6;
	else if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint")
		mRootJointDof = 3;	
	else
		mRootJointDof = 0;
	mNumActiveDof = mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof;
	if(mUseMuscle)
	{
		int num_total_related_dofs = 0;
		for(auto m : mCharacter->GetMuscles()){
			m->Update();
			num_total_related_dofs += m->GetNumRelatedDofs();
		}
		mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);
		mCurrentMuscleTuple.L = Eigen::VectorXd::Zero(mNumActiveDof*mCharacter->GetMuscles().size());
		mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
		mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
		mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size());
	}
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->addSkeleton(mCharacter->GetSkeleton());
	mWorld->addSkeleton(mGround);
	mAction = Eigen::VectorXd::Zero(mNumActiveDof);
	
	Reset(false);
	mNumState = GetState().rows();
}

void
Environment::
Reset(bool RSI)
{	
	mWorld->reset();	// reset DART simulation
	
	// Reset all exo torques to 0:
	SetLHipT(0);
	SetRHipT(0);
	SetLKneeT(0);
	SetRKneeT(0);

	// reset all forces and constraints
	mCharacter->GetSkeleton()->clearConstraintImpulses();
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();
	
	double t = 0.0;	// Set time to 0

	if(RSI)	// sets time randomly between 0 and 0.9*max time, not sure of the purpose
		t = dart::math::random(0.0,mCharacter->GetBVH()->GetMaxTime()*0.9);
	mWorld->setTime(t); 
	mCharacter->Reset();

	mAction.setZero();

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;

	mCharacter->GetSkeleton()->setPositions(mTargetPositions);
	mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
	mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
}

void
Environment::
Step()
{	
	if(mUseMuscle)	// it seems that the program will always enter this condititon, if use_muscle is set true in metadata.txt
	{
		int count = 0;
		for(auto muscle : mCharacter->GetMuscles())	// Iterate through each muscle
		{
			muscle->activation = mActivationLevels[count++];
			muscle->Update();
			muscle->ApplyForceToBody();
		}
		// TODO1: Verify that setForces does set TORQUE when called on joints (XS)
		Eigen::Vector3d T_LHip{GetLHipT(), 0, 0};
		Eigen::Vector3d T_RHip{GetRHipT(), 0, 0};
		Eigen::VectorXd T_RKnee = Eigen::VectorXd::Zero(1);
		T_RKnee << GetLKneeT();
		Eigen::VectorXd T_LKnee = Eigen::VectorXd::Zero(1);
		T_LKnee << GetRKneeT();
		mCharacter->GetSkeleton()->getBodyNode("FemurL")->getParentJoint()->setForces(T_LHip); // does this apply a force or torque?
		mCharacter->GetSkeleton()->getBodyNode("FemurR")->getParentJoint()->setForces(T_RHip); 
		mCharacter->GetSkeleton()->getBodyNode("TibiaL")->getParentJoint()->setForces(T_LKnee);
		mCharacter->GetSkeleton()->getBodyNode("TibiaR")->getParentJoint()->setForces(T_RKnee);

		if(mSimCount == mRandomSampleIndex)
		{
			auto& skel = mCharacter->GetSkeleton();
			auto& muscles = mCharacter->GetMuscles();

			int n = skel->getNumDofs();
			int m = muscles.size();
			Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);	//torque due to active muscle force?
			Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);		//torque due to passive muscle force?

			for(int i=0;i<muscles.size();i++)
			{
				auto muscle = muscles[i];
				// muscle->Update();
				Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
				auto Ap = muscle->GetForceJacobianAndPassive();

				JtA.block(0,i,n,1) = Jt*Ap.first;
				Jtp += Jt*Ap.second;
			}

			mCurrentMuscleTuple.JtA = GetMuscleTorques();
			Eigen::MatrixXd L = JtA.block(mRootJointDof,0,n-mRootJointDof,m);
			Eigen::VectorXd L_vectorized = Eigen::VectorXd((n-mRootJointDof)*m);
			for(int i=0;i<n-mRootJointDof;i++)
			{
				L_vectorized.segment(i*m, m) = L.row(i);
			}
			mCurrentMuscleTuple.L = L_vectorized;
			mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof,n-mRootJointDof);
			mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
			mMuscleTuples.push_back(mCurrentMuscleTuple);
		}
	}
	else
	{
		GetDesiredTorques();
		mCharacter->GetSkeleton()->setForces(mDesiredTorque);
	}

	mWorld->step();
	// Eigen::VectorXd p_des = mTargetPositions;
	// //p_des.tail(mAction.rows()) += mAction;
	// mCharacter->GetSkeleton()->setPositions(p_des);
	// mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
	// mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
	// mWorld->setTime(mWorld->getTime()+mWorld->getTimeStep());

	mSimCount++;
}

Eigen::VectorXd
Environment::
GetDesiredTorques()
{
	Eigen::VectorXd p_des = mTargetPositions;						// Retrieve target positions of joints? DOFs?
	p_des.tail(mTargetPositions.rows()-mRootJointDof) += mAction;	// updates desired position using the action (change in pos?), chosen by simNN ?? - XS
																	// mrootjointdof p_des is not modified as it represents the position of the whole skeleton?? - XS
	mDesiredTorque = mCharacter->GetSPDForces(p_des);				// So messing with tau in GetSPDForces changes the desired torque for the muscles to acquire
																	// not really what we wanted
	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}
Eigen::VectorXd
Environment::
GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index,JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}
	
	return mCurrentMuscleTuple.JtA;


}
double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}
double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}
double exp_of_squared(double val,double w)
{
	return exp(-w*val*val);
}


bool
Environment::
IsEndOfEpisode()
{
	bool isTerminal = false;
	
	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];
	if(root_y<1.3)
		isTerminal =true;
	else if (dart::math::isNan(p) || dart::math::isNan(v))
		isTerminal =true;
	else if(mWorld->getTime()>10.0)
		isTerminal =true;
	
	return isTerminal;
}

/**
 * @brief Get the State. Note that "body nodes" are links (?) - XS
 * 
 * @return Eigen::VectorXd representation of the state, which is:
 * (position of each link COM, velocity of each link COM (last item is pelvis (root node) velocity), how far through gait cycle)
 */
Eigen::VectorXd 
Environment::
GetState()
{
	auto& skel = mCharacter->GetSkeleton();					// Retrieve the simulation object
	dart::dynamics::BodyNode* root = skel->getBodyNode(0);	// Retrieve the root body node (the pelvis?)
	int num_body_nodes = skel->getNumBodyNodes();			// Compute total number of links

	// Initialise and configure link position and velocity vectors - 3 dim vector to describe each pos and vel
	Eigen::VectorXd p,v;			// p - "3D position of bones", v - "linear velocity of bones", phi - "[0, 1], phase variable"
	p.resize((num_body_nodes-1)*3);
	v.resize((num_body_nodes)*3);

	// Populate the pos and vel vectors
	for(int i = 1;i<num_body_nodes;i++)	// i starts at 1, to skip the root node (pelvis)
	{
		p.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOM(root);				// Get the pos of indexed link COM relative to pelvis
		v.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOMLinearVelocity();	// Get the vel of indexed link COM expressed in arbitrary frames?? not sure what this means
	}
	
	// Position of pelvis is not recorded in state, but velocity of it is added into vel vector here
	v.tail<3>() = root->getCOMLinearVelocity();	

	double t_phase = mCharacter->GetBVH()->GetMaxTime();
	double phi = std::fmod(mWorld->getTime(),t_phase)/t_phase;	// fraction of how far through the gait cycle the sim is 
																// modulus is used, so that when one full cycle is up, this progress value
																// wraps around to 0

	// scaled to match BVH reference movement??
	p *= 0.8;	
	v *= 0.2;

	// Concatenate pos, vel, and gait cycle progress into one state vector
	Eigen::VectorXd state(p.rows()+v.rows()+1);	
	state<<p,v,phi;

	return state;
}
void 
Environment::
SetAction(const Eigen::VectorXd& a)
{
	mAction = a*0.1;

	double t = mWorld->getTime();

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;
	// std::cout << mTargetPositions;
	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	mAverageActivationLevels.setZero();
}

void 
Environment::
SetExoTorques(Eigen::VectorXd Ts)
{
	SetLHipT(Ts[0]);
	SetRHipT(Ts[1]);
	SetLKneeT(Ts[2]);
	SetRKneeT(Ts[3]);
}

/**
 * @brief Get Exo Joint Torques
 * 
 * @return Eigen::VectorXd containing joint torques in the order:
 * [L_hip, L_knee, R_Hip, R_Knee]
 */
Eigen::VectorXd
Environment::
GetExoTorques()
{
	Eigen::VectorXd joint_torques(4);
	joint_torques << GetLHipT(), GetLKneeT(), GetRHipT(), GetRKneeT();
	return joint_torques;
}

/**
 * @brief Function to calculate the current reward.
 * 
 * @return double representation of the reward.
 */
double
Environment::
GetReward()
{
	auto& skel = mCharacter->GetSkeleton();	// Retrieves the simulation model

	Eigen::VectorXd cur_pos = skel->getPositions();		// Retrieves the current joint positions of the simulation
	Eigen::VectorXd cur_vel = skel->getVelocities();	// Retrieves the current joint velocities of the simulation

	Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);	// Compute the difference between actual and target joint positions
	Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);	// Compute the difference between actual and target joint velocities
	
	// Make zero vectors of size equal to the number of DOFs. One for position difference, one for velocity difference:
	Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
	Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());

	const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();	// ????

	for(auto ss : bvh_map)
	{
		auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();	// index thru each joint
		int idx = joint->getIndexInSkeleton(0);	// Retrieve index of joint, so it can be found in p_diff_all
		if(joint->getType()=="FreeJoint")
			continue;	// if the joint is a 'FreeJoint' leave its corresponding p_diff = to 0
		else if(joint->getType()=="RevoluteJoint")
			p_diff[idx] = p_diff_all[idx];	// if the joint is a 'RevoluteJoint' set p_diff = to the difference between target and actual pos
		else if(joint->getType()=="BallJoint")
			p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);	// 'BallJoint' can rotate on 3 axis -> 3 numbers to describe positional differences
	}

	// ????
	auto ees = mCharacter->GetEndEffectors();	// list of end effector objects/positions - Head, hands, and feet.
	// std::cout << ees.size()
	Eigen::VectorXd ee_diff(ees.size()*3);		// make a vector 3 times the size of end effector list, for recording end effector position difference in all 3 directions?
	Eigen::VectorXd com_diff;					// initialise a vector for the difference between target and actual centre of mass?

	for(int i =0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) = ees[i]->getCOM();	// Retrieve the position of the C.O.M. of each end effector
	com_diff = skel->getCOM();						// Retrieve the position of the C.O.M. of the whole sim

	// For calculation purposes, move the simulation to the target positions:
	skel->setPositions(mTargetPositions);				
	skel->computeForwardKinematics(true,false,false);

	// Now that the model is moved to the target position, we can find the difference between
	// Target C.O.M./E.E. positions and actual positions:
	com_diff -= skel->getCOM();
	for(int i=0;i<ees.size();i++)
		ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;	// com_diff is added here to account for the movement of the whole model -> 
																// does this imply that the E.E. position is described relative to the C.O.M.? 

	// Now that the calculations are done, put the model back to the actual position:
	skel->setPositions(cur_pos);
	skel->computeForwardKinematics(true,false,false);

	// The norm of the differences are found. I think the second parameter is a weighting for the value, but I am unsure because
	// weights are applied when computing the reward on line 410 (w_q, w_v are weights)
	double r_q = exp_of_squared(p_diff,2.0);
	double r_v = exp_of_squared(v_diff,0.1);		// isn't v_diff just a vector of zeros at this point?
	double r_ee = exp_of_squared(ee_diff,40.0);	
	double r_com = exp_of_squared(com_diff,10.0);

	double r = r_ee*(w_q*r_q + w_v*r_v);			// Why is r_com computed, but not used -> accounted for indirectly by r_ee? 
													// even so, why would you compute it then? -XS

													// Looked back at commit history of MASS and found r_com being used 
													// They just did not remove it here - ZB

	return r;
}

/**self.sim_env.GetStates()
 * @brief 
 * 
 * @return double representation of a modified reward for the exo joints
 */
double Environment::GetTReward() {
	double r_T = exp_of_squared(GetLHipT(), 10.0) + exp_of_squared(GetRHipT(), 10.0) + exp_of_squared(GetLKneeT(), 10.0) + exp_of_squared(GetRKneeT(), 10.0);
	double rT = GetReward() - r_T;

	return rT;
}