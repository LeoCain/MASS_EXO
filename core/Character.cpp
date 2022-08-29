#include "Character.h"
#include "BVH.h"
#include "DARTHelper.h"
#include "Muscle.h"
#include <dart/utils/urdf/DartLoader.hpp>
#include <tinyxml.h>
using namespace dart;
using namespace dart::dynamics;
using namespace MASS;
Character::
Character()
	:mSkeleton(nullptr),mBVH(nullptr),mTc(Eigen::Isometry3d::Identity())
{

}

void
Character::
LoadSkeleton(const std::string& path,bool create_obj)
{
	mSkeleton = BuildFromFile(path,create_obj);
	std::map<std::string,std::string> bvh_map;
	TiXmlDocument doc;
	doc.LoadFile(path);
	TiXmlElement *skel_elem = doc.FirstChildElement("Skeleton");

	for(TiXmlElement* node = skel_elem->FirstChildElement("Node");node != nullptr;node = node->NextSiblingElement("Node"))
	{
		if(node->Attribute("endeffector")!=nullptr)
		{
			std::string ee =node->Attribute("endeffector");
			if(ee == "True")
			{
				mEndEffectors.push_back(mSkeleton->getBodyNode(std::string(node->Attribute("name"))));
			}
		}
		TiXmlElement* joint_elem = node->FirstChildElement("Joint");
		if(joint_elem->Attribute("bvh")!=nullptr)
		{
			bvh_map.insert(std::make_pair(node->Attribute("name"),joint_elem->Attribute("bvh")));
		}
	}
	
	mBVH = new BVH(mSkeleton,bvh_map);
}
void
Character::
LoadMuscles(const std::string& path)
{
	
	TiXmlDocument doc;
	if(!doc.LoadFile(path)){
		std::cout << "Can't open file : " << path << std::endl;
		return;
	}

	TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
	for(TiXmlElement* unit = muscledoc->FirstChildElement("Unit");unit!=nullptr;unit = unit->NextSiblingElement("Unit"))
	{
		std::string name = unit->Attribute("name");
		double f0 = std::stod(unit->Attribute("f0"));
		double lm = std::stod(unit->Attribute("lm"));
		double lt = std::stod(unit->Attribute("lt"));
		double pa = std::stod(unit->Attribute("pen_angle"));
		double lmax = std::stod(unit->Attribute("lmax"));
		mMuscles.push_back(new Muscle(name,f0,lm,lt,pa,lmax));
		int num_waypoints = 0;
		for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))	
			num_waypoints++;
		int i = 0;
		for(TiXmlElement* waypoint = unit->FirstChildElement("Waypoint");waypoint!=nullptr;waypoint = waypoint->NextSiblingElement("Waypoint"))	
		{
			std::string body = waypoint->Attribute("body");
			Eigen::Vector3d glob_pos = string_to_vector3d(waypoint->Attribute("p"));
			if(i==0||i==num_waypoints-1)
			// if(true)
				mMuscles.back()->AddAnchor(mSkeleton->getBodyNode(body),glob_pos);
			else
				mMuscles.back()->AddAnchor(mSkeleton,mSkeleton->getBodyNode(body),glob_pos,2);

			i++;
		}
	}
	

}


// TODO
dart::dynamics::SkeletonPtr
Character::
LoadExo(const std::string& path)
{	
	// load the exo model
	dart::utils::DartLoader loader;
    dart::dynamics::SkeletonPtr model = loader.parseSkeleton("/home/medicalrobotics/MASS_EXO/data/exo_model.xml");
    model->setName("model");

    // Position its base in a reasonable way
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() = Eigen::Vector3d(-0.65, 0.0, 0.0);
    model->getJoint(0)->setTransformFromParentBodyNode(tf);

    // Get it into a useful configuration
    model->getDof(1)->setPosition(140.0 * M_PI / 180.0);
    model->getDof(2)->setPosition(-140.0 * M_PI / 180.0);

	return model;
	
}



void
Character::
LoadBVH(const std::string& path,bool cyclic)
{
	if(mBVH ==nullptr){
		std::cout<<"Initialize BVH class first"<<std::endl;
		return;
	}
	mBVH->Parse(path,cyclic);
}
void
Character::
Reset()
{
	mTc = mBVH->GetT0();
	mTc.translation()[1] = 0.0;
}
void
Character::
SetPDParameters(double kp, double kv)
{
	int dof = mSkeleton->getNumDofs();
	mKp = Eigen::VectorXd::Constant(dof,kp);	
	mKv = Eigen::VectorXd::Constant(dof,kv);	
}
/**
 * @brief: Potentially the function to modify with augmented exo torque?
 * 
 * @param p_desired: desired positions of joints? of DOFs?
 * @return: Eigen::VectorXd which describes the joint torques (maybe?)
 */
Eigen::VectorXd
Character::
GetSPDForces(const Eigen::VectorXd& p_desired)
{
	Eigen::VectorXd q = mSkeleton->getPositions();		// Retrieve the current joint positions
	Eigen::VectorXd dq = mSkeleton->getVelocities();	// Retrieve the current joint velocities
	double dt = mSkeleton->getTimeStep();				// Retrieve the size of the time step
	// Eigen::MatrixXd M_inv = mSkeleton->getInvMassMatrix();
	Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt*mKv.asDiagonal())).inverse();	// mass matrix and something else inverted?
																											// some form of inertial/momentum adjustment?

	Eigen::VectorXd qdqdt = q + dq*dt;	// Compute the new position of the joints, assuming constant velocity during this time step.

	Eigen::VectorXd p_diff = -mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt,p_desired));
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq);
	Eigen::VectorXd ddq = M_inv*(-mSkeleton->getCoriolisAndGravityForces()+p_diff+v_diff+mSkeleton->getConstraintForces());	// a = F/M or ddtheta = T/I?

	Eigen::VectorXd mod = Eigen::VectorXd::Zero(p_diff.size()); // size is 56 = number of DOFs??
	// std::cout << "*=====================" << p_diff.size() << "=========================*";
	mod << 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000;
	Eigen::VectorXd tau = p_diff + v_diff + mod - dt*mKv.cwiseProduct(ddq);
	// Eigen::VectorXd tau = mod;

	tau.head<6>().setZero();

	return tau;
}
Eigen::VectorXd
Character::
GetTargetPositions(double t,double dt)
{
	// std::cout<<"GetTargetPositions"<<std::endl;
	Eigen::VectorXd p = mBVH->GetMotion(t);	
	Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(p.head<6>());
	T_current = mBVH->GetT0().inverse()*T_current;
	Eigen::Isometry3d T_head = mTc*T_current;
	Eigen::Vector6d p_head = dart::dynamics::FreeJoint::convertToPositions(T_head);
	p.head<6>() = p_head;
	
	
	if(mBVH->IsCyclic())
	{
		double t_mod = std::fmod(t, mBVH->GetMaxTime());
		t_mod = t_mod/mBVH->GetMaxTime();

		double r = 0.95;
		if(t_mod>r)
		{
			double ratio = 1.0/(r-1.0)*t_mod - 1.0/(r-1.0);
			Eigen::Isometry3d T01 = mBVH->GetT1()*(mBVH->GetT0().inverse());
			double delta = T01.translation()[1];
			delta *= ratio;
			p[5] += delta;
		}



		double tdt_mod = std::fmod(t+dt, mBVH->GetMaxTime());
		if(tdt_mod-dt<0.0){
			Eigen::Isometry3d T01 = mBVH->GetT1()*(mBVH->GetT0().inverse());
			Eigen::Vector3d p01 = dart::math::logMap(T01.linear());
			p01[0] =0.0;
			p01[2] =0.0;
			T01.linear() = dart::math::expMapRot(p01);

			mTc = T01*mTc;
			mTc.translation()[1] = 0.0;
		}
	}
	

	return p;
}

std::pair<Eigen::VectorXd,Eigen::VectorXd>
Character::
GetTargetPosAndVel(double t,double dt)
{
	Eigen::VectorXd p = this->GetTargetPositions(t,dt);
	Eigen::Isometry3d Tc = mTc;
	Eigen::VectorXd p1 = this->GetTargetPositions(t+dt,dt);
	mTc = Tc;

	return std::make_pair(p,(p1-p)/dt);
}