/**
 * This file handles loading and application of the trained neural nets, by executing python code from C++
 * it also renders the environemtn visually.
 */
\
#include "Window.h"
#include "Environment.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include <iostream>
using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawShadow(true),mMuscleNNLoaded(false)
{
	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocusing();
	mZoom = 0.25;	
	mFocus = false;
	mNNLoaded = false;

	/** setup pybind and add relevant directory/s to its path **/
	mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = py::module::import("sys");
	
	py::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
	sys_module.attr("path").attr("insert")(1, module_dir);
	py::str module_dir2 = (std::string(MASS_ROOT_DIR)+"/Exo_agent").c_str();
	std::cout << MASS_ROOT_DIR << std::endl;
    sys_module.attr("path").attr("insert")(1, module_dir2);

	/** Execute relevant imports **/
	py::exec("import torch",mns);
	py::exec("import torch.nn as nn",mns);
	py::exec("import torch.optim as optim",mns);
	py::exec("import torch.nn.functional as F",mns);
	py::exec("import torchvision.transforms as T",mns);
	py::exec("import numpy as np",mns);
	py::exec("from Model import *",mns);
	py::exec("from plotter import Plotter", mns);

	/** instantiate the python plotter object **/
	plotter = py::eval("Plotter()", mns);

	/** Define muscle groups **/
	define_muscle_groups();

}
Window::
Window(Environment* env,const std::string& nn_path)
	:Window(env)
{
	mNNLoaded = true;

	py::str str = ("num_state = "+std::to_string(mEnv->GetNumState())).c_str();
	py::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->GetNumAction())).c_str();
	py::exec(str,mns);

	nn_module = py::eval("SimulationNN(num_state,num_action)",mns);	// Create the simulation net

	py::object load = nn_module.attr("load");						// Load the trained weights to the sim net
	load(nn_path);
}
Window::
Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path)
	:Window(env,nn_path)
{
	mMuscleNNLoaded = true;

	py::str str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetNumTotalRelatedDofs())).c_str();
	py::exec(str,mns);
	str = ("num_actions = "+std::to_string(mEnv->GetNumAction())).c_str();
	py::exec(str,mns);
	str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetMuscles().size())).c_str();
	py::exec(str,mns);

	muscle_nn_module = py::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);	// Create the muscle net

	py::object load = muscle_nn_module.attr("load");													// Load trained weights to muscle net
	load(muscle_nn_path);
}

void 
Window::
record_data()
{
	/** Load applied exo torques to be sent to plotter **/
	//just call mEnv->GetExoTorques()

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
	// mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getParentJoint()->get
	// dummy angle vector so code works
	Eigen::VectorXd joint_angles_act(4);
	Eigen::VectorXd joint_angles_ref(4);
	// Actual positions
	double l_hip_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getParentJoint()->getPositions()[0];
	double r_hip_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getParentJoint()->getPositions()[0];
	double l_knee_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TibiaL")->getParentJoint()->getPositions()[0];
	double r_knee_act = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TibiaR")->getParentJoint()->getPositions()[0];
	//Reference positions
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

	/** Send angles, torques, activations to python for plotting **/
	py::object Update_All_SIM = plotter.attr("Update_All_SIM");
    Update_All_SIM(joint_angles_act, joint_angles_ref,
					avg_activations, mEnv->GetState().tail(1));
}
void 
Window::
Plot_And_Save()
{	
	std::cout << "\nin plot_and_save";
	/** Save data to CSV **/
	py::object Save_All_To_CSV_SIM = plotter.attr("Save_All_To_CSV_SIM");
    Save_All_To_CSV_SIM();
	/** Plot the data **/
	py::object Plot_All_SIM = plotter.attr("Plot_All_SIM");
    Plot_All_SIM();
}
void 
Window::
define_muscle_groups()
{
	// L Hip Flexion DONE
	muscle_groups[LHFlex].group = {
		"L_Rectus_Femoris", "L_Rectus_Femoris1", "L_iliacus", "L_iliacus1",
		"L_iliacus2", "L_Psoas_Major", "L_Psoas_Major1", "L_Psoas_Major2", 
		"L_Psoas_Minor", 
		"L_Sartorius"
		};
	// L Hip Extension DONE
	muscle_groups[LHExt].group = {
		"L_Gluteus_Maximus", "L_Gluteus_Maximus1", "L_Gluteus_Maximus2", 
		"L_Gluteus_Maximus3", "L_Gluteus_Maximus4", "L_Adductor_Magnus", 
		"L_Adductor_Magnus1", "L_Adductor_Magnus2", "L_Adductor_Magnus3",
		"L_Adductor_Magnus4", "L_Bicep_Femoris_Longus", "L_Semitendinosus",
		"L_Semimembranosus", "L_Semimembranosus1"
	};
	// L Hip Abduction DONE
	muscle_groups[LHAbd].group = {
		"L_Gluteus_Medius", "L_Gluteus_Medius1", "L_Gluteus_Medius2", 
		"L_Gluteus_Medius3", "L_Gluteus_Minimus", "L_Gluteus_Minimus1", 
		"L_Gluteus_Minimus2", "L_Tensor_Fascia_Lata", "L_Tensor_Fascia_Lata1", 
		"L_Tensor_Fascia_Lata2"
	};
	// L Hip Adduction DONE
	muscle_groups[LHAdd].group = {
		"L_Pectineus", "L_Adductor_Longus", "L_Adductor_Longus1",
		"L_Gracilis", "L_Adductor_Brevis", "L_Adductor_Brevis1",
		"L_Adductor_Magnus", "L_Adductor_Magnus1", "L_Adductor_Magnus2",
		"L_Adductor_Magnus3", "L_Adductor_Magnus4"
	};
	// L Hip External rotation DONE
	muscle_groups[LHExtRot].group = {
		"L_Gluteus_Maximus", "L_Gluteus_Maximus1", "L_Gluteus_Maximus2", 
		"L_Gluteus_Maximus3", "L_Gluteus_Maximus4", "L_Piriformis", 
		"L_Piriformis1", "L_Quadratus_Femoris", "L_Obturator_Externus",
		 "L_Obturator_Internus", "L_Superior_Gemellus", "L_Inferior_Gemellus"
	};
	// L hip Internal rotation DONE
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
		"R_Psoas_Minor",
		 "R_Sartorius"
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
void
Window::
draw()
{	
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];
	mViewMatrix.linear() = A;
	mViewMatrix.translation() = b;

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	
	DrawGround(y);
	DrawMuscles(mEnv->GetCharacter()->GetMuscles());
	DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());

	// Eigen::Quaterniond q = mTrackBall.getCurrQuat();
	// q.x() = 0.0;
	// q.z() = 0.0;
	// q.normalize();
	// mTrackBall.setQuaternion(q);
	SetFocusing();
}

/**
 * @brief function fro handling keyboard commands while in the rendered environment - XS
 * 
 * @param _key The key press variable
 * @param _x 
 * @param _y 
 */
void
Window::
keyboard(unsigned char _key, int _x, int _y)
{
	switch (_key)
	{
	case 's': this->Step();break;				// Move forward one simulation step (?)
	case 'f': mFocus = !mFocus;break;			// Follow simulation with window
	case 'r': this->Reset();break;				// Reset sim to the start
	case ' ': mSimulating = !mSimulating;break;	// Play the simulation
	case 'o': mDrawOBJ = !mDrawOBJ;break;		// Switch between simple and complex skeleton model
	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

}
void
Window::
displayTimer(int _val)
{
	if(mSimulating)
		Step();
	glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}

/**
 * @brief Function which handles stepping through the rendering of the simulation,
 * by selecting an action, using this to set activations, then calling the Env step function - XS
 */
void
Window::
Step()
{	
	int num = mEnv->GetSimulationHz()/mEnv->GetControlHz();
	Eigen::VectorXd action;
	if(mNNLoaded)
		action = GetActionFromNN();		// Some vector from which muscle torques can be calculated?
	else
		action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
	mEnv->SetAction(action);			// Action sent to environment
	
	record_data();	// Record data to be plotted

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
	std::cout << mEnv->GetGaitReward() << '\n';
}
void
Window::
Reset()
{
	mEnv->Reset();
}
void
Window::
SetFocusing()
{
	if(mFocus)
	{
		mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
		mTrans[1] -= 0.3;

		mTrans *=1000.0;
		
	}
}

/**
 * @brief Retrieve the optimal action from the sim NN by passing the current state to it - XS
 * 
 * @return Eigen::VectorXd representation of the action (some form of target from which target muscle torques can be calculated??) 
 */
Eigen::VectorXd
Window::
GetActionFromNN()
{
	// Calls get_action function from the simulationNN python class
	return nn_module.attr("get_action")(mEnv->GetState()).cast<Eigen::VectorXd>();
}

/**
 * @brief Get an array of all muscle activations from the muscle net.
 * 
 * @param mt 
 * @return Eigen::VectorXd 
 */
Eigen::VectorXd
Window::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}
	py::object get_activation = muscle_nn_module.attr("get_activation");

	return muscle_nn_module.attr("get_activation")(mt, mEnv->GetDesiredTorques()).cast<Eigen::VectorXd>();
}

void
Window::
DrawEntity(const Entity* entity)
{
	if (!entity)
		return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}

	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}
void
Window::
DrawBodyNode(const BodyNode* bn)
{	
	if(!bn)
		return;
	if(!mRI)
		return;

	mRI->pushMatrix();
	mRI->transform(bn->getRelativeTransform());

	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	mRI->popMatrix();

}
void
Window::
DrawSkeleton(const SkeletonPtr& skel)
{
	DrawBodyNode(skel->getRootBodyNode());
}
void
Window::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	if(!mRI)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	mRI->pushMatrix();
	mRI->transform(sf->getRelativeTransform());

	DrawShape(sf->getShape().get(),va->getRGBA());
	mRI->popMatrix();
}
void
Window::
DrawShape(const Shape* shape,const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	mRI->setPenColor(color);
	if(mDrawOBJ == false)
	{
		if (shape->is<SphereShape>())
		{
			const auto* sphere = static_cast<const SphereShape*>(shape);
			mRI->drawSphere(sphere->getRadius());
		}
		else if (shape->is<BoxShape>())
		{
			const auto* box = static_cast<const BoxShape*>(shape);
			mRI->drawCube(box->getSize());
		}
		else if (shape->is<CapsuleShape>())
		{
			const auto* capsule = static_cast<const CapsuleShape*>(shape);
			mRI->drawCapsule(capsule->getRadius(), capsule->getHeight());
		}	
	}
	else
	{
		if (shape->is<MeshShape>())
		{
			const auto& mesh = static_cast<const MeshShape*>(shape);
			glDisable(GL_COLOR_MATERIAL);
			mRI->drawMesh(mesh->getScale(), mesh->getMesh());
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
		}

	}
	
	glDisable(GL_COLOR_MATERIAL);
}
void
Window::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count =0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	for(auto muscle : muscles)
	{
		auto aps = muscle->GetAnchors();
		bool lower_body = true;
		double a = muscle->activation;
		// Eigen::Vector3d color(0.7*(3.0*a),0.2,0.7*(1.0-3.0*a));
		Eigen::Vector4d color(0.4+(2.0*a),0.4,0.4,1.0);//0.7*(1.0-3.0*a));
		// glColor3f(1.0,0.0,0.362);
		// glColor3f(0.0,0.0,0.0);
		mRI->setPenColor(color);
		for(int i=0;i<aps.size();i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			mRI->pushMatrix();
			mRI->translate(p);
			mRI->drawSphere(0.005*sqrt(muscle->f0/1000.0));
			mRI->popMatrix();
		}
			
		for(int i=0;i<aps.size()-1;i++)
		{
			Eigen::Vector3d p = aps[i]->GetPoint();
			Eigen::Vector3d p1 = aps[i+1]->GetPoint();

			Eigen::Vector3d u(0,0,1);
			Eigen::Vector3d v = p-p1;
			Eigen::Vector3d mid = 0.5*(p+p1);
			double len = v.norm();
			v /= len;
			Eigen::Isometry3d T;
			T.setIdentity();
			Eigen::Vector3d axis = u.cross(v);
			axis.normalize();
			double angle = acos(u.dot(v));
			Eigen::Matrix3d w_bracket = Eigen::Matrix3d::Zero();
			w_bracket(0, 1) = -axis(2);
			w_bracket(1, 0) =  axis(2);
			w_bracket(0, 2) =  axis(1);
			w_bracket(2, 0) = -axis(1);
			w_bracket(1, 2) = -axis(0);
			w_bracket(2, 1) =  axis(0);

			
			Eigen::Matrix3d R = Eigen::Matrix3d::Identity()+(sin(angle))*w_bracket+(1.0-cos(angle))*w_bracket*w_bracket;
			T.linear() = R;
			T.translation() = mid;
			mRI->pushMatrix();
			mRI->transform(T);
			mRI->drawCylinder(0.005*sqrt(muscle->f0/1000.0),len);
			mRI->popMatrix();
		}
		
	}
	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
}
void
Window::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y) 
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}
void
Window::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4,0,-0.4);
    glColor3f(0.3,0.3,0.3);
    
    // update transform

    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }
            glBegin(face_mode);
        	for (i = 0; i < face->mNumIndices; i++)
        	{
        		int index = face->mIndices[i];

        		v[0] = (&mesh->mVertices[index].x)[0];
        		v[1] = (&mesh->mVertices[index].x)[1];
        		v[2] = (&mesh->mVertices[index].x)[2];
        		v = M*v;
        		double h = v[1]-y;
        		
        		v += h*dir;
        		
        		v[1] = y+0.001;
        		glVertex3f(v[0],v[1],v[2]);
        	}
            glEnd();
        }

    }

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        DrawAiMesh(sc, nd->mChildren[n],M,y);
    }

}
void
Window::
DrawGround(double y)
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);
	double width = 0.005;
	int count = 0;
	glBegin(GL_QUADS);
	for(double x = -100.0;x<100.01;x+=1.0)
	{
		for(double z = -100.0;z<100.01;z+=1.0)
		{
			if(count%2==0)
				glColor3f(216.0/255.0,211.0/255.0,204.0/255.0);			
			else
				glColor3f(216.0/255.0-0.1,211.0/255.0-0.1,204.0/255.0-0.1);
			count++;
			glVertex3f(x,y,z);
			glVertex3f(x+1.0,y,z);
			glVertex3f(x+1.0,y,z+1.0);
			glVertex3f(x,y,z+1.0);
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);
}