// #include "Window.h"
#include "render_exo.h"
#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

MASS::exo_Window* window;

void SIGINT_handler(sig_atomic_t s){
	std::cout << "\nSIGINT caught: Saving data, Finalising plots...";
	window->Plot_And_Save();
	exit(1); 
}

int main(int argc,char** argv)
{	
	signal (SIGINT,SIGINT_handler);

	MASS::Environment* env = new MASS::Environment();

	if(argc!=5)
	{
		std::cout<<"Provide metadata.txt, benjaSIM nets, exo agent"<<std::endl;
		return 0;
	}
	env->Initialize(std::string(argv[1]),true);
	// if(argc==3)
	// 	env->SetUseMuscle(true);
	// else
	// 	env->SetUseMuscle(false);
	// env->SetControlHz(30);
	// env->SetSimulationHz(600);

	// MASS::Character* character = new MASS::Character();
	// character->LoadSkeleton(std::string(MASS_ROOT_DIR)+std::string("/data/human.xml"),true);
	// if(env->GetUseMuscle())
	// 	character->LoadMuscles(std::string(MASS_ROOT_DIR)+std::string("/data/muscle.xml"));
	// character->LoadBVH(std::string(MASS_ROOT_DIR)+std::string("/data/motion/walk.bvh"),true);
	
	// double kp = 300.0;
	// character->SetPDParameters(kp,sqrt(2*kp));
	// env->SetCharacter(character);
	// env->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	// env->Initialize();

	glutInit(&argc, argv);

	window = new MASS::exo_Window(env, argv[2], argv[3], argv[4]);

	// if(argc == 2)
	// {
	// 	window = new MASS::exo_Window(env);
	// }
	// else
	// {
	// 	if(env->GetUseMuscle())
	// 	{
	// 		if(argc!=4){
	// 			std::cout<<"Please provide two networks"<<std::endl;
	// 			return 0;
	// 		}
	// 		window = new MASS::exo_Window(env,argv[2],argv[3]);
	// 	}
	// 	else
	// 	{
	// 		if(argc!=3)
	// 		{
	// 			std::cout<<"Please provide the network"<<std::endl;
	// 			return 0;
	// 		}
	// 		window = new MASS::exo_Window(env,argv[2]);
	// 	}
	// }
	// if(argc==1)
	// 	window = new MASS::Window(env);
	// else if (argc==2)
	// 	window = new MASS::Window(env,argv[1]);
	// else if (argc==3)
	// 	window = new MASS::Window(env,argv[1],argv[2]);
	
	window->initWindow(1920,1080,"gui");
	glutMainLoop();
}
