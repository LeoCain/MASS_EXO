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
/**
 * Main loop file for running the rendered sim WITH the 
 * torque actor enabled.
 */

MASS::exo_Window* window;

// catches cntrl-C and graphs data -> depreciated
// void SIGINT_handler(sig_atomic_t s){
// 	std::cout << "\nSIGINT caught: Saving data, Finalising plots...";
// 	window->Plot_And_Save();
// 	exit(1); 
// }

int main(int argc,char** argv)
{	
	// signal (SIGINT,SIGINT_handler); //catches cntrl-C and graphs data -> depreciated

	// Create a new simulation environment 
	MASS::Environment* env = new MASS::Environment();

	// check if command line args are correct
	if(argc!=5)
	{
		std::cout<<"Provide metadata.txt, benjaSIM nets, exo agent"<<std::endl;
		return 0;
	}
	// initialise environment for MASS and DART
	env->Initialize(std::string(argv[1]),true);
	glutInit(&argc, argv);

	// Setup render of environment and torque actor agent
	window = new MASS::exo_Window(env, argv[2], argv[3], argv[4]);
	
	// run simulation
	window->initWindow(1920,1080,"gui");
	glutMainLoop();
}
