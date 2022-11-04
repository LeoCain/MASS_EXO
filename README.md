# MASS(Muscle-Actuated Skeletal System)

![Teaser](png/Teaser.png)
## Abstract
This repository is a fork of MASS, which is a musculoskeletal simulation environment written in DART C++. Using Proximal Policy Optimisation (PPO), MASS trains a "muscle net" to select muscle activation patterns that imitate the inputted reference motion. There are a few different reference motions available in the repo, but this fork focusses on walking gait.

This fork uses MASS as part of an AI project which aims to train an exoskeleton control agent for stroke rehabilitation, by both correcting gait, and maximising patient participation.

## MASS Publications

Seunghwan Lee, Kyoungmin Lee, Moonseok Park, and Jehee Lee 
Scalable Muscle-actuated Human Simulation and Control, 
ACM Transactions on Graphics (SIGGRAPH 2019), Volume 37, Article 73. 

Project Page : http://mrl.snu.ac.kr/research/ProjectScalable/Page.htm

Youtube : https://youtu.be/a3jfyJ9JVeM

Paper : http://mrl.snu.ac.kr/research/ProjectScalable/Paper.pdf

## How to install (Verified only for Ubuntu 20)

### Install TinyXML, Eigen, OpenGL, assimp, Python3, etc...

```bash
sudo apt-get install libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui libbullet-dev pybind11-dev
```

### Install DART 6.8

You will need to install DART for C++ for this project. You do not need to install DART for python (dartpy) but if you do, make sure you are not using conda or venv, as this will cause a problem with the install, and make it very difficult to re-install dartpy (this advice was accurate as of 04/11/2022).

Installation manual from DART: (http://dartsim.github.io/install_dart_on_ubuntu.html)

Please refer to http://dartsim.github.io/ (Install version 6.8)

### Pytorch

pytorch(https://pytorch.org/)
Weird issue with torchvision can cause an error which mentions failing to "load image python extension." if you get this error, ive found the following versions seem to work:

**torch**: 1.11.0

**torchvision**: 0.12.0

**cuda**: 10.2

There are probably other combinations that will work, just try downloading all the new versions first.

### Other imports with pip

```bash
pip3 install numpy matplotlib ipython
```

## How to compile and run

### Resource

Our system requires a reference motion to imitate. We provide sample references such as walking, running, and etc... 

To learn and simulate, we should provide such a meta data. We provide default meta data in /data/metadata.txt. We parse the text and set the environment. Please note that the learning settings and the test settings should be equal.(metadata.txt should not be changed.)

Metadata files are define below:
/data/metadata.txt - This is the original metadata file for MASS, and specifies walking gait training, with an unmodified model. It is not recommended to change this file.

/data/metadata_crip.txt - This metadata file is used when the user wishes to specify a different models to use for training. For example, the current metadata_crip.txt specifies a muscle file which has knee flexors/extensors which have been reduced to 10% strength.

There is no reason why the user couldn't make more metadata files, if they desire.

### Setup
After cloning this repo, navigate to the MASS_EXO folder and execute the following to make the build directory, and compile for the first time:

```bash
mkdir build
cd build
cmake .. 
make -j8
```

In order to the exoskeleton agent, the user must COPY (not move) the two following files to the build folder:

Exo_model.py

MASS_env.py

Unfortunately, if the user wishes to modify these two files, the copies must be deleted from the build folder, then the originals can be modified, and re-copied back into the build folder. This could probably be fixed by making a modification to the cmake files, but due to time constraints, I did not get around to it.

### Compile and Run simulation (not the exoskeleton agent)

Navigate to the MASS_EXO folder, cmake and compile:

```bash
cd build
cmake .. 
make -j8
```

**Train the muscle activation net (train the model to walk using muscles)**
```bash
cd python
python3 main.py -d ../data/metadata.txt # The metadata file can be switched out, if desired
# Or if the user wishes to continue training from where another session left off:
# "max" keyword specifies the neural nets at ../nn/max.pt ../nn/max_muscle.pt
# Which is the NN weight file for the weights that performed best in the last run.
python3 main.py -d ../data/metadata.txt -m max  
```

All the training networks are saved in /nn folder. Usually a reward of ~80 will be a good gait.

**Run the UI without the muscle activation agent**
```bash
./render/render ../data/metadata.txt  # model will just fall through the floor as it is unactuated.
```

**Run simulation with trained model control nets**
```bash
./render/render ../data/metadata.txt ../nn/max.pt ../nn/max_muscle.pt
```

**If you are simulating with the torque-actuated model:**
```bash
source /path/to/virtualenv/
./render/render ../data/metadata.txt ../nn/xxx.pt
```

### Training and Running the Exoskeleton Agent
Make sure any changes are compiled:
```bash
cd build  
cmake ..  # Only needs to be run if cmake files are modified
make -j8
```

**Run training of the exoskeleton agent**
```bash
# Metadata filepath and ANN filepaths are set within Exo_agent/RLlib_MASS.py
python3 Exo_agent/RLlib_MASS.py
```

You will observe a lot of warnings - the repeated ones are normal, and from inside the RLlib library - I believe the devs are working on fixing this. As the training runs, it will save a reward function progression graph at Exo_agent/Plots/RewardPlot_torch.png. it will also report on "min episode reward, mean episode reward, max episode reward, mean episode length, checkpoint filename." In addition to this, r_T and r_dT will be printed out periodically - these are reports of the average applied torque and average smoothness of applied torque (see MASS_env.py for specifics).

**Run the simulation with exoskeleton agent applied**
```bash
# You must specify the metadata file, the simulation nets, and the exoskeleton agent check point. If any of these are wrong/incompatible it will most likely throw an error.
./Exo_agent/exo_render ..path/to/metadata/file ..path/to/simNN ..path/to/muscleNN ../path/to/exoskeleton/agent/checkpoint

# For example, I used:
./Exo_agent/exo_render ..data/metadata_crip.txt ../nn_knee_weak_rq/max.pt ../nn_knee_weak_rq/max_muscle.pt ../Exo_agent/policies/checkpoint_003000
```

This may take a little while to boot up, but eventually the UI should appear.

### UI usage guide:
The UI boots up as a separate window - there are no buttons but the view can be moved by dragging the screen using your mouse. There are also some key commands:

spacebar: Start/stop simulation.

R: Restart simulation.

O: Toggle between basic box model and skeleton render model.

F: Follow the model with the viewer.

S: step the simulation forward once.

## Repository Summary

### Top-Level Directory Summary
**Exo_agent:** Contains all the files needed to train, run and render the exoskeleton agent within the MASS simulation environment. Also contains folders for raw csv data and plots.

**Thesis_Folder:** Contains plots from different simulation that have been run (for both the MASS SIM, and the exoskeleton agent SIM).

**core:** Contains most of the functional code for the MASS simulation, including environment initialisation and stepping, reward calculations, etc.

**data:** Contains metadata files, skeletal and muscular XML config files, BVH motion reference files, and object files for the skeleton.

**png:** Directory that the MASS reward function progression graph is saved to.

**python:** Contains most of the functional code for running the MASS training algorithm.

**render:** Contains the main loop code for the MASS simulation, and the software for rendering it.

**nn_example:** Contains an example of the two weight files for the MASS sim simulation. These are for the unmodified XML config files, and the resultant sim can be viewed by executing:
```bash
./render/render ../data/metadata.txt ../nn_example/max.pt ../nn_example/max_muscle.pt
```

## Model Creation & Retargeting (This module is ongoing project.)

### This requires Maya and MotionBuilder.

There is a sample model in data/maya folder that I generally use. Currently if you are trying to edit the model, you have to make your own export maya-python code and xml writer so that the simulation code correctly read the musculoskeletal structure. 
There is also a rig model that is useful to retarget a new motion. 
