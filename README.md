# MASS(Muscle-Actuated Skeletal System)

![Teaser](png/Teaser.png)
## Abstract

This code implements a basic simulation and control for full-body **Musculoskeletal** system. Skeletal movements are driven by the actuation of the muscles, coordinated by activation levels. Interfacing with python and pytorch, it is available to use Deep Reinforcement Learning(DRL) algorithm such as Proximal Policy Optimization(PPO).

This fork aims to use this existing repo as part of an AI project which aims to train an EXO agent for stroke rehabilitation

## Publications

Seunghwan Lee, Kyoungmin Lee, Moonseok Park, and Jehee Lee 
Scalable Muscle-actuated Human Simulation and Control, 
ACM Transactions on Graphics (SIGGRAPH 2019), Volume 37, Article 73. 

Project Page : http://mrl.snu.ac.kr/research/ProjectScalable/Page.htm

Youtube : https://youtu.be/a3jfyJ9JVeM

Paper : http://mrl.snu.ac.kr/research/ProjectScalable/Paper.pdf

## How to install

### Install TinyXML, Eigen, OpenGL, assimp, Python3, etc...

```bash
sudo apt-get install libtinyxml-dev libeigen3-dev libxi-dev libxmu-dev freeglut3-dev libassimp-dev libpython3-dev python3-tk python3-numpy virtualenv ipython3 cmake-curses-gui libbullet-dev pybind11-dev
```

### Install DART 6.8

You will need to install DART for C++ for this project. If installing dartpy (as an extra), make sure you are not using conda or venv, as this will cause a problem with the install, and make it veyr difficult to re-install dartpy (I have not found a solve yet).

Please refer to http://dartsim.github.io/ (Install version 6.8)

Manual from DART(http://dartsim.github.io/install_dart_on_ubuntu.html)


### Venv

You could activate venv if you like:
```bash
virtualenv /path/to/venv --python=python3
source /path/to/venv/bin/activate
```

### Pytorch

pytorch(https://pytorch.org/)
Weird issue with torchvision can cause an error which mentions failing to "load image python extension." if you get this error, ive found the following versions seem to work:

*torch*: 1.11.0

*torchvision*: 0.12.0

*cuda*: 10.2

### Other imports with pip

- numpy, matplotlib

```bash
pip3 install numpy matplotlib ipython
```

## How to compile and run

### Resource

Our system requires a reference motion to imitate. We provide sample references such as walking, running, and etc... 

To learn and simulate, we should provide such a meta data. We provide default meta data in /data/metadata.txt. We parse the text and set the environment. Please note that the learning settings and the test settings should be equal.(metadata.txt should not be changed.)


### Compile and Run

Navigate to the MASS_EXO folder, after cloning the repo:

```bash
mkdir build
cd build
cmake .. 
make -j8
```

- Run Training
```bash
cd python
source /path/to/virtualenv/ # Only necessary if using venv
python3 main.py -d ../data/metadata.txt
```

All the training networks are saved in /nn folder.

- Run UI
```bash
source /path/to/virtualenv/ # Only necessary if using venv
./render/render ../data/metadata.txt
```

- Run Trained data
```bash
source /path/to/virtualenv/
./render/render ../data/metadata.txt ../nn/max.pt ../nn/max_muscle.pt
```

If you are simulating with the torque-actuated model, 
```bash
source /path/to/virtualenv/
./render/render ../data/metadata.txt ../nn/xxx.pt
```


## Model Creation & Retargeting (This module is ongoing project.)

### This requires Maya and MotionBuilder.

There is a sample model in data/maya folder that I generally use. Currently if you are trying to edit the model, you have to make your own export maya-python code and xml writer so that the simulation code correctly read the musculoskeletal structure. 
There is also a rig model that is useful to retarget a new motion. 
