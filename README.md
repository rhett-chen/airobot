# Acknowledgement
The main part of the code is cloned from <https://github.com/Improbable-AI/airobot> and I applied it to my project.
# Test grasp
Currently, only single object testing is supported. If you want to test multiple objects, you need to modify the code 
in ```scripts/test_grasp/``` yourself.
## Get data
Start simulation environment to get image, depth image, point cloud and other data. First, you need to set the parameters 
in ```scripts/test_grasp/options.py```, including ```object_pos, object_ori, cam_focus_pt, cam_pos, cam_height, cam_width```.
Run the following command:

```bash
cd test_grasp
python get_data.py
```
Then you will get a directory named data_x, x is the number of folders in the ```test_grasp/``` dir. There are six files in data_x dir:
```depth.jpg, depth.npy, rgb.jpg, pc.npy, info.npy, info.txt, seg.py``` The contents of info.txt/npy are identical, they contain information
about camera and object, txt is easy to read and npy is easy to load. Note that currently test_grasp only supports loading a single object.
Besides, object type and size are set in  ```test_grasp/util.py  load_single_object()```
## Get grasps
After obtainng relevant data, you can use different models to generate grasping pose. For example:

<https://github.com/rhett-chen/GPNet>

<https://github.com/rhett-chen/6dof-graspnet>

<https://github.com/rhett-chen/graspnet-baseline>

You need to move the grasp pose file generated by the above method to data_x folder.
## Visualize grasps
The visualization script will draw the point cloud data obtained in step one and the grasping posture obtained in 
 step two in a scene to check the quality of the grasping poses.
You need to set ```data_path, pose_file, method, pc_gripper_version``` in ```options.py```. ```data_path``` directing to data_x dir obtained in step one,
```pose_file``` directing npy/txt file obtained in step two.```pc_gripper_version``` specify the gripper point cloud version: panda or customized. 
 ```method``` specify which method generates ```pose_file``` because grasp pose representations of different methods are different. 
At present, only GPNet and 6dof-graspnet are supported.

Run the following command:
```bash
cd test_grasp
python visualize_grasp.py
```
## Grasp in simulation environment
The grasp script will use the robot arm to grasp object in simulation environment. You need to set ```data_path, pose_file
, robot_arm, control_mode, method``` Note that the (pose, type, size) of the object must be consistent with the settings in Get data.

Run the following command:
```bash
cd test_grasp
python grasp.py
```


The following is the original readme.
# AIRobot

[![Documentation Status](https://readthedocs.org/projects/airobot/badge/?version=latest)](https://airobot.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Improbable-AI/airobot/blob/master/LICENSE)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Improbable-AI/airobot.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Improbable-AI/airobot/context:python)

AIRobot is a python library to interface with robots. It follows a similar architecture from [PyRobot](https://pyrobot.org).
- [Citation](#citation)
- [Installation](#installation)
- [Supported Robots](#supported-robots)
- [Getting Started](#getting-started)
- [Credits](#credits)
- [Build API Docs](#build-api-docs)
- [Run Tests](#run-tests)
- [License](#license)
- [Coding Style](#coding-style)
- [Acknowledgement](#acknowledgement)

## Citation

If you use AIRobot in your research, please use the following BibTeX entry.
```
@misc{airobot2019,
  author =       {Tao Chen and Anthony Simeonov and Pulkit Agrawal},
  title =        {{AIRobot}},
  howpublished = {\url{https://github.com/Improbable-AI/airobot}},
  year =         {2019}
}
```

## Installation

### Pre-installation steps for ROS (only required if you want to use the real robot)

#### Option 1:
If you want to use ROS to interface with robots, please install [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) first.

If you want to use ROS for UR5e robots, please install [ur5e driver](https://github.com/Improbable-AI/ur5e_robotiq_2f140). 

If you are using RealSense cameras, please follow the instructions [here](https://github.com/IntelRealSense/realsense-ros#installation-instructions) or [here](https://github.com/Improbable-AI/camera_calibration/tree/qa).

#### Option 2:
Use the dockerfile provided in this library. Checkout the [instructions](https://github.com/Improbable-AI/airobot/blob/master/docker) for docker usage. **Docker is not needed if you just want to use the pybullet simulation environment.**

### Install AIRobot

You might want to install it in a virtual environment. 

If you are using ROS, you can use [virtualenv](https://virtualenv.pypa.io/en/latest/installation/). Note that Anaconda doesn't work well with ROS. And only Python 2.7 is recommended for ROS at this point.

If you only want to use the robots in the PyBullet simulation environment, then you can use Python 2.7 or Python 3.7. And you can use either [virtualenv for Python 2.7](https://virtualenv.pypa.io/en/latest/installation/), [venv for Python 3.7](https://docs.python.org/3.7/tutorial/venv.html) or [Anaconda](https://docs.anaconda.com/anaconda/install/linux/).

```bash
git clone https://github.com/Improbable-AI/airobot.git
cd airobot
pip install -e .
```

## Supported Robots
* [UR5e](https://www.universal-robots.com/products/ur5-robot/) with ROS
* [UR5e](https://www.universal-robots.com/products/ur5-robot/) in PyBullet
* [ABB YuMi](https://new.abb.com/products/robotics/industrial-robots/irb-14000-yumi) in PyBullet
* [Franka Robot](https://frankaemika.github.io/docs/) in Pybullet

If you want to use [Sawyer](https://www.rethinkrobotics.com/sawyer) or [LoCoBot](https://locobot-website.netlify.com/), you can use [PyRobot](https://pyrobot.org).

## Getting Started
A sample script is shown below. You can find more examples [here](https://github.com/Improbable-AI/airobot/examples)

```python
from airobot import Robot
# create a UR5e robot in pybullet
robot = Robot('ur5e',
              pb=True,
              pb_cfg={'gui': True})
robot.arm.go_home()
robot.arm.move_ee_xyz([0.1, 0.1, 0.1])
```

## Credits
**AIRobot** is maintained by the Improbable AI team at MIT. It follows a similar architecture in [PyRobot](https://pyrobot.org). Contributors include:
* [Tao Chen](https://taochenshh.github.io/)
* [Anthony Simeonov](https://anthonysimeonov.github.io/)
* [Pulkit Agrawal](http://people.csail.mit.edu/pulkitag/)


## Build API Docs

Run the following commands to build the API webpage.

```bash
cd docs
./make_api.sh
```

Then you can use any web browser to open the API doc (**`docs/build/html/index.html`**)

## Run tests

[pytest](https://docs.pytest.org/en/latest/) is used for unit tests. TODO: more test cases need to be added.
```bash
cd airobot/tests
./run_pytest.sh
```

## License
MIT license

## Coding Style

AIRobot uses [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings. We use [Flake8](https://pypi.org/project/flake8/) to perform additional formatting and semantic checking of code.

## Acknowledgement

We gratefully acknowledge the support from Sony Research Grant and DARPA Machine Common Sense Program.


