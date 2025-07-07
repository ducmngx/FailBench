# GenAISim - Saad_Dev Branch

## TODO
1. Write models/model_builder.py for robustness - this file should be able to combine different robots and different sensors depending on the arguments given to the file and the configuration files of the robots. 
    - Right now the sensors are hard-coded on top of the jackal so we can go to the next step. (Done)
2. Write a python wrapper around the rangefinder sensors to take in N amount of sensor values and output a single array that concatenates all the sensor readings. For eg, VLP16 has 360x16 rangefinder sensors. The wrapper should take in the rf_{horizontal}_{vertical} sensor and produce a single array of size (360, 16).
    - Regarding where this file should exist, if we follow robosuite then we can either put all xml/mesh files in an "assets" folder and then have a seperate folder for these wrappers around sensors. Or we can have a utils folder in root and put the wrappers there. Im inclined to the first approach since the wrappers are specific and not general.
3. Write a navigation planner that takes in sensor input, goal coordinates, and current coordinates and outputs a plan in the form of a sequence linear/angular velocities to get to the goal.

## Code Organization
Currently its a bit messy. ill fix it over the week. The jackal xml file can be found in model/jackal.xml. I want to base the code organization on a mujoco example I found online - [Franka Panda](https://github.com/justagist/mujoco_panda/tree/master). We can adjust as we develop this repo.

On Aaron's suggestion, I'll follow [Robosuite](http://github.com/ARISE-Initiative/robosuite/tree/master) for writing the controllers. 

We can switch to follow Robosuite for the models too (rather than Franka Panda). For now (06.29.24) we will follow Franka.

## Helpful things to know
- mujoco can load urdf files with minimal changes. However its better to convert a URDF file to a mujoco xml file via `./compile /path/to/model.urdf /path/to/mujuco_model.xml`. 
    - References:
        - https://mujoco.readthedocs.io/en/stable/modeling.html#modeling
        - https://github.com/robotlearning123/dual_ur5_husky_mujoco/tree/dual_ur5_husky_mujoco
- Parallel Training:
    - "On a side note, what we are doing here w Hopkins is I’m building the jackal and set up 90 house environments (matterport 3d). They plan to train with IsaacLab, which should allow thousands if episodes a second with parallel envs. In the future, if we plan to do training, we should take advantage of that too"


