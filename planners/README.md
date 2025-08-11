# Navigation
Running `mjpython main.py` runs an A* planner on the kitchen environment to move the jackal from the start to goal positions. 

### Tasks Left
1. **Write local planner**: Navigation is done by position only, essentially jackal is jumping. We have a differential driver to move the jackal given linear and angular velocities (see diff_drive_tester.py). We need to write a local planner that uses the diff. driver.
2. **Integrate model_builder**: Jackal is manually put into the environment (see assets/environments/jackal_in_kitchen.xml). We need to integrate the model_builder branch into this code to build up jackal (and other robots) into random environments
3. **Code clean up**: Code is kind of messy. Methods need to be commented and cleaned. ReadMes need to get updated or deleted.
4. **Robot Class**: Somewhat related to navigation as currently the jackal's size is manually inputted into the code (see main.py). If we can initialize a jackal object and extract the size from it that would be better. This is related to Task 3 as it would make the platform's code cleaner in general (such as encapsulating mjData).
5. **Environment Assets**: Clearly we need to include the environment assets. Installing robocasa and generating them dynamically is not ideal. One way is to create all the desired assets and just put it in our codebase. However we should keep our repository small - see [Repository size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#repository-size-limits).


# Unrelated to Navigation
## Tasks 
1. Robot classes need to be created for ease of coding up the platform (encapsulating mjData).
2. Integrate Camera -> needed for Amir, not necessary for Aaron. 
3. Write a python wrapper around the rangefinder sensors to take in N amount of sensor values and output a single array that concatenates all the sensor readings. For eg, VLP16 has 360x16 rangefinder sensors. The wrapper should take in the rf_{horizontal}_{vertical} sensor and produce a single array of size (360, 16).
    - Not urgent as we assume that we have the whole map from mujoco.
    - Regarding where this file should exist, if we follow robosuite then we can either put all xml/mesh files in an "assets" folder and then have a seperate folder for these wrappers around sensors. Or we can have a utils folder in root and put the wrappers there. Im inclined to the first approach since the wrappers are specific and not general.


## Helpful things to know
- mujoco can load urdf files with minimal changes. However its better to convert a URDF file to a mujoco xml file via `./compile /path/to/model.urdf /path/to/mujuco_model.xml`. 
    - References:
        - https://mujoco.readthedocs.io/en/stable/modeling.html#modeling
        - https://github.com/robotlearning123/dual_ur5_husky_mujoco/tree/dual_ur5_husky_mujoco
- Parallel Training:
    - "On a side note, what we are doing here w Hopkins is I’m building the jackal and set up 90 house environments (matterport 3d). They plan to train with IsaacLab, which should allow thousands if episodes a second with parallel envs. In the future, if we plan to do training, we should take advantage of that too"




# Planners

Saad files:
- base_planner
- global_planners
- test

Aaron files that need to be incorporated into the platform:
- arm_planner
- inverse_kinematics 

Basically, whats happening is that aaron wrote some really nice planners and I want to make them generic across all robots. I wrote a AbstractBasePlanner that shold be a super class to all planners and then different types of planners can inherit from it. For now this is the current hierarchy tree for my planners:
AbstractBasePlanner
|-- AbstractGraphPlanner
    |-- A_StarPlanner

Naming could change as I can see how the grid map can be used for RRT planners too. 

test.py simply tests my code.

# Mapping

## Update to Mapping (Jul 17th, 2025):

There are two files:
- map
- generate_map

generate_map contructs a map.Map object that is essentially a 2d boolean mask of the same size as the environment but in cm (Mujoco env. is defined in meters). map.Map offers many helper methods for planning such as changing the size of the user resolution so graph planning does not happen over every grid cell. See global_planners.test() for example. 
