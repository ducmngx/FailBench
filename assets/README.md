# Assets Organization
Based on the franka_emika_panda directory and robosuite the assets directory is made up as follows:
assets
|--environments (contains environments including fixtures, cups, etc.)
    |-- (.xml files such as dumped_kitchen_noassets.xml)
|--models (contain robots + sensors)
    |-- robots 
        |-- franka_emika_panda
            |-- assets (contain mesh or obj files)
            |-- (.xml files that are different constructs of panda)
            |-- (other files such as pictures and a README)
        |-- (other robots)
    |-- sensors
        |-- vlp16
            |-- assets (mesh files)
            |-- (.xml files that are different constructs of vlp16)
        |-- (other sensors)
|-- config.yaml 
|-- model_builder.py (builds an environment based on the config.yaml file)

### Todo
1. since the files are now completely restructured, the code for model_builder.py needs to be updated. Similarly, the dependencies within the xml files need to be updated.

## Andrew's tutorial on generating environment
For visualization we have to follow Andrew's tutorial on how to generate the environments, we cannot simply download the assets  method does not work as assets are actually generated from yaml files. See [robocasa/model_zoo](https://github.com/robocasa/robocasa/tree/main/robocasa/scripts/model_zoo).


### Helpful tips
Conversion betwen ROS URDF coordinate system -> Mujoco XML:  
Ros URDF xyz = (forward, left, up)  
Mujoco XML pos = (right, forward, up)

ROS URDF rotation = in radians  
Mujoco XML rotation = generally in degrees but you can set `<compiler angle="radian" />`
