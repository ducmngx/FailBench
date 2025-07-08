# Readme
The simple_world.xml loads up the environment and places the jackal. However, the jackal is not correctly placed and falls off the world. 
So we gotta fix that.

## Environments
We currently have only 1 environment that andrew has extracted from robocasa. I stripped away all textures and meshes and its now a scary red colored environment. Purpose of it is to simply go through the generated XML tree and understand how to do mapping, planning, etc. 

For visualization we can either follow Andrew's tutorial on how to generate the environments or we can download the assets for the environments from robocasa's github. However, the latter method is not guaranteed to work as *maybe* the assets themselves are generated. 

## Models 

### Meshes
I've only kept the jackal meshes that im using. The rest can be downloaded from https://github.com/jackal/jackal/tree/noetic-devel/jackal_description/meshes.

Velodyne meshes downloaded from velodyne-description package.

Going forward we can probably organize our robot specific meshes in different folders within the meshes folder or whatever is standard.

## General
Conversion betwen ROS URDF coordinate system -> Mujoco XML:  
Ros URDF xyz = (forward, left, up)  
Mujoco XML pos = (right, forward, up)

ROS URDF rotation = in radians  
Mujoco XML rotation = generally in degrees but you can set `<compiler angle="radian" />`
