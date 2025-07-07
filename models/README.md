## General

Conversion betwen ROS URDF coordinate system -> Mujoco XML:  
Ros URDF xyz = (forward, left, up)  
Mujoco XML pos = (right, forward, up)

ROS URDF rotation = in radians  
Mujoco XML rotation = generally in degrees but you can set `<compiler angle="radian" />`

## Meshes
I've only kept the jackal meshes that im using. The rest can be downloaded from https://github.com/jackal/jackal/tree/noetic-devel/jackal_description/meshes.

Velodyne meshes downloaded from velodyne-description package.

Going forward we can probably organize our robot specific meshes in different folders within the meshes folder.