# Mapping

## Rough sketch of what needs to be done
1. Get all geoms and make a 2d occupancy map. Some details:
    - the plane will be the entire 2d map, it shouldnt count as an obstacle. 
        - Confirm with andrew that many (if not all) environments have a geom of type plane as the ground. Update: Andrew has said that robocasa uses a box type for the ground but the size is available explicitly via python objects. Not sure if its available in XML.
    - all geoms except plane and the geoms of the robot will be obstacles. This should take into account the shape of the geoms and basically whatever their collision boxes are.
        - another exceptional case are invisible geoms or whatever geoms that dont participate in collision checking according to mujoco
        - the algorithm should be as efficient as possible. for eg, if there are overlapping geoms then dont run the algorithm twice.
2. Details of the Occupancy map data structure:
    - take inspiration from ROS move base occupancy map?
    - resolution doesnt have to be fixed, can be finer grained nearer obstacles (i forgot what this is called) so that circular geoms can be accurately depicted.
    - Get origin from mujoco

## Simple timeline
1. Assume fixed, very small resolution - 0.01m and make the map based on plane.
    - Get map size from XML -> geom name="floor_room_g0", type="box"
2. Write the algorithm simply covering geom type="box" only.
3. Repeat step 2, making the algorithm faster while covering more types that are in the robocasa environment. One important milestone is being able to take into account meshes
4. ~~Finally we can integrate dynamic resolution mapping if needed.~~


## After completion of a discrete fixed resolution map
1. We want a continuous map that we can work with for sample based planners. The discrete map is what we need for graph based planners but cannot be used for sampling based ones. 
2. To do this all we need is:
    1. a list of obstacles in 2d - boxes and circular ones.
    2. the length and width of the map.
    3. transformation matrix between map and mjc environment.

I believe this should be easy to do since we already have all of that information. See 5th reference for what we will use this for.


## References
1. https://github.com/AtsushiSakai/PythonRobotics
2. https://motion.cs.illinois.edu/RoboticSystems/CoordinateTransformations.html#2D-coordinate-frames
3. https://motion.cs.illinois.edu/RoboticSystems/CoordinateTransformations.html#Rotations-in-2D
4. https://github.com/ai-winter/python_motion_planning/tree/master -> this will be what we will pass our map to. 
5. https://github.com/ai-winter/python_motion_planning/blob/d2daf0db239d4c8673c771494ad41534ca26a76f/src/python_motion_planning/utils/environment/env.py#L83 -> continuous map for sampling based planners such as RRT