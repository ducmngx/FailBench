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
4. Finally we can integrate dynamic resolution mapping if needed.

## References
1. https://github.com/AtsushiSakai/PythonRobotics
