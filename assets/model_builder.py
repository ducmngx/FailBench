# this file is supposed to combine the different xml files in a similar way to robosuite


"""
1- This file will first load the config file: config/jackal_in_kitchen.yaml

2- Depending on the robot, env and robot sensors, it will load up the different xml files from 
    - environments/ 
    - models/robots
    - models/sensors

3- place the sesnors on the robot, based on the config file.
4- and then place the robot in the env, based on the config file.
5- generates an xml file that has the sensors on the robot, the robot in the environment.
"""