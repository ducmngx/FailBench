I am not sure how to translate the controllers section of the Franka Panda example. So bear with me if its messy.


Update (06.29.24):

I will use [Robosuite](http://github.com/ARISE-Initiative/robosuite/tree/master) for writing the controllers. 


Findings:
1. Robosuite's Mobile Base controller converts desired velocity to torques. I am just using Mujoco to take care of converting vel->torque. But the issue with using mujoco is that the kp value has to be very low (fails when > 3). I could directly convert to torque too. 
2. actions (control input) are clipped to be within [input_min, input_max] and then scaled to be within [output_min, output_max].