# GenAISim - Saad_Dev Branch


## Code Organization
Currently its a bit messy. ill fix it over the week. The jackal xml file can be found in model/jackal.xml. I want to base the code organization on a mujoco example I found online - https://github.com/justagist/mujoco_panda/tree/master. We can adjust as we develop this repo.

## Helpful things to know
- mujoco can load urdf files with minimal changes. However its better to convert a URDF file to a mujoco xml file via `./compile /path/to/model.urdf /path/to/mujuco_model.xml`. 
    - References:
        - https://mujoco.readthedocs.io/en/stable/modeling.html#modeling
        - https://github.com/robotlearning123/dual_ur5_husky_mujoco/tree/dual_ur5_husky_mujoco
- Parallel Training:
    - "On a side note, what we are doing here w Hopkins is I’m building the jackal and set up 90 house environments (matterport 3d). They plan to train with IsaacLab, which should allow thousands if episodes a second with parallel envs. In the future, if we plan to do training, we should take advantage of that too"







