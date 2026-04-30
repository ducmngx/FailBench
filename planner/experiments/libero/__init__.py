"""LIBERO replay workflow — sibling to the IK+RRT trajectory generator.

Loads HDF5 demos from LIBERO's robosuite-based dataset, replays them in MuJoCo,
injects FailBench failure modes mid-trajectory, and captures contacts. Reuses
ContactExtractor / OffscreenRenderer / SimStateCheckpoint as-is.
"""
