#!/bin/bash
python -m planner.examples.pick_and_place_safety_L2 -f scene.xml
sleep 10
python -m planner.examples.pick_and_place_safety_L2 -f scene_level2.xml
sleep 10
python -m planner.examples.pick_and_place_safety_L2 -f scene_level3.xml