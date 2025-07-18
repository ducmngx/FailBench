**Documentation: Get Robocasa/View Kitchen Environment**

**Robocasa (Robosuite installs during the process):**

[https://github.com/robocasa/robocasa](https://github.com/robocasa/robocasa)

Notes:

* **Follow the steps in robocasa, with these in mind:**  
* Use the conda environment, it is required  
* Install robosuite v1.5, not v1.5.1  
* Mujoco should be 3.2.6, numpy should be 1.23.3  
* The download for assets is \~5 GB, so make sure you have space  
* When you first run the test, it will say there are macro errors, and a virtual environment is needed, it is fine (I ignore them)

Windows Notes:

* Drop mujoco.dll into the robosuite.utils folder, used for running mujoco (this **must** be done in the robosuite folder in the virtual environment from the file explorer, as the one outside of the virtual environment does not work. The path looks something like this: C:\\Users\\amars\\anaconda3\\envs\\robocasa\\Lib\\site-packages\\robosuite\\utils)

Mac Notes:

* Not sure, but I think you need to do the same for the .dylib file from mujoco (I don’t use mac, so I can’t really help there, sorry)

**Kitchen Environment**

* Download scripts from GitHub, and replace each one with its corresponding one from robosuite/robocasa (This should be done from the folders outside of the virtual environment one, and I would recommend be done through file explorer)  
  * At the top of each script, I put the path of where the current file is for the new script to replace  
* Run **test-simulate.py** in VSCode, using the python from your Virtual environment instead of your computer’s python, choosing the **layout id** and **style id**  
  * The path looks something like this: C:/Users/amars/anaconda3/envs/robocasa/python.exe  
* It should dump an xml file to your computer, allowing you to access the xml file locally

###Acknowledgement
This ReadME was created using Andrew's google doc: https://docs.google.com/document/d/1lTeJCkHhdj3Ju5xPeo2-O3NBbzo1-NJtam14spqxyGA/edit?tab=t.0
