import mujoco_py
import os
import time

def main():
    # Path to the humanoid.xml model file
    mujoco_dir = os.path.expanduser('~/.mujoco/mujoco210')
    model_path = os.path.join(mujoco_dir, 'model', 'humanoid.xml')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model and create a simulation
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)

    # Create a viewer window
    viewer = mujoco_py.MjViewer(sim)

    # Run the simulation for 1000 steps
    for _ in range(1000):
        sim.step()
        viewer.render()
        time.sleep(0.01)  # Slow down for visualization

if __name__ == '__main__':
    main()
