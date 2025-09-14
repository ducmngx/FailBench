from typing import List

class ExperimentTrajectoryManager:
    def __init__(self):
        self.trajectories = {} # Store trajectories keyed by scenario name
    
    def store_trajectory(self, scenario_name, phase, trajectory, start_config=None, goal_config=None, goal_pos=None, me_cost=None, safety_cost=None):
        """Plan with and without failure costs, store both."""
        print(f"Storing trajectory for {scenario_name} at phase '{phase}'")

        if scenario_name not in self.trajectories:
            self.trajectories[scenario_name] = {}
            
        # Store both trajectories
        self.trajectories[scenario_name][phase] = {
            # 'phase': phase,
            'trajectory': trajectory,
            # 'start_config': start_config.copy(),
            # 'goal_config': goal_config.copy(),
            'goal_pos': goal_pos,
            "me_cost": me_cost,
            "safety_cost": safety_cost
        }
        
        print(f"Stored trajectory for {scenario_name} at  phase '{phase}'")
    
    def get_trajectory(self, scenario_name, phase):
        """Get stored trajectory for evaluation."""
        if scenario_name not in self.trajectories:
            return None
        
        return self.trajectories[scenario_name][phase]
    
    def report(self):
        """Report stored trajectories."""
        print(f"\nStored Experiment Trajectories:")
        
        # Summary overview
        total_scenarios = len(self.trajectories)
        total_phases = sum(len(phases) for phases in self.trajectories.values())
        print(f"Total scenarios: {total_scenarios}")
        print(f"Total phases: {total_phases}")
        
        # List scenarios and their phases
        for scenario_name, phases in self.trajectories.items():
            phase_names = list(phases.keys())
            print(f"- {scenario_name}: {phase_names}")
        
        print("\nDetailed Trajectories:")
        for scenario_name, phases in self.trajectories.items():
            print(f"\nScenario: {scenario_name}")
            for phase_name, data in phases.items():

                # Check if trajectories exist and get their lengths
                plan = data['trajectory'] if data['trajectory'] else []
                if len(plan) == 0:
                    print(f"No traj saved in {scenario_name}/{phase_name}...")
                    continue
                print(f"  Phase: {phase_name}")
                print(f"    Length of traj: {len(plan)}.")
                for k, val in data.items():
                    print(f"    {k} ({type(val)}): {val}.")
                # # print(f"    Scenario: {scenario_name}")
                # print(f"    Trajectory: {plan}.")
                # print(f"    Data type: {type(plan[0])}")
                # print(f"    Trajectory: {plan}.")
                # print(f"    Data type: {type(plan[0])}")
    
    def save_to_file(self, filename="experiment_trajectories.pkl"):
        """Save all trajectories to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectories, f)
        print(f"Saved trajectories to {filename}")
    
    def load_from_file(self, filename="experiment_trajectories.pkl"):
        """Load trajectories from file."""
        import pickle
        with open(filename, 'rb') as f:
            self.trajectories = pickle.load(f)
        print(f"Loaded trajectories from {filename}")

    def update_from_file(self, filenames:str):
        """Load trajectories from file and adds to the dictionary of trajectories."""
        import pickle
        with open(filenames, 'rb') as f:
            trajectories = pickle.load(f)
        self.trajectories.update(trajectories)
        print(f"Loaded trajectories from {filenames}")



# # Usage example:
# trajectory_manager = ExperimentTrajectoryManager()

# # Plan all scenarios upfront
# '''

# Description -- Phase -- Trajectory

# '''
# scenarios = [
#     ("level01_base", "Pick", [0, 1, 2]),
#     # ("transport", grasp_config, place_config, "transit"), 
#     # ("retreat", place_config, home_config, "transit")
# ]

# for scenario_name, phase, trajectory in scenarios:
#     trajectory_manager.store_trajectory(
#         scenario_name, phase, trajectory
#     )

# # Save for later use
# trajectory_manager.save_to_file("my_experiment_trajectories.pkl")

# # During evaluation:
# # trajectory_manager.load_from_file("my_experiment_trajectories.pkl")
# # safe_trajectory = trajectory_manager.get_trajectory("pick_object3", use_failure_cost=True)
# # baseline_trajectory = trajectory_manager.get_trajectory("pick_object3", use_failure_cost=False)