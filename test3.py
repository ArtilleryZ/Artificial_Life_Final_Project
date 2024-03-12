import numpy as np
import mujoco
import mujoco_viewer
import final_xml_generator

# Initialize simulation parameters
parameters = np.array([0.18, 0.13, 0.08, 0.3, 0.02, 0.02, 0.3, 0.02, 0.02])
filename = "final_robot.xml"

# Function to modify parameters based on action
def modify_parameters(action, parameters):
    param_index = action // 2
    direction = 1 if action % 2 == 0 else -1
    increment = direction * 0.01
    parameters[param_index] = np.clip(parameters[param_index] + increment, 0.02, 0.5)
    return parameters

# Function to run a simulation step and return new state and reward
def simulate_step(parameters):
    final_xml_generator.generate_robot_xml(filename, *parameters)
    m = mujoco.MjModel.from_xml_path(filename)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    height = []
    for _ in range(1500):
        height.append(d.sensordata[2])
        mujoco.mj_step(m, d)
    viewer.close()
    max_height = max(height)
    return parameters, max_height

# Example of running a simulation step with an action
action = 2  # Example action to decrease the third parameter
parameters = modify_parameters(action, parameters)
new_state, reward = simulate_step(parameters)
print(f"New state: {new_state}, Reward: {reward}")
