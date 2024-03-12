import gym
from gym import spaces
import mujoco
import numpy as np
import mujoco_viewer
import time
from mujoco import viewer
import dm_control.mujoco as dm
from stable_baselines3 import PPO
import final_xml_generator  # Ensure this module is properly defined

class RobotJumpEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(RobotJumpEnv, self).__init__()
        # Action space: 18 actions (increase or decrease each of the 9 parameters)
        self.action_space = spaces.Discrete(18)
        # Observation space: the ranges of each parameter
        self.observation_space = spaces.Box(
            low=np.array([0.18, 0.13, 0.08, 0.3, 0.02, 0.02, 0.3, 0.02, 0.02]),
            high=np.array([0.22, 0.17, 0.12, 0.5, 0.15, 0.15, 0.5, 0.15, 0.15]),
            dtype=np.float32)
        self.state = self.observation_space.sample()  # Randomly sample an initial state
        self.filename = "final_robot.xml"

    def step(self, action):
        # Determine parameter to modify based on action
        param_index = action // 2  # Integer division; maps action to parameter index
        direction = 1 if action % 2 == 0 else -1  # Even actions increase, odd actions decrease
        
        # Update parameter within bounds
        increment = direction * 0.01  # Increase or decrease by 0.01
        self.state[param_index] = np.clip(self.state[param_index] + increment, 
                                          self.observation_space.low[param_index], 
                                          self.observation_space.high[param_index])
        
        # Run simulation with updated parameters to get reward (max height)
        max_height = self.simulate_robot()
        reward = max_height
        
        # Define done condition for the episode (modify as needed)
        done = False
        
        return self.state, reward, done, {}

    def reset(self):
        # Reset state to a new random configuration within bounds
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode='human', close=False):
        # Optional: Implement rendering if you want to visualize the environment
        pass

    def simulate_robot(self):
        # Simulate the robot using current state (parameters) and return the max height achieved
        # This should interface with your simulation setup, including updating the XML and running MuJoCo
        final_xml_generator.generate_robot_xml(self.filename, *self.state)
        m = mujoco.MjModel.from_xml_path(self.filename)
        d = mujoco.MjData(m)
        viewer = mujoco_viewer.MujocoViewer(m, d)
        for j in range(len(d.ctrl)):
            d.ctrl[j] = 5
        time.sleep(1)
        height = []
        for i in range(1500):
            
            height.append(d.sensordata[2])
            
            if viewer.is_alive:
                mujoco.mj_step(m,d)
                viewer.render()
            else:
                break
        
        viewer.close()
        return max(height)

env = RobotJumpEnv()
# check_env(env)  # Optional, checks if environment follows Gym interface

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_robot_jump")
