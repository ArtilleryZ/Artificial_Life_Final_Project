import torch
import torch.nn as nn
import numpy as np
import mujoco
import mujoco_viewer
import time
from mujoco import viewer
import dm_control.mujoco as dm
from xml_generator import generate_robot_xml

from helper_function import data_initialization
from helper_function import parameter_boundary

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)  # Added fourth hidden layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))  # Passing through the fourth hidden layer
        x = self.output_layer(x)
        return x

def apply_adjustments(params, adjustments, learning_rate):
    new_params = params + learning_rate * adjustments
    new_params = np.clip(new_params, a_min=lower_bounds, a_max=upper_bounds)
    return new_params

def simulate_robot(num_iter, x_size, y_size, z_size, parameters):
    [leg1_x_pos_l, leg1_z_pos_l,
     leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
    
     leg2_x_pos_l, leg2_z_pos_l,
     leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
   
     leg1_x_pos_r, leg1_z_pos_r,
     leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
    
     leg2_x_pos_r, leg2_z_pos_r,
     leg2_x_size_r, leg2_y_size_r, leg2_z_size_r] = parameters
    
    
    filename = "xml/final"
    filename = filename + str(num_iter+1) + ".xml"
    xml = generate_robot_xml(filename,
                             x_size, 
                             y_size, 
                             z_size,
                                
                             leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
                             leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
                             leg1_x_pos_l , leg1_z_pos_l,
                             leg2_x_pos_l , leg2_z_pos_l,
                                 
                             leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
                             leg2_x_size_r, leg2_y_size_r, leg2_z_size_r,
                             leg1_x_pos_r , leg1_z_pos_r,
                             leg2_x_pos_r , leg2_z_pos_r)
    
    m = mujoco.MjModel.from_xml_path(filename)
    
    d = mujoco.MjData(m)
    
    
    height = []
    
    viewer = mujoco_viewer.MujocoViewer(m,d)
    
    for j in range(len(d.ctrl)):
        d.ctrl[j] = 5
    time.sleep(1)
    
    
    for i in range(1000):
        
        height.append(d.sensordata[2])
        
        if viewer.is_alive:
            mujoco.mj_step(m,d)
            viewer.render()
        else:
            break
    
    viewer.close()
    
    robot_geometry = [x_size, y_size, z_size,
                      leg1_x_pos_l, leg1_z_pos_l,
                      leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
                      
                      leg2_x_pos_l, leg2_z_pos_l,
                      leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
                     
                      leg1_x_pos_r, leg1_z_pos_r,
                      leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
                      
                      leg2_x_pos_r, leg2_z_pos_r,
                      leg2_x_size_r, leg2_y_size_r, leg2_z_size_r]
    
    max_height = [max(height)]
    
    combined_data = [robot_geometry + max_height]
    
    
    with open('maxheight.csv', 'ab') as f:
        np.savetxt(f, combined_data, delimiter=',')
        
    return max_height

# Neural network initialization and training process...
# Please ensure correct implementation of data_initialization and parameter_boundary functions.

learning_rate = 0.05  # Adjust this based on your observations on optimization performance

nn_model = SimpleNeuralNetwork(input_dim=20, hidden_dim=100, output_dim=20)

robot_init = data_initialization()
x_size, y_size, z_size = robot_init[:3]
parameters = robot_init[3:]

lower_bounds, upper_bounds = parameter_boundary()

optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.005)

max_iterations = 10

for iteration in range(max_iterations):
    parameters_tensor = torch.tensor(parameters, dtype=torch.float32).unsqueeze(0)
    adjustments = nn_model(parameters_tensor).squeeze().detach().numpy()
    parameters = apply_adjustments(parameters, adjustments, learning_rate)

    max_height = simulate_robot(iteration, x_size, y_size, z_size, parameters)
    if max_height is not None:  # Ensure max_height is not None
        loss = -torch.tensor(max_height, dtype=torch.float, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Iteration {iteration+1}, Loss: {loss.item()}, Max Height: {max_height}")
    else:
        print("Error: simulate_robot returned None for max_height.")
