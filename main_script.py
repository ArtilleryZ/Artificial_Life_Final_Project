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



class four_layer_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super(four_layer_NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x)) 
        x = self.output_layer(x)
        return x

def apply_adjustment(parameter, adjustment, learning_rate):
    new = parameter + learning_rate * adjustment
    new = np.clip(new, a_min=lower_bounds, a_max=upper_bounds)
    return new

def simulate_robot(num_iter, x_size, y_size, z_size, parameter):
    [leg1_x_pos_l, leg1_z_pos_l,
     leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
    
     leg2_x_pos_l, leg2_z_pos_l,
     leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
   
     leg1_x_pos_r, leg1_z_pos_r,
     leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
    
     leg2_x_pos_r, leg2_z_pos_r,
     leg2_x_size_r, leg2_y_size_r, leg2_z_size_r] = parameter
    
    
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
    
    output_filename = "data/maxheight"
    
    output_filename = output_filename + "_x_" + str(round(x_size,3)) + "_y_" + str(round(y_size,3)) + "_z_" + str(round(z_size,3)) + ".csv"
    
    
    with open(output_filename, 'ab') as f:
        np.savetxt(f, combined_data, delimiter=',')
        
    return max_height


max_iterations = 500
learning_rate = 0.05
input_dim = 20
hidden_dim = 100
output_dim = 20
lr_NN = 0.005

nn_model = four_layer_NN(input_dim, hidden_dim, output_dim)

robot_init = data_initialization()
x_size, y_size, z_size = robot_init[:3]
parameter = robot_init[3:]

lower_bounds, upper_bounds = parameter_boundary()

optimizer = torch.optim.Adam(nn_model.parameter(), lr_NN)


for itr in range(max_iterations):
    parameter_tensor = torch.tensor(parameter, dtype=torch.float32).unsqueeze(0)
    adjustment = nn_model(parameter_tensor).squeeze().detach().numpy()
    parameter = apply_adjustment(parameter, adjustment, learning_rate)

    max_height = simulate_robot(itr, x_size, y_size, z_size, parameter)
    if max_height is not None:
        loss = -torch.tensor(max_height, dtype=torch.float, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Generation {itr+1}, Loss: {loss.item()}, Max Height: {max_height}")
    else:
        print("Error: simulator broke at this generation.")
