import torch
import torch.nn as nn
import numpy as np
import mujoco
import mujoco_viewer
import copy
import time
from mujoco import viewer
import dm_control.mujoco as dm
from xml_generator import generate_robot_xml

from helper_function import data_initialization
from helper_function import parameter_boundary


max_iterations = 200

pos_lr = 0.02
size_lr = 0.05
lr_NN = 0.005
update_ratio = 0.3


stuck_pos_lr = pos_lr * 5
stuck_size_lr = size_lr * 5
stuck_update_ratio = 0.75
stuck_count = 0
stuck_threshold = 10

input_dim = 20
hidden_dim = 200
output_dim = 20

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


def apply_adjustment(parameter, adjustment, pos_lr, size_lr, update_ratio=0.5):
    num_params = len(parameter)
    num_updates = int(update_ratio * num_params)
    update_idx = np.random.choice(num_params, num_updates, replace=False)
    
    pos_idx = [0, 1, 5, 6, 10, 11, 15, 16]
    
    new = parameter.copy()
    
    for i in update_idx:
        if i in pos_idx:
            lr = pos_lr
        else:
            lr = size_lr
        
        new[i] = parameter[i] + lr * adjustment[i]

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
        d.ctrl[j] = 3
    time.sleep(1)
    
    for i in range(800):
        
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
    
    if max_height[0] >= 4:
        return -1
    
    combined_data = [robot_geometry + max_height]
    
    raw_filename = "data/rawmaxheight"
    
    raw_filename = raw_filename + "_x_" + str(round(x_size,3)) + "_y_" + str(round(y_size,3)) + "_z_" + str(round(z_size,3)) + ".csv"
    
    
    with open(raw_filename, 'ab') as f:
        np.savetxt(f, combined_data, delimiter=',')
        
    return max_height


nn_model = four_layer_NN(input_dim, hidden_dim, output_dim)

robot_init = data_initialization()
x_size, y_size, z_size = robot_init[:3]
parameter = robot_init[3:]

best_parameter = copy.deepcopy(parameter)
best_height = -np.inf 

lower_bounds, upper_bounds = parameter_boundary()

optimizer = torch.optim.Adam(nn_model.parameters(), lr_NN)


for itr in range(max_iterations):
    if stuck_count >= stuck_threshold:
        temp_pos_lr = stuck_pos_lr
        temp_size_lr = stuck_size_lr
        temp_update_ratio = stuck_update_ratio
        print("Stuck saver activated.")
    else:
        temp_pos_lr = pos_lr
        temp_size_lr = size_lr
        temp_update_ratio = update_ratio
    
    
    parameter_tensor = torch.tensor(parameter, dtype=torch.float32).unsqueeze(0)
    adjustment = nn_model(parameter_tensor).squeeze().detach().numpy()
    new = apply_adjustment(parameter, adjustment, temp_pos_lr, temp_size_lr, temp_update_ratio)
    

    max_height = simulate_robot(itr, x_size, y_size, z_size, new)
    
    
    if max_height[0] == -1:
        print("Error: simulator broke at this generation.")
        continue
    
    if max_height[0] > best_height:
        print(f"Generation {itr}, Height Improved: {max_height}")
        best_height = max_height[0]
        best_parameter = copy.deepcopy(new)
        stuck_count = 0
        
        loss = 1 / torch.tensor([max_height[0]], dtype=torch.float, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        stuck_count += 1
        print(f"Generation {itr}, No Improvement. Reverting")
    
    parameter = best_parameter
        
    final_data = [best_parameter + best_height]
    
    main_filename = "data/maxheight"

    main_filename = main_filename + "_x_" + str(round(x_size,3)) + "_y_" + str(round(y_size,3)) + "_z_" + str(round(z_size,3)) + ".csv"
    
    with open(main_filename, 'ab') as f:
        np.savetxt(f, final_data, delimiter=',')
        
    
