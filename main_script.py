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

"""
Main parameter settings:

Max iteration
learning rate for position
learning rate for size
learning rate for Neural Network frame
Update ratio of the parameter updating

Please notice:
    if you have an initial height larger than 1.5, it might be too high and no much room left for height improvement
    Just Ctrl+C to stop it and run again.
"""
max_iterations = 200
pos_lr = 0.005
size_lr = 0.01
lr_NN = 0.0001
update_ratio = 0.3




#Here is the stuck helper, which will activated if the height isn't improved in 10 generations
stuck_pos_lr = pos_lr * 8
stuck_size_lr = size_lr * 8
stuck_update_ratio = 0.8
stuck_count = 0
stuck_threshold = 10

#layer's dimention
input_dim = 20
hidden_dim = 200
output_dim = 20

class four_layer_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        #Simple neural network structure
        super(four_layer_NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #All using ReLU
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
    
    #sort out position index
    pos_idx = [0, 1, 5, 6, 10, 11, 15, 16]
    
    new = parameter.copy()
    
    for i in update_idx:
        if i in pos_idx:
            lr = pos_lr
        else:
            lr = size_lr
        
        new[i] = parameter[i] + lr * adjustment[i]

    #make sure it is in the boundary
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
    
    #Generate the xml file for simulation
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
    
    #all actuators have the same value on movement
    for j in range(len(d.ctrl)):
        d.ctrl[j] = 5
        
    #delay for simulation
    time.sleep(1)
    
    for i in range(800):
        
        #the sensor is [x,y,z], hence [2] is the z value, which is the height
        height.append(d.sensordata[2])
        
        if viewer.is_alive:
            mujoco.mj_step(m,d)
            viewer.render()
        else:
            break
    
    viewer.close()
    
    #sort all geometry-related information
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
    
    #Physically, it can't jump higher than 4. Therefore, 4 means it hits a NaN
    #Then place a flag of -1 for it.
    if max_height[0] >= 4:
        return -1
    
    #combine the data for output
    combined_data = [robot_geometry + max_height]
    
    #raw data means including all unimproved value"
    raw_filename = "data/rawmaxheight"
    
    raw_filename = raw_filename + "_x_" + str(round(x_size,3)) + "_y_" + str(round(y_size,3)) + "_z_" + str(round(z_size,3)) + ".csv"
    
    
    with open(raw_filename, 'ab') as f:
        np.savetxt(f, combined_data, delimiter=',')
        
    return max_height


#set up the nn model
nn_model = four_layer_NN(input_dim, hidden_dim, output_dim)

robot_init = data_initialization()
#seperate out the x, y, and z value as it is only initialized once
x_size, y_size, z_size = robot_init[:3]
parameter = robot_init[3:]

#set up a best_parameter for comparison
best_parameter = copy.deepcopy(parameter)
best_height = -np.inf 

lower_bounds, upper_bounds = parameter_boundary()

#start the optimizer
optimizer = torch.optim.Adam(nn_model.parameters(), lr_NN)


for itr in range(max_iterations):
    #stuck helper
    if stuck_count >= stuck_threshold:
        temp_pos_lr = stuck_pos_lr
        temp_size_lr = stuck_size_lr
        temp_update_ratio = stuck_update_ratio
        # print("Stuck saver activated.")
    else:
        temp_pos_lr = pos_lr
        temp_size_lr = size_lr
        temp_update_ratio = update_ratio
    
    #standard NN procedure
    parameter_tensor = torch.tensor(parameter, dtype=torch.float32).unsqueeze(0)
    adjustment = nn_model(parameter_tensor).squeeze().detach().numpy()
    new = apply_adjustment(parameter, adjustment, temp_pos_lr, temp_size_lr, temp_update_ratio)
    
    #start simulation
    max_height = simulate_robot(itr, x_size, y_size, z_size, new)
    
    #Indicate an NaN is encountered, and skip this loop
    if max_height == -1:
        print("Error: simulator broke at this generation.")
        continue
    
    #0.02 is a threshold so that the model will be more flexible
    if max_height[0]-0.02 > best_height:
        print(f"Generation {itr}, Height Improved: {max_height}")
        best_height = max_height[0]
        best_parameter = copy.deepcopy(new)
        stuck_count = 0
        
        #the loss function is the height. Since I want maximum height, and NN is to lower the loss function
        #I use the reciprocal of it so that it will try to maximize the hiehgt.
        loss = 1 / torch.tensor([max_height[0]], dtype=torch.float, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        stuck_count += 1
        print(f"Generation {itr}, No Improvement. Height: {max_height}")
    
    parameter = best_parameter
    
    #sort all data, only improved data is included
    final_data = [[x_size] + [y_size] + [z_size] + best_parameter.tolist() + [best_height]]
    
    main_filename = "data/maxheight"

    main_filename = main_filename + "_x_" + str(round(x_size,3)) + "_y_" + str(round(y_size,3)) + "_z_" + str(round(z_size,3)) + ".csv"
    
    with open(main_filename, 'ab') as f:
        np.savetxt(f, final_data, delimiter=',')
        
    
