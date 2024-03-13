import mujoco
import numpy as np
import mujoco_viewer
import time
from mujoco import viewer
import dm_control.mujoco as dm

from xml_generator import generate_robot_xml

max_iter = 5;

x_size = np.random.uniform(0.18, 0.22)
y_size = np.random.uniform(0.13, 0.17)
z_size = np.random.uniform(0.08, 0.12)

for iter in range(max_iter):
    
    leg1_x_size_l = np.random.uniform(0.3, 0.5)
    leg1_y_size_l = np.random.uniform(0.02, 0.15)
    leg1_z_size_l = np.random.uniform(0.02, 0.15)
    leg2_x_size_l = np.random.uniform(0.3, 0.5)
    leg2_y_size_l = np.random.uniform(0.02, 0.15)
    leg2_z_size_l = np.random.uniform(0.02, 0.15)
    
    leg1_x_size_r = np.random.uniform(0.3, 0.5)
    leg1_y_size_r = np.random.uniform(0.02, 0.15)
    leg1_z_size_r = np.random.uniform(0.02, 0.15)
    leg2_x_size_r = np.random.uniform(0.3, 0.5)
    leg2_y_size_r = np.random.uniform(0.02, 0.15)
    leg2_z_size_r = np.random.uniform(0.02, 0.15)
    
    temp1x = x_size + leg1_x_size_l
    temp1z = z_size + leg1_z_size_l
    leg1_x_pos_l = np.random.uniform(-temp1x, temp1x) # Maintain alignment
    leg1_z_pos_l = np.random.uniform(-temp1z, temp1z) # Maintain alignment
    
    temp2x = leg1_x_size_l + leg2_x_size_l
    temp2z = leg1_z_size_l + leg2_z_size_l
    leg2_x_pos_l = np.random.uniform(-temp2x,temp2x)  # Maintain alignment
    leg2_z_pos_l = np.random.uniform(-temp2z,temp2z)  # Maintain alignment
    
    temp3x = x_size + leg1_x_size_r
    temp3z = z_size + leg1_z_size_r
    leg1_x_pos_r = np.random.uniform(-temp3x, temp3x) # Maintain alignment
    leg1_z_pos_r = np.random.uniform(-temp3z, temp3z) # Maintain alignment
    
    temp4x = leg1_x_size_r + leg2_x_size_r
    temp4z = leg1_z_size_r + leg2_z_size_r
    leg2_x_pos_r = np.random.uniform(-temp4x,temp4x)  # Maintain alignment
    leg2_z_pos_r = np.random.uniform(-temp4z,temp4z)  # Maintain alignment
    
    

    
    filename = "xml/final"
    filename = filename + str(iter+1) + ".xml"
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
    
    
    for i in range(1500):
        
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
    


