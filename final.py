import mujoco
import numpy as np
import mujoco_viewer
import time
from mujoco import viewer
import dm_control.mujoco as dm

import final_xml_generator

max_iter = 5;



for iter in range(max_iter):
    
    x_size = np.random.uniform(0.18, 0.22)
    y_size = np.random.uniform(0.13, 0.17)
    z_size = np.random.uniform(0.08, 0.12)


    leg1_x_size = np.random.uniform(0.3, 0.5)
    leg1_y_size = np.random.uniform(0.02, 0.15)
    leg1_z_size = np.random.uniform(0.02, 0.15)
    leg2_x_size = np.random.uniform(0.3, 0.5)
    leg2_y_size = np.random.uniform(0.02, 0.15)
    leg2_z_size = np.random.uniform(0.02, 0.15)
    

    
    filename = "xml/final"
    filename = filename + str(iter+1) + ".xml"
    xml = final_xml_generator.generate_robot_xml(filename,
                              x_size,
                              y_size, 
                              z_size,
                              leg1_x_size,
                              leg1_y_size,
                              leg1_z_size,
                              leg2_x_size,
                              leg2_y_size,
                              leg2_z_size)
    
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
                      leg1_x_size, leg1_y_size, leg1_z_size,
                      leg2_x_size, leg2_y_size, leg2_z_size]
    
    max_height = [max(height)]
    
    combined_data = [robot_geometry + max_height]
    
    
    with open('maxheight.csv', 'ab') as f:
        np.savetxt(f, combined_data, delimiter=',')
    


