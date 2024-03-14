import mujoco
import numpy as np
import mujoco_viewer
import time
from mujoco import viewer
import dm_control.mujoco as dm
from xml_generator import generate_robot_xml

data = np.loadtxt('data/maxheight_x_0.182_y_0.103_z_0.11.csv', delimiter=',')

height = data[:, 23]
idx_min_height = 0
min_height = height[idx_min_height]
idx_max_height = -1
max_height = height[idx_max_height]

parameters_min_height = data[idx_min_height, :-1]
parameters_max_height = data[idx_max_height, :-1]

def run_simulation(parameters, filename_suffix):
    x_size, y_size, z_size = parameters[:3]
    leg1_x_pos_l, leg1_z_pos_l, leg1_x_size_l, leg1_y_size_l, leg1_z_size_l = parameters[3:8]
    leg2_x_pos_l, leg2_z_pos_l, leg2_x_size_l, leg2_y_size_l, leg2_z_size_l = parameters[8:13]
    leg1_x_pos_r, leg1_z_pos_r, leg1_x_size_r, leg1_y_size_r, leg1_z_size_r = parameters[13:18]
    leg2_x_pos_r, leg2_z_pos_r, leg2_x_size_r, leg2_y_size_r, leg2_z_size_r = parameters[18:23]
    
    filename = "sample/maxheight_{filename_suffix}.xml"
    generate_robot_xml(filename,
                       x_size, y_size, z_size,
                       leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
                       leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
                       leg1_x_pos_l, leg1_z_pos_l,
                       leg2_x_pos_l, leg2_z_pos_l,
                       leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
                       leg2_x_size_r, leg2_y_size_r, leg2_z_size_r,
                       leg1_x_pos_r, leg1_z_pos_r,
                       leg2_x_pos_r, leg2_z_pos_r)

    m = mujoco.MjModel.from_xml_path(filename)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    
    for j in range(len(d.ctrl)):
        d.ctrl[j] = 5
    time.sleep(1)

    for i in range(2000):
        if viewer.is_alive:
            mujoco.mj_step(m, d)
            viewer.render()
        else:
            break
    
    viewer.close()


run_simulation(parameters_min_height, "min_height")
print(f"Generation {0}, Height: {min_height}")

run_simulation(parameters_max_height, "max_height")
print(f"Generation {200}, Height: {max_height}")
