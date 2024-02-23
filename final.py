import mujoco
import numpy as np
import mujoco_viewer
import time
from mujoco import viewer
import dm_control.mujoco as dm

# Load the model
m = mujoco.MjModel.from_xml_path('final.xml')
d = mujoco.MjData(m)



# mujoco.viewer.launch(m,d)

# with mujoco.viewer.launch_passive(m,d) as viewer:
#     for j in range(len(d.ctrl)):
#         d.ctrl[j] = 1
#     time.sleep(1)
#     for i in range(10000):
        
#         # if i%50 == 0:
        
#         dm.mj_step(m,d)
        
#         viewer.sync()
#         time.sleep(0.01)

height = []

viewer = mujoco_viewer.MujocoViewer(m,d)

for j in range(len(d.ctrl)):
    d.ctrl[j] = 5
time.sleep(1)
for i in range(2000):
    
    height.append(d.sensordata[2])
    # print(pos)
    if viewer.is_alive:
        mujoco.mj_step(m,d)
        viewer.render()
    else:
        break

viewer.close()