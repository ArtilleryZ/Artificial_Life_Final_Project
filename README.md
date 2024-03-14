# Artificial_Life_Final_Project
This is the repository of the final project of Artificial Life

Video links:
https://youtu.be/nWwEX8BtGNs

# Overview
In this project, the goal is to maximize the jumping height of a robot with two legs.  
The genotype is shown as below:  
![Genotype](/img/Genotype.png)

Unlike projects from last year, which use pybullet and pyrosim as their frame, this project uses Mujoco for the physical simulation.  
Without a well-structured frame, it is a little bit tricky to have the code implemented, and the result would probably not meet the quality of projects from last year.  
  
Instead of using a Genetic algorithm, which is specially designed for evolution, or Reinforcement learning, which can use large training time for better results, I am using a 4 hidden-layer neural network to have the algorithm evolved.  
Since it is not specifically designed for evolution, the result will largely depend on tunning and initialization status. Additionally, it will get stuck in the local minima, which will stop the evolution process.

# Sample Result

# How to run it?
There are 4 main scripts in the directory, and they are called:  
`main_script.py`, which is used for the main simulator.  
`run_result.py`, which is used for simulating the Gen0 and last Gen of the result.  
`helper_function.py`, which is for robot parameter initialization.  
`xml_generator.py`, which is for generating the xml file used for Mujoco to run.  

There are 4 folders in the directory, and they are called:  
`data`, used for storing the csv file output  
`xml`, used for storing the xml being used  
`sample`, used for sample and future output for `run_result.py`  
`img`, used for storing figures for this README  
  
There is a block for setting the parameters of the algorithm and performance in the `main_script.py`  
`max_iterations = 200`, control the total iteration numbers  
`pos_lr = 0.005`, control the adjustment steps to position while having the Neural Network running    
`size_lr = 0.01`  controls the adjustment steps to size while having the Neural Network running  
`lr_NN = 0.0001`  learning rate of the Neural Network  
`update_ratio = 0.3`  The portion of the parameters that will be changed during one iteration  

To run the final result in `run_result.py`, please change the name of the csv file according to the latest file in the `data` folder  
It would be something like `data/maxheight_x_0.186_y_0.16_z_0.089.csv`

# Robot Geometry Explanation
As described in the overview section, it has a main body and two legs. The following is an example of what it looks like:  
![Phenotype](/img/Phenotype.png)
Since it is a robot that aims to jump height, the geometry is simplified. The robot can only have its leg in y directions, and it will be adjacent to its previous body.  
For example, leg1 is exactly adjacent to the main body, and leg2 is exactly adjacent to the leg1.  

The main body will be only randomly generated once at the beginning, and all further adjustments are limited to the legs' size and position.  

The main body size has the following boundaries:  
`x_size` = [0.15, 0.25]  
`y_size` = [0.1, 0.12]  
`z_size` = [0.1, 0.12]  

Similarly, the size of the legs has the following boundaries:  
`leg_x_size` = [0.2, 0.6]  
`leg_y_size` = [0.02, 0.15]  
`leg_z_size` = [0.02, 0.15]  

The position of leg1 is dependent on the main body size and the position of leg2 is dependent on the leg1 size  
when generating the random start, it has the logic of:  
temp1x = x_size + leg1_x_size_l  temp1z = z_size + leg1_z_size_l  
leg1_x_pos_l = np.random.uniform(-temp1x, temp1x)  leg1_z_pos_l = np.random.uniform(-temp1z, temp1z)  
temp2x = leg1_x_size_l + leg2_x_size_l  temp2z = leg1_z_size_l + leg2_z_size_l  
leg2_x_pos_l = np.random.uniform(-temp2x,temp2x)  leg2_z_pos_l = np.random.uniform(-temp2z,temp2z)  

Hence  
the lower limit of `leg1_x_pos` is -(0.15+0.2), `leg1_z_pos` is -(0.1+0.02)  
the upper limit of `leg1_x_pos` is  (0.05+0.6), `leg1_z_pos` is  (0.1+0.15)  
the lower limit of `leg2_x_pos` is -2*(0.2),    `leg2_z_pos` is -2*(0.02)  
the upper limit of `leg2_x_pos` is  2*(0.6),    `leg2_z_pos` is  2*(0.15)  

# Robot Simulation Explanation
To reduce the complexity and remain focused on size and position, the actuators are all using a position actuator with a kp of 10.  
Also, the control number, which is the `ctrl`, is set to 3 for all actuators to expel the effect from the jumping force.  
In each round of simulation, it lets the robot operate a time of `800` ticks, which is enough for the robot to jump.  
  
To retrieve the height data, a sensor is attached to the center of the robot. Since it is the z data, it is [2] to the sensordata.  
After each simulation, the max height during the simulation will be recorded, and the outlier value will be removed. It is flagged as -1 so that the evolution process can ignore this change.
The robot state is identified as:  
[x_size, y_size, z_size,  
leg1_x_pos_l, leg1_z_pos_l,  
leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,  
leg2_x_pos_l, leg2_z_pos_l,  
leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,  
leg1_x_pos_r, leg1_z_pos_r,  
leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,  
leg2_x_pos_r, leg2_z_pos_r,  
leg2_x_size_r, leg2_y_size_r, leg2_z_size_r]  
  
Combined with the max height data, it will be stored in a csv file called `rawmaxheight.csv`, with the robot's x, y, and z dimensions included in this name. It will be stored in the `data` directory.


# XML Generator Explanation
There is a file called xml_generator.py, which is used to create the xml file for running Mujoco.  
It combines `generate_environment`, `generate_main_body`, and `generate_legs` as its sub-functions.  
Then `generate_robot_xml` will take parameter inputs and generate a well-formated xml for running Mujoco.  
The xml file will be stored in the folder of `xml` in the root directory.

# Neural Network/Evolution Principle Explanation
To implement the neural network, I use the library of torch and torch.nn, which are standard neural network frames.  
  
The network is a 4 hidden-layer network, with the activation function all using ReLU. The input and output sizes are both 20 according to the robot's parameters. The middle layer's size is 200.  
The parameters that are going to change are only the size and position of the legs, there is no main body involved.  
  
For parameter update, there is a function called `apply_adjustment`, which has 3 main parameters: `pos_lr`, `size_lr`, and `update_ratio`. Due to the change and boundary of position and size are different, they have different step tunning. The update ratio controls the portion of the 20 parameters that will be changed during one generation.  

The loss function is identified as the reciprocal of the height. As the goal is to maximize the height, and neural network will try to minimize the loss function, the reciprocal of the height will meet this requirement.
  
When the new generation doesn't get height improved, the parameters will revert to the when the best height has been achieved by far. It will output if the height is improved or not in the console. The updating threshold is 0.02.  
  
To solve the problem the neural network gets stuck in the local minima and wastes many iterations on not improving, a stuck saver is introduced to this procedure.  
After 10 generations it is not improving, the learning rate of position and size will be 5 times as it should, and have the update_ratio to a higher one.  
But please still remember that it can still get stuck in the local minima due to the inherent feature of neural network.
  
Combined with the max height data, it will be stored in a csv file called `maxheight.csv`, with the robot's x, y, and z dimensions included in this name. It will be stored in the `data` directory.
