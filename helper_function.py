import numpy as np


def data_initialization():
    x_size = np.random.uniform(0.18, 0.22)
    y_size = np.random.uniform(0.13, 0.17)
    z_size = np.random.uniform(0.08, 0.12)

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
    
    
    state = [x_size, y_size, z_size,
            leg1_x_pos_l, leg1_z_pos_l,
            leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
            
            leg2_x_pos_l, leg2_z_pos_l,
            leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
           
            leg1_x_pos_r, leg1_z_pos_r,
            leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
            
            leg2_x_pos_r, leg2_z_pos_r,
            leg2_x_size_r, leg2_y_size_r, leg2_z_size_r]
    
    return state

def parameter_boundary():
    x_size_lower, x_size_upper = 0.18, 0.22
    y_size_lower, y_size_upper = 0.13, 0.17
    z_size_lower, z_size_upper = 0.08, 0.12
    
    leg_size_x_lower, leg_size_x_upper = 0.3, 0.5
    leg_size_y_lower, leg_size_y_upper = 0.02, 0.15
    leg_size_z_lower, leg_size_z_upper = 0.02, 0.15
    
    lower = [-(x_size_lower + leg_size_x_lower), -(z_size_lower + leg_size_z_lower),
             leg_size_x_lower, leg_size_y_lower, leg_size_z_lower,
             -2*leg_size_x_lower, -2*leg_size_z_lower,
             leg_size_x_lower, leg_size_y_lower, leg_size_z_lower,
             -(x_size_lower + leg_size_x_lower), -(z_size_lower + leg_size_z_lower),
             leg_size_x_lower, leg_size_y_lower, leg_size_z_lower,
             -2*leg_size_x_lower, -2*leg_size_z_lower,
             leg_size_x_lower, leg_size_y_lower, leg_size_z_lower]
    
    
    upper = [(x_size_upper + leg_size_x_upper), (z_size_upper + leg_size_z_upper),
             leg_size_x_upper, leg_size_y_upper, leg_size_z_upper,
             2*leg_size_x_upper, 2*leg_size_z_upper,
             leg_size_x_upper, leg_size_y_upper, leg_size_z_upper,
             (x_size_upper + leg_size_x_upper), (z_size_upper + leg_size_z_upper),
             leg_size_x_upper, leg_size_y_upper, leg_size_z_upper,
             2*leg_size_x_upper, 2*leg_size_z_upper,
             leg_size_x_upper, leg_size_y_upper, leg_size_z_upper]
    
    
    return lower, upper