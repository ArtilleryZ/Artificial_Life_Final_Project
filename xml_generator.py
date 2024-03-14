import numpy as np

def generate_environment():
    #Get the initial environment into xml
    return [
        '    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>',
        '    <geom type="plane" size="5 5 0.1" rgba=".9 0.9 0.9 1"/>'
    ]

def generate_main_body(x_size, y_size, z_size):
    #generate the main body randomly, it will only run once
    
    return [
        '    <body name="main body" pos="0 0 .2">',
        '        <joint type="free"/>',
        f'        <geom type="box" size="{x_size} {y_size} {z_size}" rgba="0 0 1 1" mass="1"/>',
        '        <site name="marker" pos="0 0 0" size="0.05"/>',
        '        <site name="robot_center" pos="0 0 0" size="0.01"/>',
    ], (x_size, y_size, z_size)

def generate_legs(side, main_body_sizes, leg1_x_size, leg1_y_size, leg1_z_size,
                                         leg2_x_size, leg2_y_size, leg2_z_size,
                                         leg1_x_pos , leg1_z_pos,
                                         leg2_x_pos , leg2_z_pos):
    #generate the legs and it will offset by y-direction so that it is adjunct
    #to the main body on exact position
    x_size, y_size, z_size = main_body_sizes
    
   
    #set the exact leg1 location, y is exact and x/z are random
    leg1_y_pos = (y_size + leg1_y_size) if side == 'left' else -(y_size + leg1_y_size)
   
    
    #set the exact leg2 location, y is exact and x/z are random
    leg2_y_pos = leg1_y_size + leg2_y_size   if side == 'left' else -(leg1_y_size + leg2_y_size)
    
    #sort them into xml
    xml_lines = [
        f'        <body name="{side}_leg1" pos="{leg1_x_pos} {leg1_y_pos} {leg1_z_pos}">',
        f'            <joint name="{side}_leg1_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>',
        f'            <geom type="box" size="{leg1_x_size} {leg1_y_size} {leg1_z_size}" rgba="0 .5 0 1" mass="1"/>',
        f'            <body name="{side}_leg2" pos="{leg2_x_pos} {leg2_y_pos} {leg2_z_pos}">',  # Nested within leg1
        f'                <joint name="{side}_leg2_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>',
        f'                <geom type="box" size="{leg2_x_size} {leg2_y_size} {leg2_z_size}" rgba="0 .6 0 1" mass="1"/>',
        '            </body>',
        '        </body>'
    ]
    
    return xml_lines

def generate_robot_xml(filename,
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
                       leg2_x_pos_r , leg2_z_pos_r):
    
    #sort all into xml
    xml_lines = ['<mujoco>', '    <worldbody>']
    xml_lines.extend(generate_environment())
    
    
    main_body_xml, main_body_sizes = generate_main_body(x_size, y_size, z_size)
    xml_lines.extend(main_body_xml)
    

    xml_lines.extend(generate_legs('left', main_body_sizes, leg1_x_size_l, leg1_y_size_l, leg1_z_size_l,
                                                            leg2_x_size_l, leg2_y_size_l, leg2_z_size_l,
                                                            leg1_x_pos_l , leg1_z_pos_l,
                                                            leg2_x_pos_l , leg2_z_pos_l))
    
    xml_lines.extend(generate_legs('right', main_body_sizes, leg1_x_size_r, leg1_y_size_r, leg1_z_size_r,
                                                             leg2_x_size_r, leg2_y_size_r, leg2_z_size_r,
                                                             leg1_x_pos_r , leg1_z_pos_r,
                                                             leg2_x_pos_r , leg2_z_pos_r))
    
    #add sensors
    xml_lines.extend([
        '    </body>',
        '    </worldbody>',
        '    <sensor>',
        '        <framepos objtype="site" objname="marker"/>',
        '        <velocimeter name="robot_velocity" site="robot_center"/>',
        '    </sensor>',
        '    <actuator>'
    ])
    
    #add position actuator
    for side in ['left', 'right']:
        xml_lines.append(f'        <position joint="{side}_leg1_joint" kp="10"/>')
        xml_lines.append(f'        <position joint="{side}_leg2_joint" kp="10"/>')
    
    xml_lines.append('    </actuator>')
    xml_lines.append('</mujoco>')

    with open(filename, 'w') as file:
        file.write('\n'.join(xml_lines))
