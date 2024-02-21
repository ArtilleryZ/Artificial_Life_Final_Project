    <actuator>
        <velocity name="left_motor1" joint="left1" kv="10"/>
        <velocity name="left_motor2" joint="left2" kv="10"/>
        <velocity name="right_motor1" joint="right1" kv="10"/>
        <velocity name="right_motor2" joint="right2" kv="10"/>
    </actuator>
    
    <actuator>
        <motor name="left_motor1" joint="left1" gear="1"/>
        <motor name="left_motor2" joint="left2" gear="1"/>
        <motor name="right_motor1" joint="right1" gear="1"/>
        <motor name="right_motor2" joint="right2" gear="1"/>
    </actuator>
    
    
    <mujoco>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 0.1" rgba=".9 0.9 0.9 1"/>
        
        <body name="main body" pos="0 0 .11">
            <joint type="free"/>
            <geom type="box" size=".2 .15 .1" rgba="0 0 1 1" mass="1"/>
            <site name="marker" pos = "0 0 0" size="0.05" />
            <site name="robot_center" pos="0 0 0" size="0.01"/>
            
            <body name="left_leg1" pos="0 0.3 0.0">
                <joint name="left1" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom type="box" size=".2 .02 .01" rgba="0 .5 0 1" mass="1"/>
                
                <body name="left_leg2" pos="0 0.3 0.0">
                    <joint name="left2" type="hinge" axis="0 1 0" pos="0 0 0"/>
                    <geom type="box" size=".2 .02 .01" rgba="0 .5 0 1" mass="1"/>
                </body>
                
            </body>
            
            <body name="right_leg1" pos="0 -0.3 0.0">
                <joint name="right1" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom type="box" size=".2 .02 .01" rgba="0 .5 0 1" mass="1"/>
                
                <body name="right_leg2" pos="0 -0.3 0.0">
                    <joint name="right2" type="hinge" axis="0 1 0" pos="0 0 0"/>
                    <geom type="box" size=".2 .02 .01" rgba="0 .5 0 1" mass="1"/>
                </body>
            </body>
            
            
        </body>
    </worldbody>
    
    <sensor>
		<framepos objtype="site" objname="marker"/>
        <velocimeter name="robot_velocity" site="robot_center"/>
	</sensor>
	
    <actuator>
        <position name="left_motor1" joint="left1" kp="10"/>
        <position name="left_motor2" joint="left2" kp="10"/>
        <position name="right_motor1" joint="right1" kp="10"/>
        <position name="right_motor2" joint="right2" kp="10"/>
    </actuator>
</mujoco>















import numpy as np

def generate_environment():
    return [
        '    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>',
        '    <geom type="plane" size="5 5 0.1" rgba=".9 0.9 0.9 1"/>'
    ]

def generate_main_body():
    x_size = np.random.uniform(0.18, 0.22)
    y_size = np.random.uniform(0.13, 0.17)
    z_size = np.random.uniform(0.08, 0.12)
    return [
        '    <body name="main body" pos="0 0 .11">',
        '        <joint type="free"/>',
        f'        <geom type="box" size="{x_size} {y_size} {z_size}" rgba="0 0 1 1" mass="1"/>',
        '        <site name="marker" pos="0 0 0" size="0.05"/>',
        '        <site name="robot_center" pos="0 0 0" size="0.01"/>',
    ], (x_size, y_size, z_size)

def generate_legs(side, main_body_sizes):
    leg1_x_size = np.random.uniform(0.15, 0.25)
    leg1_y_size = np.random.uniform(0.15, 0.25)
    leg1_z_size = np.random.uniform(0.05, 0.15)
    
    x_size, y_size, z_size = main_body_sizes
    leg1_x_pos = np.random.uniform(-x_size / 2 - leg1_x_size/2, x_size / 2 + leg1_x_size/2)
    leg1_z_pos = np.random.uniform(-z_size / 2 - leg1_z_size/2, z_size / 2 + leg1_z_size/2)
    leg1_y_pos = (y_size + leg1_y_size) if side == 'left' else -(y_size + leg1_y_size)


    # Second leg's size and adjusted position for y-direction adjacency
    leg2_x_size = np.random.uniform(0.15, 0.25)
    leg2_y_size = np.random.uniform(0.15, 0.25)
    leg2_z_size = np.random.uniform(0.05, 0.15)
    leg2_y_pos = leg1_y_size + leg2_y_size   if side == 'left' else -(leg1_y_size + leg2_y_size)

    xml_lines = [
        f'        <body name="{side}_leg1" pos="{leg1_x_pos} {leg1_y_pos} {leg1_z_pos}">',
        f'            <joint name="{side}_leg1_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>',
        f'            <geom type="box" size="{leg1_x_size} {leg1_y_size} {leg1_z_size}" rgba="0 .5 0 1" mass="1"/>',
        f'            <body name="{side}_leg2" pos="0 {leg2_y_pos} 0">',  # Nested within leg1
        f'                <joint name="{side}_leg2_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>',
        f'                <geom type="box" size="{leg2_x_size} {leg2_y_size} {leg2_z_size}" rgba="0 .6 0 1" mass="1"/>',
        '            </body>',  # Close leg2
        '        </body>'  # Close leg1
    ]
    
    return xml_lines

def generate_robot_xml(filename):
    xml_lines = ['<mujoco>', '    <worldbody>']
    xml_lines.extend(generate_environment())
    main_body_xml, main_body_sizes = generate_main_body()
    xml_lines.extend(main_body_xml)
    
    for side in ['left', 'right']:
        xml_lines.extend(generate_legs(side, main_body_sizes))
    
    xml_lines.extend([
        '    </body>',  # Correctly close the main body tag
        '    </worldbody>',
        '    <actuator>'
    ])
    
    for side in ['left', 'right']:
        xml_lines.append(f'        <position joint="{side}_leg1_joint" kp="100"/>')
        xml_lines.append(f'        <position joint="{side}_leg2_joint" kp="100"/>')
    
    xml_lines.extend([
        '    </actuator>',
        '</mujoco>'
    ])

    with open(filename, 'w') as file:
        file.write('\n'.join(xml_lines))

if __name__ == "__main__":
    generate_robot_xml('final.xml')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#backup for final_xml_generator:
    import numpy as np

    def generate_environment():
        return [
            '    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>',
            '    <geom type="plane" size="5 5 0.1" rgba=".9 0.9 0.9 1"/>'
        ]

    def generate_main_body():
        x_size = np.random.uniform(0.18, 0.22)
        y_size = np.random.uniform(0.13, 0.17)
        z_size = np.random.uniform(0.08, 0.12)
        return [
            '    <body name="main body" pos="0 0 .11">',
            '        <joint type="free"/>',
            f'        <geom type="box" size="{x_size} {y_size} {z_size}" rgba="0 0 1 1" mass="1"/>',
            '        <site name="marker" pos="0 0 0" size="0.05"/>',
            '        <site name="robot_center" pos="0 0 0" size="0.01"/>',
        ], (x_size, y_size, z_size)

    def generate_legs(side, main_body_sizes):
        x_size, y_size, z_size = main_body_sizes
        
        # Sizes for leg1
        leg1_x_size = np.random.uniform(0.15, 0.25)
        leg1_y_size = np.random.uniform(0.15, 0.25)
        leg1_z_size = np.random.uniform(0.05, 0.15)
        
        # Corrected position calculation for leg1
        temp1x = x_size+leg1_x_size
        temp1z = z_size+leg1_z_size
        leg1_x_pos = np.random.uniform(-temp1x, temp1x)
        leg1_z_pos = np.random.uniform(-temp1z, temp1z)
        leg1_y_pos = (y_size + leg1_y_size) if side == 'left' else -(y_size + leg1_y_size)
        
        # Sizes for leg2
        leg2_x_size = np.random.uniform(0.15, 0.25)
        leg2_y_size = np.random.uniform(0.15, 0.25)
        leg2_z_size = np.random.uniform(0.05, 0.15)
        
        # Explicit position calculation for leg2
        temp2x = leg1_x_pos+leg2_x_size
        temp2z = leg1_z_pos+leg2_z_size
        leg2_x_pos = np.random.uniform(-temp2x,temp2x)  # Maintain alignment in the x direction with leg1
        leg2_z_pos = np.random.uniform(-temp2z,temp2z)  # Maintain alignment in the z direction with leg1
        leg2_y_pos = leg1_y_size + leg2_y_size   if side == 'left' else -(leg1_y_size + leg2_y_size)

        xml_lines = [
            f'        <body name="{side}_leg1" pos="{leg1_x_pos} {leg1_y_pos} {leg1_z_pos}">',
            f'            <joint name="{side}_leg1_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>',
            f'            <geom type="box" size="{leg1_x_size} {leg1_y_size} {leg1_z_size}" rgba="0 .5 0 1" mass="1"/>',
            f'            <body name="{side}_leg2" pos="{leg2_x_pos} {leg2_y_pos} {leg2_z_pos}">',  # Nested within leg1, with explicit x, z positioning
            f'                <joint name="{side}_leg2_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>',
            f'                <geom type="box" size="{leg2_x_size} {leg2_y_size} {leg2_z_size}" rgba="0 .6 0 1" mass="1"/>',
            '            </body>',  # Close leg2
            '        </body>'  # Close leg1
        ]
        
        return xml_lines

    def generate_robot_xml(filename):
        xml_lines = ['<mujoco>', '    <worldbody>']
        xml_lines.extend(generate_environment())
        main_body_xml, main_body_sizes = generate_main_body()
        xml_lines.extend(main_body_xml)
        
        for side in ['left', 'right']:
            xml_lines.extend(generate_legs(side, main_body_sizes))
        
        xml_lines.extend([
            '    </body>',  # Correctly close the main body tag
            '    </worldbody>',
            '    <actuator>'
        ])
        
        for side in ['left', 'right']:
            xml_lines.append(f'        <position joint="{side}_leg1_joint" kp="10"/>')
            xml_lines.append(f'        <position joint="{side}_leg2_joint" kp="10"/>')
        
        xml_lines.extend([
            '    </actuator>',
            '</mujoco>'
        ])

        with open(filename, 'w') as file:
            file.write('\n'.join(xml_lines))

    if __name__ == "__main__":
        generate_robot_xml('final.xml')