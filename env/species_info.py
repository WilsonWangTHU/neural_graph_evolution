'''
    @brief:
        document the constraints of parameters for different creatures
'''
import init_path
import numpy as np

CREATURE_ROOT_INFO = {
    'fish': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.01,
        'b_size': 0.08,
        'c_size': 0.04,
        'joint_range': 60
    },
    'walker': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.07,
        'b_size': 0.3,
        'c_size': 233,
        'joint_range': 60
    },
    'hopper': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.0653,
        'b_size': 0.1025,
        'c_size': 0.03,
        'joint_range': 60
    },
    'cheetah': {
        'geom_type': -1,
        'u': -1,
        'v': -1,
        'axis_x': -233,
        'axis_y': -233,
        'a_size': 0.046,
        'b_size': 0.5,
        'c_size': 0.03,
        'joint_range': 60
    }
}

CREATURE_HARD_CONSTRAINT = {
    'fish': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.002, 0.03),
        'b_size': (0.002, 0.03),
        'c_size': (0.002, 0.03),
        'joint_range': (30, 120)
    },
    'walker': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.03, 0.07),
        'b_size': (0.1, 0.3),
        'c_size': (30, 40), # c_size server as the value for gear
        'joint_range': (30, 90)  
    },
    'hopper': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.03, 0.07),
        'b_size': (0.010, 0.035),
        'c_size': (0.002, 0.03),
        'joint_range': (30, 120)
    },
    'cheetah': {
        'u': (1e-9, 2*np.pi),
        'v': (1e-9,   np.pi),
        'axis_x': (-1, 1),
        'axis_y': (-1, 1),
        'a_size': (0.002, 0.03),
        'b_size': (0.03, 0.15),
        'c_size': (30, 120),
        'joint_range': (30, 120)
    }
}

CREATURE_ORIGINAL_ATTR = {
    'fish': [
        # tail1
        {
            'u': 3 * np.pi / 2,
            'v': np.pi / 2,
            'axis_x': -0.99,
            'axis_y': 1e-16,
            'a_size': 0.001,
            'b_size': 0.008,
            'c_size': 0.016,
            'joint_range': 30,
            'geom_type': 0
        },
        # tail2
        {
            'u': np.pi / 2,
            'v': np.pi / 2,
            'axis_x': 0.99,
            'axis_y': 1e-16, 
            'a_size': 0.001,
            'b_size': 0.018,
            'c_size': 0.035,
            'joint_range': 30,
            'geom_type': 0   # the original attribute is defined using stiffness 
        },
        # left fin
        {
            'u': np.pi,
            'v': np.pi / 2,
            'axis_x': 1e-16,
            'axis_y': 0.99,
            'a_size': 0.02, 
            'b_size': 0.015, 
            'c_size': 0.001,
            'joint_range': 30,
            'geom_type': 0 # the original attribute is defined using tendon
        },
        # right fin
        {
            'u': 0,
            'v': np.pi / 2,
            'axis_x': 1e-16,
            'axis_y': -0.99,
            'a_size': 0.02,
            'b_size': 0.015,
            'c_size': 0.001,
            'joint_range': 30,
            'geom_type': 0
        }
    ],
    'walker': [
        # left thigh 
        # NOTE: left and right thigh should be symmetric
        {
            'u': 4 * np.pi / 9,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.05, 
            'b_size': 0.225,
            'c_size': 40, 
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.04, 
            'b_size': 0.25,
            'c_size': 40, 
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 0,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.05, 
            'b_size': 0.08,
            'c_size': 40, 
            'joint_range': 30,
            'geom_type': 0
        }
    ],
    'hopper': [
        {
            'u': np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.065, 
            'b_size': 0.075,
            'c_size': 30, 
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 0,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.04, 
            'b_size': 0.165,
            'c_size': 30, 
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.03, 
            'b_size': 0.16,
            'c_size': 30, 
            'joint_range': 30,
            'geom_type': 0
        },
        {
            'u': 0,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.04, 
            'b_size': 0.09,
            'c_size': 30, 
            'joint_range': 30,
            'geom_type': 0
        }
    ],
    'cheetah': [
        {# back thigh
            'u': -4 * np.pi / 9,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.046, 
            'b_size': 0.145,
            'c_size': 120, 
            'joint_range': 30,
            'geom_type': 0
        },
        {# back shin
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.046, 
            'b_size': 0.15,
            'c_size': 90, 
            'joint_range': 30,
            'geom_type': 0
        },
        {# back foot
            'u': -4 * np.pi / 9,
            'v': np.pi,
            'axis_x': -1,
            'axis_y': -1, 
            'a_size': 0.046, 
            'b_size': 0.15,
            'c_size': 60, 
            'joint_range': 30,
            'geom_type': 0
        },
        {# front thigh
            'u': 2 * np.pi,
            'v': np.pi,
            'axis_x': 1,
            'axis_y': -1, 
            'a_size': 0.046, 
            'b_size': 0.145,
            'c_size': 90, 
            'joint_range': 30,
            'geom_type': 0
        },
        {# front shin
            'u': 0,
            'v': np.pi,
            'axis_x': 1,
            'axis_y': -1, 
            'a_size': 0.046, 
            'b_size': 0.106,
            'c_size': 60, 
            'joint_range': 30,
            'geom_type': 0
        },
        {# front foot
            'u': 7 * 2 * np.pi / 9,
            'v': np.pi,
            'axis_x': 1,
            'axis_y': -1, 
            'a_size': 0.046, 
            'b_size': 0.07,
            'c_size': 30, 
            'joint_range': 30,
            'geom_type': 0
        }
    ]
}

# local imports for generating the original creatures
from env.hierarchy_model import Node
from env.hierarchy_model import Tree


def get_original_fish():
    ''' create the structure that defines the original fish
    '''
    r_node = Node(0, CREATURE_ROOT_INFO['fish'])
    tail1 = Node(0, CREATURE_ORIGINAL_ATTR['fish'][0])
    tail2 = Node(0, CREATURE_ORIGINAL_ATTR['fish'][1])
    fin_l = Node(0, CREATURE_ORIGINAL_ATTR['fish'][2])
    fin_r = Node(0, CREATURE_ORIGINAL_ATTR['fish'][3])

    tail1.add_child(tail2)
    r_node.add_child(tail1)
    r_node.add_child(fin_l)
    r_node.add_child(fin_r)

    return r_node

def get_original_walker():
    ''' 
    '''
    r_node = Node(0, CREATURE_ROOT_INFO['walker'])
    thigh_r = Node(0, CREATURE_ORIGINAL_ATTR['walker'][0])
    leg_r = Node(0, CREATURE_ORIGINAL_ATTR['walker'][1])
    foot_r = Node(0, CREATURE_ORIGINAL_ATTR['walker'][2])
    thigh_l = Node(0, CREATURE_ORIGINAL_ATTR['walker'][0])
    leg_l = Node(0, CREATURE_ORIGINAL_ATTR['walker'][1])
    foot_l = Node(0, CREATURE_ORIGINAL_ATTR['walker'][2])

    leg_r.add_child(foot_r)
    thigh_r.add_child(leg_r)
    r_node.add_child(thigh_r)

    leg_l.add_child(foot_l)
    thigh_l.add_child(leg_l)
    r_node.add_child(thigh_l)

    return r_node

def get_original_cheetah():
    '''
    '''
    r_node = Node(0, CREATURE_ROOT_INFO['cheetah'])
    b_thigh = Node(0, CREATURE_ORIGINAL_ATTR['cheetah'][0])
    b_shin = Node(0, CREATURE_ORIGINAL_ATTR['cheetah'][1])
    b_foot = Node(0, CREATURE_ORIGINAL_ATTR['cheetah'][2])

    f_thigh = Node(0, CREATURE_ORIGINAL_ATTR['cheetah'][3])
    f_shin = Node(0, CREATURE_ORIGINAL_ATTR['cheetah'][4])
    f_foot = Node(0, CREATURE_ORIGINAL_ATTR['cheetah'][5])

    f_shin.add_child(f_foot)
    f_thigh.add_child(f_shin)
    b_shin.add_child(b_foot)
    b_thigh.add_child(b_shin)
    r_node.add_child(b_thigh)
    r_node.add_child(f_thigh)

    return r_node

def get_original_hopper():
    '''
    '''
    r_node = Node(0, CREATURE_ROOT_INFO['hopper'])
    pelvis = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][0])
    thigh = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][1])
    calf = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][2])
    foot = Node(0, CREATURE_ORIGINAL_ATTR['hopper'][3])

    calf.add_child(foot)
    thigh.add_child(calf)
    pelvis.add_child(thigh)
    r_node.add_child(pelvis)

    return r_node

