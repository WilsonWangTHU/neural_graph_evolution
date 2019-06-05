'''
    function to generate a mujoco model xml file
'''
from __future__ import division
import init_path
import pdb
import lxml.etree as etree
import numpy as np
import random
import copy
import os

# local import
from util import model_gen_util
from env import species_info
# from env import hierarchy_model

# library to find the intersection between the ellipsoid and the line
from sympy import Symbol
from sympy.solvers import solve

# fixed ratio for the geometry
ELLIPSOID_X_RATIO = 1
ELLIPSOID_Y_RATIO = 8
ELLIPSOID_Z_RATIO = 4
CYLINDER_R_RATIO = 1
CYLINDER_H_RATIO = 6

RAD_TO_DEG = 57.29577951308232
BASE_PATH = init_path.get_abs_base_dir()

def save_generated_xml(root, filename):
    '''
    '''

    if filename is not None:
        tree = etree.ElementTree(root)
        file_path = os.path.join(init_path.get_base_dir(), 
            'env/assets/gen',
            filename + '.xml'
        )
        tree.write(file_path, 
            pretty_print=True, xml_declaration=True, 
            encoding='utf-8'
        )    
    return None

def fish_xml_generator(adj_matrix, node_attr, options=None, filename=None):
    '''
        input:
            adj_matrix: numpy array of size N x N
                        the matrix describes the tree structure of the model
                        assuming the root of the tree (torso) is node 0
                        binary encoding to specify the connections
                        1: x   direction hinge
                        2: y   direction hinge
                        4: z   direction hinge
                        3: xy  direction hinge
                        5: xz  direction hinge
                        6: yz  direction hinge
                        7: xyz direction hinge
            node_attr: list of size N
                       containing 7-D params to specify
                       (1d: type of geom
                        2d: size of the ellipsoid
                            along x and y axis
                        2d: u and v
                            for sampling the stitching point on the parent's body
                        2d: axis_x and axis_y
                            for getting the x-axis
                        )
            options: an optional dictionary of mujoco modelling settings
                     1. support water density
                     2. whether to enable gravity
            filename: if given filename, save the generated xml to file

        output:
            1. xml strings: an xml strings that defines a mujoco model
            2. (optional), saved xml file

        NOTE:
            the input adj_matrix should be a tree structure whose root is at node 0
            and needs to have the following characteristic:
                - the tree structure doesn't contain loop
                - the tree structure doesn't contain separate nodes that is not connected
    '''
    def add_mujoco_header(root):
        ''' generate preliminary dm_control mujoco environement
        '''
        # some include files
        incl_1 = etree.Element('include', file='./common/visual.xml')
        incl_2 = etree.Element('include', file='./common/materials.xml')
        root.append(incl_1)
        root.append(incl_2)
        # adding fish assets
        asset = etree.Element('asset')
        texture = etree.SubElement(asset, 'texture', name='skybox',
                                   type='skybox', builtin='gradient',
                                   rgb1=".4 .6 .8", rgb2="0 0 0",
                                   width="800", height="800", mark="random",
                                   markrgb="1 1 1")
        root.append(asset)
        # add compiler settings
        compiler = etree.Element('compiler', eulerseq='xyz')
        root.append(compiler)
        return root

    def add_mujoco_options(root, options):
        ''' add default and options
            QUESTION: a lot of the options of the model is environment specific
                      should we create separate function for loading different environment
                      based on the options
        '''
        try:
            density = str(options['density'])
        except:
            density = '5000'
        try:
            timestep = str(options['timestep'])
        except:
            timestep = '0.004'
        option = etree.Element('option', timestep=timestep, density=density)

        flag = etree.Element('flag', gravity='disable', constraint='disable')
        option.append(flag)
        root.append(option)

        # some defaults for fish
        default_sec = etree.Element('default')
        general = etree.Element('general', ctrllimited='true')
        default_sec.append(general)
        # joints and geom defaults for fish
        fish_default = etree.fromstring('<default class="fish"></default>')
        fish_default_joint = etree.Element('joint', type='hinge',
                                           limited='false',
                                           range='-60 60', damping='2e-5',
                                           solreflimit='.1 1',
                                           solimplimit='0 .8 .1')
        fish_default_geom = etree.Element('geom', material='self')
        fish_default.append(fish_default_joint)
        fish_default.append(fish_default_geom)
        default_sec.append(fish_default)

        root.append(default_sec)
        return root

    def add_worldbody(root, adj_matrix, node_attr_list):
        ''' parse the adj_matrix to get the tree structure
        '''
        def worldbody_preliminary(worldbody):
            ''' setting up cameras
            '''
            camera1 = etree.Element('camera', name='tracking_top', pos='-.1 .2 .2',
                                    xyaxes='-2-1 0 -0 -.5 1', mode='trackcom')
            camera2 = etree.Element('camera', name='tracking_x', pos='-.3 0 .2',
                                    xyaxes='0 -1 0 0.342 0 0.940', mode='trackcom')
            camera3 = etree.Element('camera', name='tracking_y', pos='0 -.3 .2',
                                    xyaxes='1 0 0 0 0.342 0.940', mode='trackcom')
            camera4 = etree.Element('camera', name='fixed_top', pos='0 0 5.5',
                                    fovy='10')
            worldbody.append(camera1)
            worldbody.append(camera2)
            worldbody.append(camera3)
            worldbody.append(camera4)
            geom1 = etree.Element('geom', name='ground', type='plane',
                                  size='.5 .5 .1', material='grid')
            geom2 = etree.Element('geom', name='target', type='sphere',
                                  pos='0 .4 .1', size='.04', material='target')
            worldbody.append(geom1)
            worldbody.append(geom2)
            return worldbody

        def torso_preliminary(torso):
            ''' copying from the dm_control fish default settings
            '''
            light = etree.Element('light', name="light", diffuse=".6 .6 .6", pos="0 0 0.5", dir="0 0 -1", specular=".3 .3 .3", mode="track")
            torso.append(light)
            joint = etree.Element('joint', name="root", type="free", damping="0", limited="false")
            # torso.append(joint)
            site = etree.Element('site', name="torso", size=".01", rgba="0 0 0 0")
            torso.append(site)
            eye_geom = etree.Element('geom', name="eye", type="ellipsoid", pos="0 .055 .015", size=".008 .012 .008", euler="-10 0 0", material="eye", mass="0")
            torso.append(eye_geom)
            eye_cam  = etree.Element('camera', name="eye", pos="0 .06 .02", xyaxes="1 0 0 0 0 1")
            torso.append(eye_cam)
            mouth1_geom = etree.Element('geom', name="mouth", type="capsule", fromto="0 .079 0 0 .07 0", size=".005", material="effector", mass="0")
            torso.append(mouth1_geom)
            mouth2_geom = etree.Element('geom', name="lower_mouth", type="capsule", fromto="0 .079 -.004 0 .07 -.003", size=".0045", material="effector", mass="0")
            torso.append(mouth2_geom)
            torso_geom = etree.Element('geom', name="torso", type="ellipsoid", size=".01 .08 .04", mass="0")
            torso.append(torso_geom)
            torso_massive_geom = etree.Element('geom', name="torso_massive", type="box", size=".0075 .06 .03", group="4")
            torso.append(torso_massive_geom)

            torso_y = etree.fromstring('<joint name="rooty" type="slide" axis="0 1 0" pos="0 .1 0"/>')
            torso_x = etree.fromstring('<joint name="rootx" type="slide" axis="1 0 0" pos="0 .1 0"/>')
            torso_z = etree.fromstring('<joint name="rootz" type="hinge" axis="0 0 1" pos="0 .1 0" range="-180 180" limited="false" stiffness="0" armature="0"/>')
            torso.append(torso_x)
            torso.append(torso_y)
            torso.append(torso_z)
            return torso 

        def rotation_matrix(axis, angle):
            ''' rotation matrix generated by angle around axis
            '''
            if not (angle >= 0 and angle <= np.pi):
                pdb.set_trace()
            cos = np.cos(angle)
            sin = np.sin(angle)

            if axis == 'x':
                R = np.array([[1,   0,    0],
                              [0, cos, -sin],
                              [0, sin,  cos]])
            elif axis == 'y':
                R = np.array([[ cos, 0, sin],
                              [   0, 1,   0],
                              [-sin, 0, cos]])
            elif axis == 'z':
                R = np.array([[cos, -sin, 0],
                              [sin,  cos, 0],
                              [  0,    0, 1]])
            else:
                raise RuntimeError('given axis not available')
            return R

        def euler_rotation(angle1, angle2, angle3):
            ''' return the rotation matrix as described by euler angle
            angle1: rotate around z axis
            angle2: rotate around y' axis
            angle3: rotate around z'' axis
            '''
            R = np.zeros((3, 3))
            R_z = rotation_matrix('z', angle1)
            R_y = rotation_matrix('y', angle2)
            R = np.matmul(R_z, R_y)
            R_z = rotation_matrix('z', angle3)
            R = np.matmul(R, R_z)
            return R


        def ellipsoid_line_intersection(a, b, c, dx, dy, dz):
            '''
                input: 1. a, b, c: params to specify an ellipsoid (centered at origin)
                       2. dx, dy, dz: direction vector to denote a line at origin
                output: 
                        1. the point coordinate where the line intersects the ellipsoid
                           that is in the direction of the line vector
                NOTE: 1. there is guarantee that there will be intersection between the line
                         and the ellipsoid
                      2. it seems that the solver will always return the neg result first
                         [negative result, positive result]
                         Thus we need to further determine which one is closer to the direction
                         vector and use that as the joint pivot point
            '''
            t = Symbol('t')
            # plug in the line equation p(t) = dx * t + dy * t + dz * t 
            # to the ellipsoid function
            ellipsoid = (dx * t)**2 / a**2 + (dy * t)**2 / b**2 + (dz * t)**2 / c**2 - 1
            res = solve(ellipsoid, t)
            one_res = res[1].evalf()
            return float(one_res) * np.array([dx, dy, dz]) 

        

        def homogeneous_inverse(H):
            ''' given a homogeneous transformation, find its inverse
            '''
            R = H[0:3, 0:3]
            d = H[0:3, 3]
            
            invH = np.zeros((4,4))
            invH[0:3, 0:3] = R.T 
            invH[0:3, 3] = - np.matmul(R.T, d)
            invH[3, 3] = 1
            return invH

        def homogeneous_transform(R, d):
            ''' return the homogeneous transformation given the rotation matrix
            and translation vector
            '''
            H = np.zeros((4, 4))
            H[0:3, 0:3] = R
            H[0:3, 3] = d
            H[3, 3] = 1
            return H

        def homogeneous_representation(p):
            ''' return the homogeneous representation of point p of dim 3
            '''
            P = np.hstack((p, np.array(1)))
            return P

        def ellipsoid_normal_vec(a, b, c, p):
            ''' return the unit normal vector of an ellipsoid at origin with param [a, b, c]
            at point p
            '''
            # verify point is on the ellipsoid
            x, y, z = p.tolist()
            assert np.isclose(1, x**2/a**2 + y**2/b**2 + z**2/c**2, atol=1e-6)
            # equivalent to partial derivate of the ellipsoid function
            vec_x = 2 * x / a**2
            vec_y = 2 * y / b**2
            vec_z = 2 * z / c**2
            vec = np.array([vec_x, vec_y, vec_z])
            return vec / np.linalg.norm(vec)

        def angle_between(vec1, vec2):
            ''' return the value in radians and in degrees
            '''
            theta = np.arccos( np.matmul(vec1.T, vec2) / (1e-8 + np.linalg.norm(vec1) * np.linalg.norm(vec2)) )
            return (theta, theta * RAD_TO_DEG)

        def ellipsoid_boundary_pt(a, b, c):
            ''' given [a, b, c] which specifies an ellipsoid at origin
            return the 6 boundary points' coordinate
            '''
            x1 = np.array([a, 0, 0])
            x2 = np.array([-a, 0, 0])
            y1 = np.array([0, b, 0])
            y2 = np.array([0, -b, 0])
            z1 = np.array([0, 0, c])
            z2 = np.array([0, 0, -c])
            return [x1, x2, y1, y2, z1, z2]


        ################ START PARSING THE WORLDBODY ################
        worldbody = etree.Element('worldbody')
        worldbody = worldbody_preliminary(worldbody)
        
        # parse the matrix
        N, _ = adj_matrix.shape

        body_dict = {}
        info_dict = {} # log the needed information given a node
        
        # the root of the model is always fixed
        body_root = etree.Element('body', name='torso', pos='0 0 .1', childclass='fish')
        body_root = torso_preliminary(body_root)
        root_info = {}

        root_info['a_size'] = 0.01
        root_info['b_size'] = 0.08
        root_info['c_size'] = 0.04
        root_info['abs_trans'] = homogeneous_transform(np.eye(3), np.zeros(3))
        root_info['rel_trans'] = homogeneous_transform(np.eye(3), np.zeros(3))

        info_dict[0] = root_info
        body_dict[0] = body_root

        # initilize the parent list to go throught the entire matrix
        parent_list = [0]
        while len(parent_list) != 0:
            parent_node = parent_list.pop(0)

            parent_row = np.copy(adj_matrix[parent_node])
            for i in range(parent_node+1): parent_row[i] = 0
            child_list = np.where(parent_row)[0].tolist()

            while True:
                
                try: child_node = child_list.pop(0)
                except: break

                # parent-child relationship 
                # print('P-C relationship:', parent_node, child_node)
                node_attr = node_attr_list[child_node]
                node_name = 'node-%d'%(child_node)

                # this is parent's ellipsoid information
                parent_info = info_dict[parent_node]
                a_parent = parent_info['a_size']
                b_parent = parent_info['b_size']
                c_parent = parent_info['c_size']

                # randomly sample a point on ellipsoid
                u = node_attr['u']
                v = node_attr['v']
                x, y, z = model_gen_util.vectorize_ellipsoid(a_parent, b_parent, c_parent,
                                                             u, v)

                # use the normal vector as the child y-axis
                normal_vector = model_gen_util.ellipsoid_normal_vec(a_parent, b_parent, c_parent, 
                                                                    np.array([x, y, z]))
                y_axis_vec = normal_vector
                # according to the y-axis, sample an x-axis,
                # the two vectors of x and y axis dot product need to be zero, fix x, y -> get z 
                axis_x = node_attr['axis_x']
                axis_y = node_attr['axis_y']
                axis_z = (normal_vector[0] * axis_x + normal_vector[1] * axis_y) / (-normal_vector[2] + 1e-8)
                x_axis_vec = np.array([axis_x, axis_y, axis_z])
                x_axis_vec = x_axis_vec / (np.linalg.norm(x_axis_vec) + 1e-8)

                # use cross-product (x-axis cross-product y-axis) (according to right-hand rule) 
                # find the z-axis 
                z_axis_vec = np.cross(x_axis_vec, y_axis_vec)

                # WARN: this is not euler angle even by x-y-z rotation
                # calculate the xyz euler angle, project onto the original frame vector and find the angle
                # this part is not used because of the usage of xyaxes when dealing with frame orientation in mujoco
                # theta_x = angle_between(x_axis_vec, np.array([1, 0, 0]))[0]
                # theta_y = angle_between(y_axis_vec, np.array([0, 1, 0]))[0]
                # theta_z = angle_between(z_axis_vec, np.array([0, 0, 1]))[0]
                # euler_angle = '%d %d %d' % (int(theta_x), int(theta_y), int(theta_z))
                
                a_child = node_attr['a_size']
                b_child = node_attr['b_size']
                c_child = node_attr['c_size']
                # the translation vector: moving from center to the randomly sampled point
                # and move along the z-direction with length c_child
                d_trans = np.array([x, y, z]) + normal_vector * b_child

                # compute the translational and rotational matrix
                child_info = {}
                translation_vec = d_trans
                ''' this part won't be used because of the way we set up rotation coordinates
                also, the transformation matrices are not needed at this stage
                '''
                # R_x = rotation_matrix('x', theta_x)  
                # R_y = rotation_matrix('y', theta_y)
                # R_z = rotation_matrix('z', theta_z)
                # try: R = np.matmul( np.matmul(R_z, R_y), R_x )
                # except: pdb.set_trace()
                # 
                # child_info['rel_trans'] = homogeneous_transform(R, translation_vec)
                # child_info['abs_trans'] = np.matmul( info_dict[parent_node]['abs_trans'], child_info['rel_trans'] )


                # store attributes that defines the child's geom
                child_info['a_size'] = a_child
                child_info['b_size'] = b_child
                child_info['c_size'] = c_child
                
                # body translation
                dx, dy, dz = translation_vec.tolist() 
                body_pos = '%f %f %f' % (dx, dy, dz)
                # joint_pos = np.matmul(child_info['rel_trans'], homogeneous_representation(np.array([x, y, z])))[0:3]
                # joint_x, joint_y, joint_z = joint_pos.tolist()
                joint_pos = '%f %f %f' % (0, -b_child, 0)
                x_axis_x, x_axis_y, x_axis_z = x_axis_vec.tolist()
                y_axis_x, y_axis_y, y_axis_z = y_axis_vec.tolist()
                
                xyaxes = '%f %f %f %f %f %f' % (x_axis_x, x_axis_y, x_axis_z, y_axis_x, y_axis_y, y_axis_z)

                # now create the body
                body_child = etree.Element('body', name=node_name, pos=body_pos, xyaxes=xyaxes)
                # add geom
                geom_type = node_attr['geom_type']
                if geom_type == 0:   # ellipsoid
                    ellipsoid_size = '%f %f %f' % (a_child, b_child, c_child)
                    geom = etree.Element('geom', name=node_name, type='ellipsoid', size=ellipsoid_size)
                elif geom_type == 1: # cylinder
                    cylinder_size = '%f %f' % (geom_size, geom_size * CYLINDER_H_RATIO / 2)
                    geom = etree.Element('geom', name=node_name, type='cylinder', size=cylinder_size)
                else:
                    from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
                    raise RuntimeError('geom type not supported')
                body_child.append(geom)
                
                # add joints
                joint_type = adj_matrix[parent_node, child_node]
                joint_axis = [int(item) for item in list(bin(joint_type)[2:])]
                joint_axis = [0 for i in range(3-len(joint_axis))] + joint_axis
                joint_axis = list(reversed(joint_axis))
                joint_range = node_attr['joint_range']
                if joint_axis[0] == 1:
                    x_joint = etree.fromstring("<joint name='%d-%d_x' axis='1 0 0' pos='%s' range='-%d %d'/>" % (parent_node, child_node, joint_pos, joint_range, joint_range))
                    body_child.append(x_joint)
                if joint_axis[1] == 1:
                    y_joint = etree.fromstring("<joint name='%d-%d_y' axis='0 1 0' pos='%s' range='-%d %d'/>" % (parent_node, child_node, joint_pos, joint_range, joint_range))
                    body_child.append(y_joint)
                if joint_axis[2] == 1:
                    z_joint = etree.fromstring("<joint name='%d-%d_z' axis='0 0 1' pos='%s' range='-%d %d'/>" % (parent_node, child_node, joint_pos, joint_range, joint_range))
                    body_child.append(z_joint)

                site = etree.Element('geom', type='sphere', pos=joint_pos, size='0.003', material='target')
                body_child.append(site)
                
                body_dict[parent_node].append(body_child)
                body_dict[child_node] = body_child # register child's body struct in case it has child
                info_dict[child_node] = child_info
                # child becomes the parent for further examination
                parent_list.append(child_node)

        worldbody.append(body_dict[0])
        root.append(worldbody)
        return root

    def add_mujoco_actuator(root, adj_matrix):
        ''' essentially, all the joints are actuators.
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        actuator = etree.Element('actuator')
        for edge in dfs_order:
            edge_x = '%s_x' % edge
            edge_y = '%s_y' % edge
            edge_z = '%s_z' % edge
            edges = [edge_x, edge_y, edge_z]
            positions = [etree.Element('position', name=item, joint=item, ctrlrange='-1 1', kp='5e-4') 
                         for item in edges]
            for item in positions: actuator.append(item)
        root.append(actuator)
        return root

    def add_mujoco_sensor(root):
        '''
        '''
        sensor = etree.Element('sensor')
        velocimeter = etree.Element('velocimeter', name='velocimeter', site='torso')
        gyro = etree.Element('gyro', name='gyro', site='torso')
        sensor.append(velocimeter)
        sensor.append(gyro)
        root.append(sensor)
        return root

    ######################### START OF THE FUNCTION #########################
    root = etree.Element("mujoco", model='fish')
    root = add_mujoco_header(root)
    root = add_mujoco_options(root, options)
    root = add_worldbody(root, adj_matrix, node_attr)
    root = add_mujoco_actuator(root, adj_matrix)
    root = add_mujoco_sensor(root)

    save_generated_xml(root, filename)
    
    return root

def walker_xml_generator(adj_matrix, node_attr, options=None, filename=None):
    '''
        the generator of the walker model
        NOTE:
            the majority of the codes resemble that of fish_xml_generator()
            the main differences are in mujoco_header, mujoco_options, and 
            it doesn't have the velocimeter
    '''
    def add_mujoco_header(root):
        incl_1 = etree.Element('include', file='./common/visual.xml')
        incl_2 = etree.Element('include', file='./common/skybox.xml')
        incl_3 = etree.Element('include', file='./common/materials.xml')
        root.append(incl_1)
        root.append(incl_2)
        root.append(incl_3)
        return root

    def add_mujoco_options(root, options):
        ''' @brief:
                1. different from the fish environment, it has several different 
                   options 
                        a) option
                        b) statistic
                        c) default
                2. instead of using cylinder, we are using the ellipsoid for any 
                   body parts
        '''
        option = etree.Element('option', timestep='0.0025')
        statistic = etree.Element('statistic', extent='2', center='0 0 1')
        root.append(option)
        root.append(statistic)

        default_sec = etree.Element('default')

        # other defaults
        motor = etree.Element('motor', ctrlrange='-1 1', ctrllimited='true')
        site = etree.Element('site', size='0.01')
        
        # default geom and joint
        default = etree.fromstring('<default class="walker"></default>')
        default_geom = etree.Element('geom', material='self', type='capsule', 
            contype='1', conaffinity='0', friction='.7 .1 .1', condim='1',
            solimp='.99 .99 .003', solref='.015 1'
        )
        default_joint = etree.Element('joint', axis='0 -1 0',
            damping='.1', armature='0.01', limited='true',
            solimplimit='0 .99 .01'
        )
        default.append(default_geom)
        default.append(default_joint)

        default_sec.append(motor)
        default_sec.append(site)
        default_sec.append(default)

        root.append(default_sec)

        return root

    def add_worldbody(root, adj_matrix, node_attr_list):
        ''' parse the adj_matrix to get the tree structure
        '''
        def worldbody_preliminary(worldbody):
            ''' setting up cameras
            '''
            geom1 = etree.Element('geom', name='floor', type='plane',
                                  conaffinity='1', pos='248 0 0',
                                  size='250 .8 .2', material='grid',
                                  zaxis='0 0 1')
            worldbody.append(geom1)

            return worldbody

        def torso_preliminary(torso):
            ''' copying from the dm_control fish default settings
            '''
            light = etree.Element('light', name="light", mode='trackcom') 
            torso.append(light)

            camera1 = etree.Element('camera', name='side', pos='0 -2 0.7',
                                    euler='60 0 0', mode='trackcom')
            camera2 = etree.Element('camera', name='back', pos='-5 0 0.5',
                                    xyaxes='0 -1 0 1 0 3', mode='trackcom')
            torso.append(camera1)
            torso.append(camera2)
    
            torso_size = '0.07 0.3'        
            torso_geom = etree.Element('geom', name="torso", size=torso_size)
            torso.append(torso_geom)

            # free_joint = etree.fromstring('<joint name="rooty" type="free" stiffness="0" limited="false" armature="0" damping="0"/>')
            torso_z = etree.fromstring('<joint name="rootz" type="slide" axis="0 0 1" limited="false" armature="0" damping="0"/>')
            torso_x = etree.fromstring('<joint name="rootx" type="slide" axis="1 0 0" limited="false" armature="0" damping="0"/>')
            torso_y = etree.fromstring('<joint name="rooty" type="hinge" axis="0 1 0" limited="false" armature="0" damping="0"/>')
            
            # NOTE: this order is mandatory for this particular task.
            # the bug may be caused by dm_control suite
            torso.append(torso_z)
            torso.append(torso_x)
            torso.append(torso_y)

            return torso 

        
        ################ START PARSING THE WORLDBODY ################
        worldbody = etree.Element('worldbody')
        worldbody = worldbody_preliminary(worldbody)
        
        # parse the matrix
        N, _ = adj_matrix.shape

        body_dict = {}
        info_dict = {} # log the needed information given a node
        
        # the root of the model is always fixed
        body_root = etree.Element('body', name='torso', pos='0 0 1.3', 
                                  childclass='walker')
        body_root = torso_preliminary(body_root)
        root_info = {}

        root_info['a_size'] = node_attr_list[0]['a_size']
        root_info['b_size'] = node_attr_list[0]['b_size']
        root_info['c_size'] = node_attr_list[0]['c_size']
        # for determining the center of the capsule relative to the body joint
        root_info['center_rel_pos'] = 0


        info_dict[0] = root_info
        body_dict[0] = body_root

        # initilize the parent list to go throught the entire matrix
        parent_list = [0]
        while len(parent_list) != 0:
            parent_node = parent_list.pop(0)

            parent_row = np.copy(adj_matrix[parent_node])
            for i in range(parent_node+1): parent_row[i] = 0
            child_list = np.where(parent_row)[0].tolist()

            while True:
                
                try: child_node = child_list.pop(0)
                except: break

                # parent-child relationship 
                # print('P-C relationship:', parent_node, child_node)
                node_attr = node_attr_list[child_node]
                node_name = 'node-%d'%(child_node)

                # this is parent's ellipsoid information
                parent_info = info_dict[parent_node]
                a_parent = parent_info['a_size']
                b_parent = parent_info['b_size']
                c_parent = parent_info['c_size']
                center_rel_pos = parent_info['center_rel_pos']
                
                # getting node attributes from the list
                u = node_attr['u']  # using these 2 values for determining the range 
                v = node_attr['v']  # of the joint
                
                axis_x = node_attr['axis_x'] # used to determine the relative position
                axis_y = node_attr['axis_y'] # w.r.t the parent capsule

                a_child = node_attr['a_size'] # using this as the capsule radius
                b_child = node_attr['b_size'] # using this as the capsule h
                c_child = node_attr['c_size']

                # compute the translational and rotational matrix
                child_info = {}

                # store attributes that defines the child's geom
                child_info['a_size'] = a_child
                child_info['b_size'] = b_child
                child_info['c_size'] = c_child
                
                # set the stitching point relative to parent
                a = min([node_attr['axis_x'], node_attr['axis_y']])
                b = max([node_attr['axis_x'], node_attr['axis_y']])
                stitch_ratio = node_attr['axis_x'] / 1
                
                if not(stitch_ratio <= 1.01 and stitch_ratio >= -1.01):
                    import pdb; pdb.set_trace()
                stitch_pt = b_parent * stitch_ratio + center_rel_pos
                body_pos = '0 0 %f' % (stitch_pt)

                # body translation
                if node_attr['axis_x'] * node_attr['axis_y'] >= 0:
                    geom_pos = '0 0 -%f' % (b_child)
                    child_info['center_rel_pos'] = -b_child
                else:
                    geom_pos = '0 0 %f' % (b_child)
                    child_info['center_rel_pos'] = b_child

                joint_pos = '0 0 0'

                # now create the body
                body_child = etree.Element('body', name=node_name, pos=body_pos)
                
                # add geom
                geom_type = 2 # for all planar creates, we use capsule
                capsule_size = '%f %f' % (a_child, b_child)
                geom = etree.Element('geom', name=node_name, pos=geom_pos, size=capsule_size)
                body_child.append(geom)
                
                # add joints
                joint_type = adj_matrix[parent_node, child_node]
                joint_axis = model_gen_util.get_encoding(joint_type)
                joint_range = node_attr['joint_range']
                range1 = node_attr['u'] / (2.0 * np.pi) * -90
                range2 = node_attr['v'] / (1.0 * np.pi) * 60 + 1
                range2 = range1 + 90 + 1
                if joint_axis[0] == 1:
                    x_joint = etree.fromstring("<joint name='%d-%d_x' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(x_joint)
                if joint_axis[1] == 1:
                    y_joint = etree.fromstring("<joint name='%d-%d_y' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(y_joint)
                if joint_axis[2] == 1:
                    z_joint = etree.fromstring("<joint name='%d-%d_z' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(z_joint)

                # need to add 2 sites as sensory inputs 
                site1_pos = '0 0 0'
                site2_pos = '0 0 %f' % (2 * child_info['center_rel_pos'])
                site1 = etree.Element('site', name='touch1-%s' % (node_name), pos=site1_pos)
                site2 = etree.Element('site', name='touch2-%s' % (node_name), pos=site2_pos)
                body_child.append(site1)
                body_child.append(site2)

                # logging the information
                body_dict[parent_node].append(body_child)
                body_dict[child_node] = body_child # register child's body struct in case it has child
                info_dict[child_node] = child_info
                # child becomes the parent for further examination
                parent_list.append(child_node)

        worldbody.append(body_dict[0])
        root.append(worldbody)
        return root

    def add_mujoco_sensor(root, adj_matrix):
        ''' this is specific for planar creature
        the body parts need to be informed with their collision with the ground
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        node_order = [0] + [int(item.split('-')[-1]) for item in dfs_order]

        sensor = etree.Element('sensor')

        for i in range(1, len(node_order)):
            node_name = 'node-%d' % (i)

            site1 = etree.Element('touch', name='touch1-%s' % (node_name),
                site='touch1-%s' % (node_name)
            )
            site2 = etree.Element('touch', name='touch2-%s' % (node_name),
                site='touch2-%s' % (node_name)
            )
            sensor.append(site1)
            sensor.append(site2)

        root.append(sensor)
        return root

    def add_mujoco_actuator(root, adj_matrix, node_attr):
        ''' essentially, all the joints are actuators.
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        actuator = etree.Element('actuator')

        for edge in dfs_order:

            p, c = [int(x) for x in edge.split('-')]

            joint_type = adj_matrix[p, c]
            joint_axis = model_gen_util.get_encoding(joint_type)
            edges = []
            if joint_axis[0] == 1:
                edge_x = '%s_x' % edge
                edges.append(edge_x)
            elif joint_axis[1] == 1:
                edge_y = '%s_y' % edge
                edges.append(edge_y)
            elif joint_axis[2] == 1:
                edge_z = '%s_z' % edge
                edges.append(edge_z)

            c_info = node_attr[c]['c_size']
            gear_value = c_info

            positions = [etree.Element('motor', 
                         name=item, joint=item, gear='%d' % (gear_value)) 
                         for item in edges]
            for item in positions: actuator.append(item)
        root.append(actuator)
        return root



    ############## ACTUAL CODES FOR GENERATION ##############
    root = etree.Element("mujoco", model='planar walker')
    root = add_mujoco_header(root)
    root = add_mujoco_options(root, options)
    root = add_worldbody(root, adj_matrix, node_attr)
    root = add_mujoco_sensor(root, adj_matrix)
    root = add_mujoco_actuator(root, adj_matrix, node_attr)
    # root = add_mujoco_sensor(root)

    save_generated_xml(root, filename)

    return root

def hopper_xml_generator(adj_matrix, node_attr, options=None, filename=None):
    ''' generate xml for hopper
    '''
    def add_mujoco_header(root):
        incl_1 = etree.Element('include', file='./common/skybox.xml')
        incl_2 = etree.Element('include', file='./common/visual.xml')
        incl_3 = etree.Element('include', file='./common/materials.xml')
        root.append(incl_1)
        root.append(incl_2)
        root.append(incl_3)
        return root

    def add_mujoco_options(root, options):
        # similar to walker but the attribute value is different
        option = etree.Element('option', timestep='0.005')
        statistic = etree.Element('statistic', extent='2', center='0 0 0.5')
        root.append(option)
        root.append(statistic)

        # default
        default_sec = etree.Element('default')
        
        # default 1: hopper class
        default_1 = etree.fromstring('<default class="hopper"></default>')
        default_1_joint = etree.Element('joint', type='hinge',
            axis='0 1 0', limited='true', damping='0.05', armature='.2'
        )
        default_1_geom = etree.Element('geom', type='capsule', material='self')
        default_1_site = etree.Element('site', type='sphere', size='0.05', group='3')
        default_1.append(default_1_joint)
        default_1.append(default_1_geom)
        default_1.append(default_1_site)

        default_2 = etree.fromstring('<default class="free"></default>')
        default_2_joint = etree.Element('joint', limited='false', damping='0',
            armature='0', stiffness='0'
        )
        default_2.append(default_2_joint)

        default_motor = etree.Element('motor', ctrlrange='-1 1', ctrllimited='true')
        
        default_sec.append(default_1)
        default_sec.append(default_2)
        default_sec.append(default_motor)

        root.append(default_sec)
        return root

    def add_worldbody(root, adj_matrix, node_attr_list):
        '''
        '''
        def worldbody_preliminary(worldbody):
            '''
            '''
            geom = etree.Element('geom', name='floor', type='plane',
                conaffinity='1', pos='48 0 0', size='50 1 .2', material='grid'
            )
            worldbody.append(geom)

            camera1 = etree.Element('camera', name='cam0',
                pos='0 -2.8 0.8', euler='90 0 0',
                mode='trackcom'
            )
            camera2 = etree.Element('camera', name='back',
                pos='-2 -.2 1.2', xyaxes='0.2 -1 0 .5 0 2',
                mode='trackcom'
            )
            worldbody.append(camera1)
            worldbody.append(camera2)
 
            return worldbody

        def torso_preliminary(torso):
            light = etree.Element('light', name='top', pos='0 0 2', mode='trackcom')
            torso.append(light)

            joint_x = etree.fromstring('<joint name="rootx" type="slide" axis="1 0 0" class="free"/>') 
            joint_y = etree.fromstring('<joint name="rooty" type="hinge" axis="0 1 0" class="free"/>') 
            joint_z = etree.fromstring('<joint name="rootz" type="slide" axis="0 0 1" class="free"/>') 
            torso.append(joint_x)
            torso.append(joint_z)
            torso.append(joint_y)
            
            geom1 = etree.Element('geom', name='torso', fromto='0 0 -.05 0 0 .2', size='0.0653')
            geom2 = etree.Element('geom', name='nose', fromto='.08 0 .13 .15 0 .14', size='0.03')
            torso.append(geom1)
            torso.append(geom2)

            return torso

        ########### START OF PARSING THE WORLDBODY ###########
        # this part should be identical to that of walker
        worldbody = etree.Element('worldbody')
        worldbody = worldbody_preliminary(worldbody)
        
        # parse the matrix
        N, _ = adj_matrix.shape

        body_dict = {}
        info_dict = {} # log the needed information given a node
        
        # the root of the model is always fixed
        body_root = etree.Element('body', name='torso', pos='0 0 1', 
                                  childclass='hopper')
        body_root = torso_preliminary(body_root)
        root_info = {}

        root_info['a_size'] = node_attr_list[0]['a_size']
        root_info['b_size'] = node_attr_list[0]['b_size']
        root_info['c_size'] = node_attr_list[0]['c_size']
        # for determining the center of the capsule relative to the body joint
        root_info['center_rel_pos'] = 0


        info_dict[0] = root_info
        body_dict[0] = body_root

        # initilize the parent list to go throught the entire matrix
        parent_list = [0]
        while len(parent_list) != 0:
            parent_node = parent_list.pop(0)

            parent_row = np.copy(adj_matrix[parent_node])
            for i in range(parent_node+1): parent_row[i] = 0
            child_list = np.where(parent_row)[0].tolist()

            while True:
                
                try: child_node = child_list.pop(0)
                except: break

                # parent-child relationship 
                # print('P-C relationship:', parent_node, child_node)
                node_attr = node_attr_list[child_node]
                node_name = 'node-%d'%(child_node)

                # this is parent's ellipsoid information
                parent_info = info_dict[parent_node]
                a_parent = parent_info['a_size']
                b_parent = parent_info['b_size']
                c_parent = parent_info['c_size']
                center_rel_pos = parent_info['center_rel_pos']
                
                # getting node attributes from the list
                u = node_attr['u']  # using these 2 values for determining the range 
                v = node_attr['v']  # of the joint
                
                axis_x = node_attr['axis_x'] # used to determine the relative position
                axis_y = node_attr['axis_y'] # w.r.t the parent capsule

                a_child = node_attr['a_size'] # using this as the capsule radius
                b_child = node_attr['b_size'] # using this as the capsule h
                c_child = node_attr['c_size']

                # compute the translational and rotational matrix
                child_info = {}

                # store attributes that defines the child's geom
                child_info['a_size'] = a_child
                child_info['b_size'] = b_child
                child_info['c_size'] = c_child
                
                # set the stitching point relative to parent
                a = min([node_attr['axis_x'], node_attr['axis_y']])
                b = max([node_attr['axis_x'], node_attr['axis_y']])
                stitch_ratio = node_attr['axis_x'] / 1
                
                if not(stitch_ratio <= 1.01 and stitch_ratio >= -1.01):
                    import pdb; pdb.set_trace()
                stitch_pt = b_parent * stitch_ratio + center_rel_pos
                body_pos = '0 0 %f' % (stitch_pt)

                # body translation
                if node_attr['axis_x'] * node_attr['axis_y'] >= 0:
                    geom_pos = '0 0 -%f' % (b_child)
                    child_info['center_rel_pos'] = -b_child
                else:
                    geom_pos = '0 0 %f' % (b_child)
                    child_info['center_rel_pos'] = b_child

                joint_pos = '0 0 0'

                # now create the body
                body_child = etree.Element('body', name=node_name, pos=body_pos)
                
                # add geom
                geom_type = 2 # for all planar creates, we use capsule
                capsule_size = '%f %f' % (a_child, b_child)
                geom = etree.Element('geom', name=node_name, pos=geom_pos, size=capsule_size)
                body_child.append(geom)
                
                # add joints
                joint_type = adj_matrix[parent_node, child_node]
                joint_axis = model_gen_util.get_encoding(joint_type)
                joint_range = node_attr['joint_range']
                range1 = node_attr['u'] / (2.0 * np.pi) * -90
                range2 = node_attr['v'] / (1.0 * np.pi) * 150 + 1
                range2 = range1 + 90 + 1
                if joint_axis[0] == 1:
                    x_joint = etree.fromstring("<joint name='%d-%d_x' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(x_joint)
                if joint_axis[1] == 1:
                    y_joint = etree.fromstring("<joint name='%d-%d_y' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(y_joint)
                if joint_axis[2] == 1:
                    z_joint = etree.fromstring("<joint name='%d-%d_z' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(z_joint)

                site1_pos = '0 0 0'
                site2_pos = '0 0 %f' % (2 * child_info['center_rel_pos'])
                site1 = etree.Element('site', name='touch1-%s' % (node_name), pos=site1_pos)
                site2 = etree.Element('site', name='touch2-%s' % (node_name), pos=site2_pos)
                body_child.append(site1)
                body_child.append(site2)

                # logging the information
                body_dict[parent_node].append(body_child)
                body_dict[child_node] = body_child # register child's body struct in case it has child
                info_dict[child_node] = child_info
                # child becomes the parent for further examination
                parent_list.append(child_node)

        worldbody.append(body_dict[0])
        root.append(worldbody)
        return root

    def add_mujoco_sensor(root, adj_matrix):
        ''' this is specific for planar creature
        the body parts need to be informed with their collision with the ground
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        node_order = [0] + [int(item.split('-')[-1]) for item in dfs_order]

        sensor = etree.Element('sensor')

        for i in range(1, len(node_order)):
            node_name = 'node-%d' % (i)

            site1 = etree.Element('touch', name='touch1-%s' % (node_name),
                site='touch1-%s' % (node_name)
            )
            site2 = etree.Element('touch', name='touch2-%s' % (node_name),
                site='touch2-%s' % (node_name)
            )
            sensor.append(site1)
            sensor.append(site2)

        root.append(sensor)
        return root

    def add_mujoco_actuator(root, adj_matrix):
        ''' this part should be exactly the same as walker
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        actuator = etree.Element('actuator')

        for edge in dfs_order:

            p, c = [int(x) for x in edge.split('-')]

            joint_type = adj_matrix[p, c]
            joint_axis = model_gen_util.get_encoding(joint_type)
            edges = []
            if joint_axis[0] == 1:
                edge_x = '%s_x' % edge
                edges.append(edge_x)
            elif joint_axis[1] == 1:
                edge_y = '%s_y' % edge
                edges.append(edge_y)
            elif joint_axis[2] == 1:
                edge_z = '%s_z' % edge
                edges.append(edge_z)

            positions = [etree.Element('motor', name=item, joint=item, gear='30') 
                         for item in edges]
            for item in positions: actuator.append(item)
        root.append(actuator)
        return root

    ################## Actual codes for generating hopper ##################

    root = etree.Element('mujoco', model='planar hopper')
    root = add_mujoco_header(root)
    root = add_mujoco_options(root, options)
    root = add_worldbody(root, adj_matrix, node_attr)
    root = add_mujoco_sensor(root, adj_matrix)
    root = add_mujoco_actuator(root, adj_matrix)

    if filename is not None:
        tree = etree.ElementTree(root)
        tree.write('./assets/gen' + filename + '.xml', 
            pretty_print=True,
            xml_declaration=True, encoding='utf-8'
        )
    return root

def cheetah_xml_generator(adj_matrix, node_attr, options=None, filename=None):
    ''' generate the xml file 
    '''
    def add_mujoco_header(root):
        incl_1 = etree.Element('include', file='./common/visual.xml')
        incl_2 = etree.Element('include', file='./common/skybox.xml')
        incl_3 = etree.Element('include', file='./common/materials.xml')
        root.append(incl_1)
        root.append(incl_2)
        root.append(incl_3)
        return root

    def add_mujoco_options(root, options):
        ''' @brief:
                several attributes that are different from hopper
        '''
        compiler = etree.Element('compiler', settotalmass='14')
        root.append(compiler)

        default_sec = etree.Element('default')
        default_1 = etree.fromstring('<default class="cheetah"></default>')
        default_1_joint = etree.Element('joint', limited='true', damping='0.01',
            armature='0.1', stiffness='8', type='hinge', axis='0 1 0'
        )
        default_1_geom = etree.Element('geom', contype='1', conaffinity='1',
            condim='3', friction='.4 .1 .1', material='self'
        )
        default_1_site = etree.Element('site', type='sphere', size='0.05', group='3')
        default_1.append(default_1_joint)
        default_1.append(default_1_geom)
        default_1.append(default_1_site)
        default_2 = etree.fromstring('<default class="free"></default>')
        default_2_joint = etree.Element('joint', limited='false', damping='0',
            armature='0', stiffness='0'
        )
        default_2.append(default_2_joint)
        motor = etree.Element('motor', ctrllimited='true', ctrlrange='-1 1')

        default_sec.append(default_1)
        default_sec.append(default_2)
        default_sec.append(motor)

        root.append(default_sec)
        return root

    def add_worldbody(root, adj_matrix, node_attr_list):
        '''
        '''
        def worldbody_preliminary(worldbody):
            '''
            '''
            geom1 = etree.Element('geom', name='ground', type='plane', 
                conaffinity='1', pos='98 0 0', size='100 .8 .5',
                material='grid'
            )
            worldbody.append(geom1)
            return worldbody

        def torso_preliminary(torso):
            '''
            '''
            light = etree.Element('light', name='light', 
                pos='0 0 2', mode='trackcom'
            )
            camera1 = etree.Element('camera', name='side', pos='0 -3 0',
                quat='0.707 0.707 0 0', mode='trackcom'
            )
            camera2 = etree.Element('camera', name='back', pos='-1.8 -1.3 0.8',
                xyaxes='0.45 -0.9 0 0.3 0.15 0.94', mode='trackcom'
            )

            torso.append(light)
            torso.append(camera1)
            torso.append(camera2)

            jointx = etree.fromstring(' <joint name="rootx" type="slide" axis="1 0 0" class="free"/>')
            jointy = etree.fromstring(' <joint name="rooty" type="hinge" axis="0 1 0" class="free"/>')
            jointz = etree.fromstring(' <joint name="rootz" type="slide" axis="0 0 1" class="free"/>')
            torso.append(jointx)
            torso.append(jointz)
            torso.append(jointy)

            geom1 = etree.Element('geom', name='torso', type='capsule',
                size='0.046 0.5'
            )
            geom2 = etree.Element('geom', name='head', type='capsule',
                pos='-0.1 0 .6', euler='0 140 0', size='0.046 0.15'
            )
            torso.append(geom1)
            torso.append(geom2)


            return torso

        ################ START PARSING THE WORLDBODY ################
        worldbody = etree.Element('worldbody')
        worldbody = worldbody_preliminary(worldbody)
        
        # parse the matrix
        N, _ = adj_matrix.shape

        body_dict = {}
        info_dict = {} # log the needed information given a node
        
        # the root of the model is always fixed
        body_root = etree.Element('body', name='torso', pos='0 0 1.3', 
                                  childclass='cheetah')
        body_root = torso_preliminary(body_root)
        root_info = {}

        root_info['a_size'] = node_attr_list[0]['a_size']
        root_info['b_size'] = node_attr_list[0]['b_size']
        root_info['c_size'] = node_attr_list[0]['c_size']
        # for determining the center of the capsule relative to the body joint
        root_info['center_rel_pos'] = 0


        info_dict[0] = root_info
        body_dict[0] = body_root

        # initilize the parent list to go throught the entire matrix
        parent_list = [0]
        while len(parent_list) != 0:
            parent_node = parent_list.pop(0)

            parent_row = np.copy(adj_matrix[parent_node])
            for i in range(parent_node+1): parent_row[i] = 0
            child_list = np.where(parent_row)[0].tolist()

            while True:
                
                try: child_node = child_list.pop(0)
                except: break

                # parent-child relationship 
                # print('P-C relationship:', parent_node, child_node)
                node_attr = node_attr_list[child_node]
                node_name = 'node-%d'%(child_node)

                # this is parent's ellipsoid information
                parent_info = info_dict[parent_node]
                a_parent = parent_info['a_size']
                b_parent = parent_info['b_size']
                c_parent = parent_info['c_size']
                center_rel_pos = parent_info['center_rel_pos']
                
                # getting node attributes from the list
                u = node_attr['u']  # using these 2 values for determining the range 
                v = node_attr['v']  # of the joint
                
                axis_x = node_attr['axis_x'] # used to determine the relative position
                axis_y = node_attr['axis_y'] # w.r.t the parent capsule

                a_child = node_attr['a_size'] # using this as the capsule radius
                b_child = node_attr['b_size'] # using this as the capsule h
                c_child = node_attr['c_size']

                # compute the translational and rotational matrix
                child_info = {}

                # store attributes that defines the child's geom
                child_info['a_size'] = a_child
                child_info['b_size'] = b_child
                child_info['c_size'] = c_child
                
                # set the stitching point relative to parent
                a = min([node_attr['axis_x'], node_attr['axis_y']])
                b = max([node_attr['axis_x'], node_attr['axis_y']])
                stitch_ratio = node_attr['axis_x'] / 1
                
                if not(stitch_ratio <= 1.01 and stitch_ratio >= -1.01):
                    import pdb; pdb.set_trace()
                stitch_pt = b_parent * stitch_ratio + center_rel_pos
                body_pos = '0 0 %f' % (stitch_pt)

                # body translation
                if node_attr['axis_x'] * node_attr['axis_y'] >= 0:
                    geom_pos = '0 0 -%f' % (b_child)
                    child_info['center_rel_pos'] = -b_child
                else:
                    geom_pos = '0 0 %f' % (b_child)
                    child_info['center_rel_pos'] = b_child

                joint_pos = '0 0 0'

                # now create the body
                body_child = etree.Element('body', name=node_name, pos=body_pos)
                
                # add geom
                geom_type = 2 # for all planar creates, we use capsule
                capsule_size = '%f %f' % (a_child, b_child)
                geom = etree.Element('geom', name=node_name, 
                    pos=geom_pos, size=capsule_size, type='capsule'
                    )
                body_child.append(geom)
                
                # add joints
                joint_type = adj_matrix[parent_node, child_node]
                joint_axis = model_gen_util.get_encoding(joint_type)
                joint_range = node_attr['joint_range']
                range1 = node_attr['u'] / (2.0 * np.pi) * -90
                range2 = node_attr['v'] / (1.0 * np.pi) * 60 + 1
                range2 = range1 + 90 + 1
                if joint_axis[0] == 1:
                    x_joint = etree.fromstring("<joint name='%d-%d_x' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(x_joint)
                if joint_axis[1] == 1:
                    y_joint = etree.fromstring("<joint name='%d-%d_y' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(y_joint)
                if joint_axis[2] == 1:
                    z_joint = etree.fromstring("<joint name='%d-%d_z' axis='0 -1 0' pos='%s' range='%d %d'/>" % \
                        (parent_node, child_node, joint_pos, range1, range2))
                    body_child.append(z_joint)

                # site1_pos = '0 0 0'
                # site1 = etree.Element('geom', type='sphere', pos=site1_pos, size='0.08', material='target')
                # body_child.append(site1)

                # center = child_info['center_rel_pos']
                # site2_pos = '0 0 %f' % (2 * center)
                # site2 = etree.Element('geom', type='sphere', pos=site2_pos, size='0.08', material='target')
                # body_child.append(site2)
                

                # need to add 2 sites as sensory inputs 
                site1_pos = '0 0 0'
                site2_pos = '0 0 %f' % (2 * child_info['center_rel_pos'])
                site1 = etree.Element('site', name='touch1-%s' % (node_name), pos=site1_pos)
                site2 = etree.Element('site', name='touch2-%s' % (node_name), pos=site2_pos)
                body_child.append(site1)
                body_child.append(site2)

                # logging the information
                body_dict[parent_node].append(body_child)
                body_dict[child_node] = body_child # register child's body struct in case it has child
                info_dict[child_node] = child_info
                # child becomes the parent for further examination
                parent_list.append(child_node)

        worldbody.append(body_dict[0])
        root.append(worldbody)
        return root



    def add_mujoco_sensor(root, adj_matrix):
        ''' this is specific for planar creature
        the body parts need to be informed with their collision with the ground
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        node_order = [0] + [int(item.split('-')[-1]) for item in dfs_order]

        sensor = etree.Element('sensor')

        for i in range(1, len(node_order)):
            node_name = 'node-%d' % (i)

            site1 = etree.Element('touch', name='touch1-%s' % (node_name),
                site='touch1-%s' % (node_name)
            )
            site2 = etree.Element('touch', name='touch2-%s' % (node_name),
                site='touch2-%s' % (node_name)
            )
            sensor.append(site1)
            sensor.append(site2)

        root.append(sensor)
        return root

    def add_mujoco_actuator(root, adj_matrix, node_attr):
        ''' essentially, all the joints are actuators.
        '''
        dfs_order = model_gen_util.dfs_order(adj_matrix)
        actuator = etree.Element('actuator')

        for edge in dfs_order:

            p, c = [int(x) for x in edge.split('-')]

            joint_type = adj_matrix[p, c]
            joint_axis = model_gen_util.get_encoding(joint_type)
            edges = []
            if joint_axis[0] == 1:
                edge_x = '%s_x' % edge
                edges.append(edge_x)
            elif joint_axis[1] == 1:
                edge_y = '%s_y' % edge
                edges.append(edge_y)
            elif joint_axis[2] == 1:
                edge_z = '%s_z' % edge
                edges.append(edge_z)

            c_info = node_attr[c]['c_size']
            gear_value = c_info

            positions = [etree.Element('motor', 
                         name=item, joint=item, gear='%d' % (gear_value)) 
                         for item in edges]
            for item in positions: actuator.append(item)
        root.append(actuator)
        return root

    ################## Actual codes for generating hopper ##################
    root = etree.Element('mujoco', model='cheetah')
    root = add_mujoco_header(root)
    root = add_mujoco_options(root, options)
    root = add_worldbody(root, adj_matrix, node_attr)
    root = add_mujoco_sensor(root, adj_matrix)
    root = add_mujoco_actuator(root, adj_matrix, node_attr)

    if filename is not None:
        tree = etree.ElementTree(root)
        tree.write('./assets/gen' + filename + '.xml', 
            pretty_print=True,
            xml_declaration=True, encoding='utf-8'
        )
    return root


################# SOME CODES FOR TESTING #################

def gen_test_adj_mat(task='fish', hinge_opt=7, shape=(5, 5)):
    ''' generate a testing matrix
    '''
    adj_matrix = np.random.randint(hinge_opt, size=shape)
    N, _ = adj_matrix.shape

    # node cannot connect to itself
    for i in range(N):
        adj_matrix[i, i] = 0

    # lower half of the matrix is empty
    for i in range(N):
        for j in range(i):
            adj_matrix[i, j] = 0

    # except for root, other nodes (col with id >= 1)'s col entry must have 
    #                 at least 1 being non-zero -> nothing being left out
    #                 at most 1 being non-zero  -> can only be connected once
    for i in range(1, N):
        col_i = adj_matrix[:i,i]
        connected = np.where(col_i > 0)[0]

        if connected.size == 1: 
            continue
        elif connected.size == 0: # randomly select a node to connect from
            parent_node = np.random.randint(i)
            col_i[parent_node] = 1 # this should be randomly sampled [1, 7]
        else: # clear up the abundant node
            connected = connected.tolist()
            while len(connected) > 1:
                rm_node = random.sample(connected, 1)[0]
                col_i[rm_node] = 0
                connected.remove(rm_node)
            
    # make the edge be bi-directional
    for i in range(N):
        for j in range(i, N):
            adj_matrix[j, i] = adj_matrix[i, j]

    # for now, all the connection is a socket
    if 'fish' in task:
        adj_matrix[adj_matrix > 0] = 7
    elif 'walker' in task:
        adj_matrix[adj_matrix > 0] = 2

    return adj_matrix

def gen_test_node_attr(task='fish', node_num=5, discrete_rv=True):
    '''
    '''
    def root_default(task):
        root_attr = {}

        if 'fish' in task:
            # trivial information that shouldn't be used
            root_attr['geom_type'] = -1
            root_attr['u'] = -1
            root_attr['v'] = -1
            root_attr['axis_x'] = -1 
            root_attr['axis_y'] = -1
            # this is the default value as in dm_control original repo
            root_attr['a_size'] = 0.0075
            root_attr['b_size'] = 0.06
            root_attr['c_size'] = root_attr['a_size'] * ELLIPSOID_Z_RATIO
            # 
            root_attr['joint_range'] = 60
        elif 'walker' in task:
            # some trivial attributes
            root_attr['geom_type'] = -1
            root_attr['u'] = -1
            root_attr['v'] = -1
            root_attr['axis_x'] = -1
            root_attr['axis_y'] = -1
            root_attr['joint_range'] = 60
            # setting the torso, using ellipsoid to approximate cylinder
            root_attr['a_size'] = 0.07
            root_attr['b_size'] = 0.3
            root_attr['c_size'] = -1
        elif 'hopper' in task:
            root_attr = species_info.CREATURE_ROOT_INFO['hopper']
        elif 'cheetah' in task:
            root_attr = species_info.CREATURE_ROOT_INFO['cheetah']
        else:
            assert 0, 'task: %s, not supported' % task

        return root_attr

    def gen_one_attr(task, discrete_rv=True):
        '''
        '''
        node_attr = {}

        # define geom type
        node_attr['geom_type'] = random.randint(0, 0)

        if 'fish' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['fish']
            needed_attr = ['u', 'v', 'axis_x', 'axis_y', 
                           'a_size', 'b_size', 'c_size', 'joint_range']
            for item in needed_attr:
                low, high = constraint[item]
                node_attr[item] = model_gen_util.get_uniform(
                    low, high, discrete=discrete_rv
                    )
                if item == 'joint_range':
                    node_attr[item] = int(node_attr[item])

        elif 'walker' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['walker']
            needed_attr = ['u', 'v', 'axis_x', 'axis_y', 
                           'a_size', 'b_size', 'c_size', 'joint_range']
            for item in needed_attr:
                low, high = constraint[item]
                node_attr[item] = model_gen_util.get_uniform(
                    low, high, discrete=discrete_rv
                    )
                if item == 'joint_range':
                    node_attr[item] = int(node_attr[item])

        elif 'hopper' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['hopper']
            needed_attr = ['u', 'v', 'axis_x', 'axis_y', 
                           'a_size', 'b_size', 'c_size', 'joint_range']
            for item in needed_attr:
                low, high = constraint[item]
                node_attr[item] = model_gen_util.get_uniform(
                    low, high, discrete=discrete_rv
                    )
                if item == 'joint_range':
                    node_attr[item] = int(node_attr[item])
        elif 'cheetah' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['walker']
            needed_attr = ['u', 'v', 'axis_x', 'axis_y', 
                           'a_size', 'b_size', 'c_size', 'joint_range']
            for item in needed_attr:
                low, high = constraint[item]
                node_attr[item] = model_gen_util.get_uniform(
                    low, high, discrete=discrete_rv
                    )
                if item == 'joint_range':
                    node_attr[item] = int(node_attr[item])

        return node_attr

    node_attr_list = []
    for i in range(node_num):
        node_attr = gen_one_attr(task, discrete_rv=discrete_rv)
        node_attr_list.append(node_attr)

    node_attr_list[0] = root_default(task)
    return node_attr_list


def get_initial_settings():
    ''' used for brute force topological search
    '''
    body_num = random.randint(2, 10)
    adj_matrix = gen_test_adj_mat(shape=(body_num, body_num))
    node_attr = gen_test_node_attr(node_num=body_num)

    xml_root = fish_xml_generator(adj_matrix, node_attr, options=None)
    return adj_matrix, node_attr, xml_root


if __name__ == '__main__':

    # ################ START OF THE CODE #################
    adj_matrix = gen_test_adj_mat(shape=(5, 5))
    node_attr = gen_test_node_attr(node_num=5)

    root = xml_generator(adj_matrix, node_attr, options=None)

    import pdb; pdb.set_trace()

