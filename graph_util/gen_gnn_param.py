'''
    @brief:
    @author:
        Henry Zhou, March 31st, 2018
    @update:
'''

import init_path
import numpy as np
from util import model_gen_util
from env import model_gen

ROOT_PATH = init_path.get_abs_base_dir()
'''
    definition of nodes:
    @root:
        the 'root' node

'''
_NODE_TYPE = ['root', 'joint', 'body']  # geom is removed
EDGE_TYPE = {'self_loop': 0, 'root-root': 0,  # root-root is loop, also 0
             'joint-joint': 1, 'geom-geom': 2, 'body-body': 3, 'tendon': 4,
             'joint-geom': 5, 'geom-joint': -5,  # pc-relationship
             'joint-body': 6, 'body-joint': -6,
             'body-geom': 7, 'geom-body': -7,
             'root-geom': 8, 'geom-root': -8,
             'root-joint': 9, 'joint-root': -9,
             'root-body': 10, 'body-root': -10}


def gen_gnn_param(task_name,
                  adj_mat, node_attr_list,
                  gnn_node_option='nG,yB',
                  root_connection_option='nN, Rn, sE',
                  gnn_output_option='shared',
                  gnn_embedding_option='parameter'):
    ''' this function resembles the functionality of parse_mujoco_graph
        in mujoco_parser.py
        WARNING: this is subject to evolution fish3d environment only
        which has the following features
            1. the mujoco model is deterministically generated from
               adj_mat and node_attr_list
            2. the structure of the model follows
                a. body-body connection is through 3 joints
                b. the root body has 3 joints
    '''
    def actuator_order(adj_mat):
        ''' given the connection information,
            generate the order of the actuator
            NOTE: this should be consistant with model_gen.py -> add_mujoco_actuator()
        '''
        N, _ = adj_mat.shape

        connection_list = []
        for i in range(N):
            for j in range(i, N):
                node_edge = '%d-%d' % (i, j)
                connection_list.append(node_edge)
        return connection_list

    # step 1: directly adopt the relation_matrix from the current adj_matrix
    connection_list = model_gen_util.dfs_order(adj_mat)
    relation_matrix = np.zeros(adj_mat.shape)
    relation_matrix[adj_mat > 0] = 2

    # step 2: tree structure parsed from the adj_mat
    tree, node_type_dict = _get_tree_struct(adj_mat, node_attr_list)

    # step 3: map the input_list
    input_dict, ob_size = _get_input_dict(tree, adj_mat, task_name)

    # step 4: map the action_list
    output_list, output_type_dict, action_size = _get_output_dict(
        tree, adj_mat, gnn_output_option, task_name
    )

    node_param, param_size_dict = _append_node_attr(tree, adj_mat, node_attr_list)

    # step 5: get the node parameters
    debug_info = {'ob_size': ob_size, 'action_size': action_size}

    # post_process, if using noninput_embedding

    if gnn_embedding_option == 'noninput_separate':
        node_param['root'] = np.reshape(np.array(0, dtype=np.int), [1, 1])
        node_param['joint'] = np.reshape(
            np.array(range(1, 1 + len(node_type_dict['joint'])), dtype=np.int),
            [-1, 1]
        )
        param_size_dict = {'joint': 1, 'root': 1}
        # from util import fpdb; fpdb.fpdb().set_trace()

    elif gnn_embedding_option == 'noninput_shared':
        assert False
        node_param['root'] = np.reshape(np.array(0, dtype=np.int), [1, 1])
        node_param['joint'] = np.reshape(
            np.ones(len(node_type_dict['joint']), dtype=np.int),
            [-1, 1]
        )
        param_size_dict = {'joint': 1, 'root': 1}

    else:
        assert gnn_embedding_option == 'parameter'
    # from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()

    return dict(tree=tree,
                relation_matrix=relation_matrix,
                node_type_dict=node_type_dict,
                output_type_dict=output_type_dict,
                input_dict=input_dict,
                output_list=output_list,
                debug_info=debug_info,
                node_parameters=node_param,
                para_size_dict=param_size_dict,
                num_nodes=len(tree))


def _get_tree_struct(adj_mat, node_attr_list):
    ''' parse the adj_mat and log the information from node_attr_list
        to its corresponding node_id
        output:
            tree: a dictionary indexed by the node_id
                tree[i]['name'] -> node_id
                tree[i]['neighbour'] -> list of parents and child
                tree[i]['parent'] -> list of parent (at most 1, root has zero parent)
                tree[i]['child'] -> 0 for leaf, reasonable amount for all the others
                tree[i]['info'] -> whatever attributes given by node_attr_list
                tree[i]['is_output_node'] -> False if root True otherwise
                tree[i]['output_type'] -> list of length 3
                                          index 0 1 2 corresponds to existance of x y z
                                          joints
                tree[i]['joint_num'] -> sum(tree[i]['output_type']) for ease of access
            node_type_dict:
                historic reason, preserving from mujoco_parser.py
    '''
    N, _ = adj_mat.shape

    tree = {}

    # give name and find neighbour
    for i in range(N):
        # initialize tree struct
        tree[i] = {}
        # check child
        child = adj_mat[i][i + 1:]
        child = list(set(np.where(child > 0)[0]))
        # check its parent
        parent = adj_mat[:, i][0: i]
        parent = list(set(np.where(parent > 0)[0]))
        # remove duplicate elements
        neighbour = child + parent
        # fill the node attribute
        tree[i]['name'] = str(i)
        tree[i]['neighbour'] = neighbour
        tree[i]['parent'] = parent  # can have at most 1 parent
        tree[i]['child'] = child

    # add debug info
    for i in range(N):
        tree[i]['info'] = node_attr_list[i]

    # add output and type of output
    tree[0]['is_output_node'] = False
    tree[0]['joint_num'] = 3
    for i in range(1, N):
        tree[i]['is_output_node'] = True
        parent_node = tree[i]['parent'][0]
        joint_type = adj_mat[parent_node, i]
        joint_axis = [int(item) for item in list(bin(joint_type)[2:])]
        joint_axis = [0 for i in range(3 - len(joint_axis))] + joint_axis
        joint_axis = list(reversed(joint_axis))
        tree[i]['output_type'] = joint_axis
        tree[i]['joint_num'] = int(np.array(joint_axis).sum())
        pass

    # all node types should be of joint
    tree[0]['type'] = 'root'
    for i in range(1, N):
        tree[i]['type'] = 'joint'

    # create node type dict
    node_type_dict = {}
    node_type_dict['root'] = [0]
    node_type_dict['joint'] = list(range(1, N))

    return tree, node_type_dict


def _get_input_dict(tree, adj_mat, task_name):
    '''
        output:
            input_dict (for fish only): dm_control returns a 
                                        concatenation of 'joint_angles', 'velocity'
                                        and optionally, 
                                        'target', depending on the task
                maps index of the tree to index of the observation vector corresponding
                to the node
                eg. input_dict[0] == [0, 1, 2, 31, 32, 33, 34, 35, 36]
                    root's observation corresponds to the specific indeces inside the
                    overall ob vector
            input_dict (for planar creatures such as walker, cheetah, hopper):
                our customized environment returns a concatenation of
                (assuming the creature contains N nodes, including the torso)
                'orientation': 2 * N
                'height': 1
                'velocity': 3 (for the rootxyz) + (N - 1)
                'touch': 2 * (N - 1) -- all body nodes 
                         except the root will have 2 seonsors

    '''
    # from util.fpdb import fpdb; fpdb().set_trace()
    N, _ = adj_mat.shape
    input_dict = {}
    ob_size = 0

    dfs_order = model_gen_util.dfs_order(adj_mat)
    node_order = [0] + [int(item.split('-')[-1]) for item in dfs_order]
    node_order = np.array(node_order)

    # ### map the node id to observation entry in the observation dict
    if 'fish' in task_name:
        joint_ptr = 0
        velocity_ptr = (N - 1) * 3
        if 'speed' in task_name:
            velocity_ptr += 1

        for i, node_id in enumerate(node_order):
            node_joint_num = tree[i]['joint_num']

            # add observation for joint angles
            if i == 0:  # root node
                if 'speed' in task_name:  # add the observation for rootz
                    input_dict[node_id] = [0]
                    joint_ptr += 1
                    ob_size += 1
                else:
                    input_dict[node_id] = []
            else:
                input_dict[node_id] = list(range(joint_ptr, joint_ptr + node_joint_num))
                joint_ptr += node_joint_num
                ob_size += node_joint_num

            # add observation for velocity
            input_dict[node_id] += list(range(velocity_ptr, velocity_ptr + node_joint_num))
            velocity_ptr += node_joint_num
            ob_size += node_joint_num

        # if the task is 'fish3d-swim', process target
        if 'swim' in task_name:
            input_dict[0] = input_dict[0] + list(range(ob_size, ob_size + 3))
            ob_size += 3

    elif 'walker' in task_name or 'hopper' in task_name or 'cheetah' in task_name:
        orient_ptr = 1
        velocity_ptr = orient_ptr + 2 * N
        touch_ptr = velocity_ptr + 3 + (N - 1)

        for i, node_id in enumerate(node_order):
            # print tree[i]
            node_joint_num = tree[i]['joint_num']
            # print 'success', i, node_id

            # process observations that are different between root and body
            # process height, velocity, touch
            if i == 0: # the root node
                assert 'speed' in task_name
                # process height
                input_dict[node_id] = [0]
                ob_size += 1
                # process velocity
                input_dict[node_id] += list(range(velocity_ptr,
                                                  velocity_ptr + 3))
                velocity_ptr += 3
                ob_size += 3
            else:
                # process velocity
                input_dict[node_id] =  list(range(velocity_ptr,
                                                  velocity_ptr + node_joint_num))
                velocity_ptr += node_joint_num
                ob_size += node_joint_num
                # process touch
                input_dict[node_id] += list(range(touch_ptr,
                                                  touch_ptr + 2))
                touch_ptr += 2
                ob_size += 2 
            
            # process orientations
            input_dict[node_id] += list(range(orient_ptr,
                                              orient_ptr + 2))
            orient_ptr += 2
            ob_size += 2

    return input_dict, ob_size


def _get_output_dict(tree, adj_mat, gnn_output_option, task_name):
    '''
        how to order the output of GGNN to the corresponding mujoco xml file
        the order is stored in output_order
    '''
    if 'fish' in task_name:
        # the order of apperence for each node in the xml file
        dfs_order = model_gen_util.dfs_order(adj_mat)
        output_order = [int(item.split('-')[-1]) for item in dfs_order]

        # output_type_dict
        output_type_dict = {}
        output_type_dict[gnn_output_option] = output_order

        # how many actions should each 1 node output
        # output size can be infered from the tree structure
        # for each node_i in output_order, it should output tree[node_i]['output_type']
        # tree[node_i]['output_type'] is a list of 3 elements
        # index 0 - exists x_axis joint output
        # index 1 - exists y_axis joint output
        # index 2 - exists z_axis joint output

        # all the actuators need one output
        N, _ = adj_mat.shape
        # the joints on the root are not actuators
        # action_size = joint_num =
        # sum([v['joint_num'] for k, v in tree.items()]) - tree[0]['joint_num']
        action_size = sum([v['joint_num'] for k, v in tree.items()]) - tree[0]['joint_num']
    elif 'walker' in task_name or 'cheetah' in task_name or 'hopper' in task_name:
        # NOTE: seemingly the same thing as the fish
        # the order of apperence for each node in the xml file
        dfs_order = model_gen_util.dfs_order(adj_mat)
        output_order = [int(item.split('-')[-1]) for item in dfs_order]

        # output_type_dict
        output_type_dict = {}
        output_type_dict[gnn_output_option] = output_order

        # how many actions should each 1 node output
        # output size can be infered from the tree structure
        # for each node_i in output_order, it should output tree[node_i]['output_type']
        # tree[node_i]['output_type'] is a list of 3 elements
        # index 0 - exists x_axis joint output
        # index 1 - exists y_axis joint output
        # index 2 - exists z_axis joint output

        # all the actuators need one output
        N, _ = adj_mat.shape
        # the joints on the root are not actuators
        # action_size = joint_num =
        # sum([v['joint_num'] for k, v in tree.items()]) - tree[0]['joint_num']
        action_size = sum([v['joint_num'] for k, v in tree.items()]) - tree[0]['joint_num']


    return output_order, output_type_dict, action_size


def _append_node_attr(tree, adj_mat, node_attr_list):
    ''' output:
            node_param: node parameters that can fully re-construct the
                        mujoco model structure
                        key: node_type
                        value: a numpy array of size
                               (num_of_node_type x number of features)
            param_size_dict:
                        number of features for each type of node
                        key: node_type
                        value: number of features of the node type in int
        NOTE: in nervenet we use a fixed number of features,
              and a fixed type of nodes
    '''
    N, _ = adj_mat.shape
    n_root_feature = 2
    n_joint_feature = 2 + 2 + 3 + 1  # see below for the definition details

    node_param = {'root': np.zeros(n_root_feature),
                  'joint': np.zeros((N - 1, n_joint_feature))}
    param_size_dict = {'root': n_root_feature, 'joint': n_joint_feature}

    # need to be consistant with the dfs order
    dfs_order = model_gen_util.dfs_order(adj_mat)
    node_order = [0] + [int(item.split('-')[-1]) for item in dfs_order]

    for i, node_id in enumerate(list(range(len(node_order)))):
        node_attr = node_attr_list[node_id]
        if i == 0:
            node_param['root'] = np.array([node_attr['a_size'], node_attr['b_size']])
        else:
            node_param['joint'][i - 1] = np.array(
                [node_attr['a_size'], node_attr['b_size'], node_attr['c_size'],
                 node_attr['u'], node_attr['v'], node_attr['axis_x'],
                 node_attr['axis_y'], node_attr['joint_range']]
            )

    return node_param, param_size_dict


if __name__ == '__main__':
    pass

    N = 5
    from config.config import get_config
    args = get_config(evolution=True)

    # get the
    adj_mat = model_gen.gen_test_adj_mat(task=args.task, shape=(N, N))
    node_attr = model_gen.gen_test_node_attr(task=args.task, node_num=N)
    if 'fish' in args.task:
        xml_struct = model_gen.fish_xml_generator(
            adj_mat, node_attr, options=None, filename='debug_gnn_param'
        )
    elif 'walker' in args.task:
        xml_struct = model_gen.walker_xml_generator(
            adj_mat, node_attr, options=None, filename='debug_gnn_param'
        )
    res = gen_gnn_param(args.task, adj_mat, node_attr)

    # create an environment based on the adj_mat and node_attr
    from env import test_model_gen
    import lxml.etree as etree
    xml_str = etree.tostring(xml_struct)
    import pdb; pdb.set_trace()
    test_model_gen.run_one_ep_given_model(
        args, adj_mat, xml_str, max_time_step=5
    )
