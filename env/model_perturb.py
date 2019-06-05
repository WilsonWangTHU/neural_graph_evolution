'''
    function to perturb a mujoco model given the 'genes' (adj_matrix and node_attributes)
    the function was originally implemented in model_gen.py
    create a new directory for better structure
'''

from __future__ import division
import init_path
import pdb
import numpy as np
import random
import copy

# local import
from util import model_gen_util
from env import model_gen
from env import species_info

# fixed ratio for the geometry
ELLIPSOID_X_RATIO = 1
ELLIPSOID_Y_RATIO = 8
ELLIPSOID_Z_RATIO = 4
CYLINDER_R_RATIO = 1
CYLINDER_H_RATIO = 6

RAD_TO_DEG = 57.29577951308232
BASE_PATH = init_path.get_abs_base_dir()


def perturb_topology(adj_matrix, node_attr, evolve_option,
                     perturb_geom=False, perturb_discrete=True):

    def perturb_graph(adj_matrix, node_attr_list, perturb_discrete=True,
                      add_node=True):
        '''
            brief: we allow
                        1. 'add_node' for every node
                        2. 'remove_node' for leaf node
                        3. change connection types

            NOTE: some basic stuff
                        1. all nodes in the graph should be connected
                        2. cannot have loop
        '''
        def verify_graph(adj_matrix):
            '''
                check whether the graph is a valid tree for generating the xml
                model
            '''
            N, _ = adj_matrix.shape

            # node cannot connect to itself
            for i in range(N):
                if adj_matrix[i, i] > 0:
                    pdb.set_trace()
                    return False

            # the matrix must be diagnol-symmetric
            for i in range(N):
                for j in range(i, N):
                    if adj_matrix[i, j] != adj_matrix[j, i]:
                        pdb.set_trace()
                        return False

            # each node must have eactly one parent (except for root)
            for i in range(1, N):
                col_i = adj_matrix[:i, i]  # only consider upper triangle of the matrix
                parent_connection = np.where(col_i > 0)[0]
                if len(parent_connection) > 1:
                    pdb.set_trace()
                    return False
            return True

        def remove_one_leaf(adj_matrix, node_attr_list):
            N, _ = adj_matrix.shape
            debug_info = {}
            if N <= 2:
                debug_info['op'] = 'none'
                # need to preserve at least one structure
                return adj_matrix, node_attr_list, debug_info

            debug_info['old_mat'] = np.copy(adj_matrix)

            # randomly sample a leaf in the tree
            leaf_list = model_gen_util.leaf_list(adj_matrix)
            node_id = random.sample(leaf_list, 1)[0]
            debug_info['delete_node'] = node_id

            # logging for debug info
            edge_dfs = model_gen_util.dfs_order(adj_matrix)
            dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
            deleted_node_order = dfs_order.index(node_id)
            debug_info['old_order'] = dfs_order
            new_order = copy.deepcopy(dfs_order)
            new_order.pop(deleted_node_order)
            debug_info['new_order'] = new_order

            debug_info['old_nodes'] = list(range(N))
            debug_info['new_nodes'] = copy.deepcopy(debug_info['old_nodes'])
            debug_info['new_nodes'].remove(node_id)

            # remove the node from the graph
            adj_matrix = np.delete(adj_matrix, node_id, 0)
            adj_matrix = np.delete(adj_matrix, node_id, 1)
            # remove the corresponding node information from the attr list
            node_attr_list = \
                node_attr_list[:node_id] + node_attr_list[node_id + 1:]

            return adj_matrix, node_attr_list, debug_info

        def add_one_node(adj_matrix, node_attr_list, perturb_discrete=True):
            N, _ = adj_matrix.shape

            # randomly sample a parent
            parent_node = int(np.random.randint(N, size=1))
            debug_info = {}
            debug_info['old_mat'] = np.copy(adj_matrix)
            debug_info['parent'] = parent_node
            # randomly sample a connection type
            connection_type = 7

            new_col = np.zeros(N)
            new_col[parent_node] = connection_type
            new_col = new_col.reshape(-1, 1)
            new_row = np.zeros(N + 1)
            new_row[parent_node] = connection_type

            # adding debug information
            # keep track of how the graph perturbation is changed
            edge_dfs = model_gen_util.dfs_order(adj_matrix)
            dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
            child_list = \
                model_gen_util.get_child_given_parent(adj_matrix, parent_node)
            debug_info['old_order'] = dfs_order
            new_order = copy.deepcopy(dfs_order)
            if len(child_list) == 0:
                insert_idx = dfs_order.index(parent_node) + 1
            else:
                insert_idx = dfs_order.index(child_list[-1]) + 1
            new_order.insert(insert_idx, -1)
            debug_info['new_order'] = new_order

            debug_info['old_nodes'] = list(range(N))
            debug_info['new_nodes'] = copy.deepcopy(debug_info['old_nodes'])
            debug_info['new_nodes'].append(-1)

            # add the new node to the graph
            adj_matrix = np.hstack((adj_matrix, new_col))
            adj_matrix = np.vstack((adj_matrix, new_row))
            # sample a new set of attributes and add it to list
            new_node_attr = model_gen.gen_test_node_attr(
                node_num=2, discrete_rv=perturb_discrete
            )[1]
            node_attr_list.append(new_node_attr)

            return adj_matrix, node_attr_list, debug_info

        def change_connection_type(adj_matrix):
            '''
            '''
            N, _ = adj_matrix.shape

            for i in range(N):
                for j in range(i, N):
                    if adj_matrix[i, j] == 0:
                        continue
                    # possibly perturb the connection
                    # right now, set everything to 7 (socket)
                    adj_matrix[i, j] = 7

            return adj_matrix

        # ## codes to perturb the graph ###
        if add_node:
            adj_mat, node_attr, debug_info = add_one_node(
                adj_matrix, node_attr_list, perturb_discrete=perturb_discrete
            )
            # TODO: process debug_info
        else:
            adj_mat, node_attr, debug_info = remove_one_leaf(
                adj_matrix, node_attr_list
            )
            # TODO: process debug_info

        # for now, we are not changing the connection type, everything is socket-connected
        # adj_matrix = change_connection_type(adj_matrix)
        assert verify_graph(adj_mat)

        return adj_mat, node_attr, debug_info

    def perturb_local(node_attr, perturb_geom=False, discrete=True):
            ''' perform local perturbation according to current attributes
            '''
            new_attr = copy.deepcopy(node_attr)
            # u, v for specifying position relative to parents
            new_attr['u'] = node_attr['u'] + \
                model_gen_util.gaussian_noise(0, np.pi / 6,
                                              discrete, np.pi / 2.0)
            new_attr['v'] = node_attr['v'] + \
                model_gen_util.gaussian_noise(0, np.pi / 6,
                                              discrete, np.pi / 2.0)
            new_attr['u'] = \
                float(np.clip(new_attr['u'], 1e-8, 2 * np.pi - 1e-8))
            new_attr['v'] = \
                float(np.clip(new_attr['v'], 1e-8, np.pi - 1e-8))

            # x, y coordinate for child's relative frame
            new_attr['axis_x'] = node_attr['axis_x'] + \
                model_gen_util.gaussian_noise(0, 0.1, discrete, 0.2)
            new_attr['axis_y'] = node_attr['axis_y'] + \
                model_gen_util.gaussian_noise(0, 0.1, discrete, 0.2)
            new_attr['axis_x'] = \
                float(np.clip(new_attr['axis_x'], 0.2, 1.0 - 1e-8))
            new_attr['axis_y'] = \
                float(np.clip(new_attr['axis_y'], 0.2, 1.0 - 1e-8))

            # a, b, c child size
            new_attr['a_size'] = node_attr['a_size'] + \
                model_gen_util.gaussian_noise(0, 0.002, discrete, 0.002)
            new_attr['b_size'] = node_attr['b_size'] + \
                model_gen_util.gaussian_noise(0, 0.002, discrete, 0.002)

            new_attr['c_size'] = node_attr['c_size'] + \
                model_gen_util.gaussian_noise(0, 0.002, discrete, 0.002)
            new_attr['a_size'] = float(np.clip(new_attr['a_size'], 0.002, 0.03))
            new_attr['b_size'] = float(np.clip(new_attr['b_size'], 0.002, 0.03))
            new_attr['c_size'] = float(np.clip(new_attr['c_size'], 0.002, 0.03))

            # joint range
            new_attr['joint_range'] = new_attr['joint_range'] + \
                model_gen_util.gaussian_noise(0, 10, discrete, 10)
            new_attr['joint_range'] = \
                int(np.clip(new_attr['joint_range'], 30, 90))

            return new_attr

    def perturb_attr(node_attr, perturb_geom=False, perturb_discrete=True):
        '''
            brief: perturb the 7D parameters

            NOTE: some basic stuff
                        1. consider whether the perturbation physically makes sense
                            (the perturbation will always makes sense
                            if we have reasonable upper and lower bounds)
        '''

        N = len(node_attr)

        # sample the number of nodes to perturb attribute
        num_perturb = random.randint(1, N - 1)
        node_list = list(
            np.random.choice(list(range(1, N)), num_perturb, replace=False)
        )
        for random_node in node_list:
            # independently sample a new set of attributes
            new_attr = perturb_local(node_attr[random_node],
                                     discrete=perturb_discrete)
            # replace the current attributes to the latest
            node_attr[random_node] = new_attr
        return node_attr

    # equal probability of choices:
    # 0. adding one node 1. removing one node 2. perturnbing one existing ndoe
    # evolve_option = random.randint(0, 2)
    debug_info = {}
    if evolve_option == 0:
        adj_matrix, node_attr, debug_info = \
            perturb_graph(adj_matrix, node_attr, perturb_discrete,
                          add_node=True)
        debug_info['op'] = 'add_node'

    elif evolve_option == 1:
        adj_matrix, node_attr, debug_info = \
            perturb_graph(adj_matrix, node_attr, perturb_discrete,
                          add_node=False)
        if 'op' not in debug_info:
            debug_info['op'] = 'rm_node'

    elif evolve_option == 2:
        node_attr = perturb_attr(node_attr, perturb_discrete=perturb_discrete)
        debug_info['op'] = 'change_attr'

    return adj_matrix.astype('int'), node_attr, debug_info



def perturb_one_local(node_attr, task='fish', perturb_geom=False, discrete=True):
        ''' perform local perturbation according to current attributes
        '''
        new_attr = copy.deepcopy(node_attr)

        if 'fish' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['fish']
        elif 'walker' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['walker']
        elif 'cheetah' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['cheetah']
        elif 'hopper' in task:
            constraint = species_info.CREATURE_HARD_CONSTRAINT['hopper']

        attr_list = ['u', 'v', 'axis_x', 'axis_y',
                     'a_size', 'b_size', 'c_size', 'joint_range']

        for attr in attr_list:
            low, high = constraint[attr]
            step_size = (high - low) / 6 # ideally this should be a hyperparameter to tune

            new_attr[attr] = node_attr[attr] + \
                model_gen_util.gaussian_noise(0, step_size/2,
                                              discrete, step_size)
            new_attr[attr] = \
                float(np.clip(new_attr[attr], low, high))
            if 'joint_range' == attr:
                new_attr[attr] = int(new_attr[attr])

        return new_attr

def perturb_one_attr(node_attr, task='fish', perturb_geom=False, perturb_discrete=True):
    '''
        brief: perturb the 7D parameters

        NOTE: some basic stuff
                    1. consider whether the perturbation physically makes sense
                        (the perturbation will always makes sense
                        if we have reasonable upper and lower bounds)
    '''

    new_attr = perturb_one_local(node_attr, task=task,
        discrete=perturb_discrete
    )
    return new_attr
