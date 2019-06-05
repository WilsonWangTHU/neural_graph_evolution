# -----------------------------------------------------------------------------
#   @brief:
#       util function for assigning running_mean and embedding for the agent
#       Note these functions should be called by the evolutionary agent!
#       written by Tingwu Wang
# -----------------------------------------------------------------------------

import init_path
import numpy as np
from env import model_gen
from env import model_perturb
from lxml import etree
from copy import deepcopy
from graph_util import graph_data_util

PATH = init_path.get_abs_base_dir()
WEIGHT_NAMES = ['policy_logstd']  # 'policy_0/w:0', 'policy_output'
INHERIT_LIST = [
    'policy_weights', 'baseline_weights', 'running_mean_info',
    'var_name', 'node_info', 'observation_size', 'action_size', 'lr'
]
GENERATED_LIST = \
    ['xml_str', 'adj_matrix', 'node_attr', 'debug_info', 'PrtID',
     'SpcID', 'node_info']
INVALID_LIST = [
    'stats', 'start_time', 'rollout_time', 'agent_id',
    'env_name', 'rank_info', 'LastRwd', 'AvgRwd',
]

ALL_LIST = INHERIT_LIST + GENERATED_LIST + INVALID_LIST
MUTATION_METHOD = ['add', 'delete', 'perturb']


def weight_s2s_update(p_species, c_species, node_order, is_nervenet):
    assert len(p_species['var_name']) == len(c_species['var_name'])

    for p_var_id, key in enumerate(deepcopy(p_species['var_name'])):
        assert key in c_species['var_name']
        c_var_id = c_species['var_name'].index(key)

        # check the shape of the weights
        c_shape = list(c_species['policy_weights'][c_var_id].shape)
        p_shape = list(p_species['policy_weights'][p_var_id].shape)

        if is_nervenet:
            # check if the var could have different shape
            if 'policy_logstd' in key:
                c_species = parse_policy_logstd(
                    c_species, p_species, c_var_id, p_var_id, node_order
                )
            elif 'embedding_HALF' in key:
                c_species = parse_embedding(
                    c_species, p_species, c_var_id, p_var_id, node_order
                )
            else:
                # the shape must match
                assert len(c_shape) == len(p_shape) and \
                    np.all([c_shape[i] == p_shape[i]
                            for i in range(len(c_shape))])
                c_species['policy_weights'][c_var_id] = \
                    p_species['policy_weights'][p_var_id]

        else:
            # if not nervenet, using fc model
            '''
            if len(c_shape) == len(p_shape) and \
                    np.all([c_shape[i] == p_shape[i]
                            for i in range(len(c_shape))]):
                # shape matched
                c_species['policy_weights'][c_var_id] = \
                    p_species['policy_weights'][p_var_id]
            '''
            pass  # ignore this weight

    return c_species


def running_stats_s2s_update(p_species, c_species, node_order, is_nervenet):
    '''
        @brief: for updating the running mean information, we only need to
            record
    '''
    c_obervation_size = c_species['observation_size']
    p_running_mean = p_species['running_mean_info']
    c_running_mean = {
        'mean': np.zeros([c_obervation_size]),
        'variance': np.zeros([c_obervation_size]),
        'step': p_running_mean['step'],
        'square_sum': np.zeros([c_obervation_size]),
        'sum': np.zeros([c_obervation_size])
    }
    running_mean_key = ['mean', 'square_sum', 'sum', 'variance']

    if is_nervenet:
        assigned_id = []
        for c_node, p_node in enumerate(node_order):
            if p_node < 0:  # if it is a new node, assign the average later
                continue

            c_pos = c_species['node_info']['input_dict'][c_node]
            p_pos = p_species['node_info']['input_dict'][p_node]

            for key in running_mean_key:
                c_running_mean[key][c_pos] = p_running_mean[key][p_pos]
                # from util import fpdb; fpdb.fpdb().set_trace()

            if c_node == 0:
                assert p_node == 0
            else:
                assigned_id.append(c_pos)

        for c_node, p_node in enumerate(node_order):
            if p_node >= 0:
                continue
            # if it is a new node, assign the average
            c_pos = c_species['node_info']['input_dict'][c_node]
            for key in running_mean_key:
                c_running_mean[key][c_pos] = np.mean(
                    [c_running_mean[key][p_pos] for p_pos in assigned_id],
                    axis=0
                )
    else:
        ob_size = min(c_species['observation_size'],
                      p_species['observation_size'])
        for key in running_mean_key:
            c_running_mean[key][:ob_size] = p_running_mean[key][:ob_size]

    c_species['running_mean_info'] = c_running_mean
    return c_species


def process_inherited_info(raw_species_info, current_species_format,
                           is_nervenet):
    if is_nervenet and 'debug_info' in raw_species_info and \
            raw_species_info['debug_info'] is not None and \
            'new_order' in raw_species_info['debug_info']:
        node_order = raw_species_info['debug_info']['new_order']

    elif is_nervenet:
        node_order = range(len(raw_species_info['adj_matrix']))

    else:
        node_order = None

    c_species = weight_s2s_update(
        raw_species_info, current_species_format, node_order, is_nervenet
    )
    c_species = running_stats_s2s_update(
        raw_species_info, current_species_format, node_order, is_nervenet
    )
    '''
    for key in raw_species_info['var_name']:
        c_var_id = c_species['var_name'].index(key)
        p_var_id = current_species_format['var_name'].index(key)

        print(key)
        print(c_species['policy_weights'][c_var_id] -
              raw_species_info['policy_weights'][p_var_id])
        # p_shape = list(p_species['policy_weights'][p_var_id].shape)
        # print(c_species['policy_weight'][key])
        # print(raw_species_info[key])
        from util import fpdb; fpdb.fpdb().set_trace()
    '''
    return c_species


def mutate_species(p_species, mutation_method, new_spc_struct=False):
    '''
        @brief:
            @mutation_method: ['add', 'delete', 'perturb']
    '''
    c_species = {}
    c_species['is_dead'] = False

    # inherit the following elements from parent
    for key in INHERIT_LIST:
        c_species[key] = p_species[key]
    c_species['adj_matrix'] = p_species['adj_matrix']
    c_species['node_attr'] = p_species['node_attr']

    # generate new adj_matrix and node_attr
    if new_spc_struct is False:
        c_adj_matrix, c_node_attr, debug_info = model_perturb.perturb_topology(
            c_species['adj_matrix'], c_species['node_attr'],
            evolve_option=MUTATION_METHOD.index(mutation_method),
            perturb_discrete=True
        )
        c_xml_str = etree.tostring(
            model_gen.fish_xml_generator(c_adj_matrix, c_node_attr,
                                         options=None),
            pretty_print=True
        )
    else:
        c_species['species'] = deepcopy(p_species['species'])
        debug_info = c_species['species'].mutate()
        c_adj_matrix, c_node_attr = c_species['species'].get_gene()
        c_xml_struct, c_xml_str = c_species['species'].get_xml()

    c_species['adj_matrix'] = c_adj_matrix
    c_species['node_attr'] = c_node_attr
    c_species['xml_str'] = c_xml_str

    c_species['debug_info'] = debug_info
    c_species['PrtID'] = p_species['SpcID']
    c_species['SpcID'] = -1
    c_species['node_info'] = p_species['node_info']

    # post process
    for key in GENERATED_LIST:
        assert key in c_species
    for key in INVALID_LIST:
        assert key not in c_species

    return c_species


def node_order_assertion(c_species, p_species):
    c_node_type_dict = c_species['node_info']['node_type_dict']
    p_node_type_dict = p_species['node_info']['node_type_dict']

    node_type_list = [node_type for node_type in c_node_type_dict]
    # from util import fpdb; fpdb.fpdb().set_trace()
    assert len(node_type_list) == 2 and \
        len(c_node_type_dict['root']) == 1 and c_node_type_dict['root'][0] == 0

    node_type_list = [node_type for node_type in p_node_type_dict]
    assert len(node_type_list) == 2 and \
        len(p_node_type_dict['root']) == 1 and p_node_type_dict['root'][0] == 0


def parse_policy_logstd(c_species, p_species, c_var_id, p_var_id, node_order):

    c_output_list = c_species['node_info']['output_type_dict']
    p_output_list = p_species['node_info']['output_type_dict']
    key = list(c_output_list.keys())[0]

    assert len(c_output_list) == 1 and len(p_output_list) == 1 \
        and key in p_output_list

    if 'fish3d' in c_species['env_name']:
        gnn_node_output_size = 3
    elif 'walker' in c_species['env_name'] or \
         'hopper' in c_species['env_name'] or \
         'cheetah' in c_species['env_name']:
        gnn_node_output_size = 1
    else:
        raise NotImplementedError

    assigned_id = []
    for c_node, p_node in enumerate(node_order):
        if p_node <= 0 or c_node == 0:  # new node or root node
            continue
        # output list: [5,3,2,1]

        c_pos = list(range(gnn_node_output_size * c_output_list[key].index(c_node),
                           gnn_node_output_size * c_output_list[key].index(c_node) +
                           gnn_node_output_size))
        p_pos = list(range(gnn_node_output_size * p_output_list[key].index(p_node),
                           gnn_node_output_size * p_output_list[key].index(p_node) +
                           gnn_node_output_size))

        # size: [1, num_output]
        c_species['policy_weights'][c_var_id][0, c_pos] = \
            p_species['policy_weights'][p_var_id][0, p_pos]
        assigned_id.append(c_pos)

    for c_node, p_node in enumerate(node_order):
        if p_node > 0 or c_node == 0:  # old node or root node
            continue

        # c_pos = list(range(3 * c_node, 3 * c_node + \
        # gnn_node_output_size))
        c_pos = list(range(gnn_node_output_size * c_output_list[key].index(c_node),
                           gnn_node_output_size * c_output_list[key].index(c_node) +
                           gnn_node_output_size))
        c_species['policy_weights'][c_var_id][0, c_pos] = np.mean(
            [c_species['policy_weights'][p_var_id][0, p_pos]
             for p_pos in assigned_id],
            axis=0
        )

    return c_species


def parse_embedding(c_species, p_species, c_var_id, p_var_id, node_order):
    # case one, noninput_shared # case two, noninput_separate
    _, c_graph_parameters, _, _, _, _, _, _, _ = \
        graph_data_util.construct_graph_input_feeddict(
            c_species['node_info'],
            np.zeros([1, c_species['node_info']['num_nodes'] * 100]),
            -1, -1, -1, -1, -1, -1, -1, request_data=['ob']
        )
    _, p_graph_parameters, _, _, _, _, _, _, _ = \
        graph_data_util.construct_graph_input_feeddict(
            p_species['node_info'],
            np.zeros([1, p_species['node_info']['num_nodes'] * 100]),
            -1, -1, -1, -1, -1, -1, -1, request_data=['ob']
        )

    # make sure that only root and joint is in the node type dict
    node_order_assertion(c_species, p_species)
    c_node_type_dict = c_species['node_info']['node_type_dict']
    p_node_type_dict = p_species['node_info']['node_type_dict']

    assigned_id = []
    for c_node, p_node in enumerate(node_order):
        if p_node < 0:  # a new joint node
            pass

        elif c_node == 0:  # the root node case
            assert p_node == 0
            c_pos = int(c_graph_parameters['root'][0, 0])
            p_pos = int(p_graph_parameters['root'][0, 0])

        else:  # c_node > 0. p_node > 0
            # shift 1 pos for the root node
            c_pos = int(
                c_graph_parameters['joint'][
                    c_node_type_dict['joint'].index(c_node), 0
                ]
            )
            # from util import fpdb; fpdb.fpdb().set_trace()
            p_pos = int(
                p_graph_parameters['joint'][
                    p_node_type_dict['joint'].index(p_node), 0
                ]
            )
        # from util import fpdb; fpdb.fpdb().set_trace()
        c_species['policy_weights'][c_var_id][c_pos, :] = \
            p_species['policy_weights'][p_var_id][p_pos, :]
        assigned_id.append(c_pos)

    for c_node, p_node in enumerate(node_order):
        if p_node < 0:  # a new node
            # shift 1 pos for the root node
            c_pos = int(
                c_graph_parameters['joint'][
                    c_node_type_dict['joint'].index(c_node), 0
                ]
            )
            # c_pos = c_graph_parameters['joint'][c_node - 1]

            c_species['policy_weights'][c_var_id][c_pos, :] = np.mean(
                [c_species['policy_weights'][p_var_id][p_pos, :]
                 for p_pos in assigned_id],
                axis=0
            )
        else:  # c_node > 0. p_node > 0
            pass
    return c_species
