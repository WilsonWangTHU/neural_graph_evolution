#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       Some helper functions to prepare the input
#   @author:
#       Tingwu Wang, April 10th, 2018
# -----------------------------------------------------------------------------


import init_path
import numpy as np
from util import logger
from util import utils
from graph_util import nervenetplus_util
from graph_util import graph_data_util


_BASE_PATH = init_path.get_abs_base_dir()


def prepare_network_feeddict(data_dict, is_nervenet, baseline_network,
                             node_info, current_idx_dict, nervenetplus=False,
                             gnn_num_prop_steps=-1):
    # feed_dict = {}

    # step 1: For ggnn and fc policy, we have different obs in feed_dict
    observations = np.concatenate([path["obs"] for path in data_dict])
    feed_dict, nervenetplus_batch_pos = prepare_policy_network_feeddict(
        observations, is_nervenet, node_info, current_idx_dict,
        data_dict, nervenetplus, gnn_num_prop_steps
    )

    # step 2: prepare the advantage function, old action / obs needed for
    # trpo / ppo updates
    advant_n = generate_advantage(data_dict, baseline_network)
    action_dist_mu = np.concatenate(
        [path["action_dists_mu"] for path in data_dict]
    )
    action_dist_logstd = np.concatenate(
        [path["action_dists_logstd"] for path in data_dict]
    )
    action_n = np.concatenate([path["actions"] for path in data_dict])
    feed_dict.update(
        {'action_placeholder': action_n,
         'advantage_placeholder': advant_n,
         'oldaction_dist_mu_placeholder': action_dist_mu,
         'oldaction_dist_logstd_placeholder': action_dist_logstd,
         'batch_size_float_placeholder': np.array(float(len(observations)))}
    )

    # step 3: feed_dict to update value function
    target_return = \
        np.concatenate([path["target_return"] for path in data_dict])
    feed_dict.update(
        {'target_return_placeholder': target_return}
    )

    return feed_dict, nervenetplus_batch_pos


def prepare_policy_network_feeddict(observations, is_nervenet,
                                    node_info, current_idx_dict,
                                    rollout_data, nervenetplus,
                                    gnn_num_prop_steps):
    '''
        @brief: prepare the feed dict for the policy network part
    '''
    nervenetplus_batch_pos = None

    if not is_nervenet:
        # it is the most easy case, nice and easy
        return {'obs_placeholder': observations}, -1

    # construct the graph input feed dict
    # in this case, we need to get the receive_idx, send_idx,
    # node_idx, inverse_node_idx ready. These index will be helpful
    # to telling the network how to pass and update the information
    if not nervenetplus:
        graph_obs, graph_parameters, \
            receive_idx, send_idx, node_type_idx, inverse_node_type_idx, \
            output_type_idx, inverse_output_type_idx, last_batch_size = \
            graph_data_util.construct_graph_input_feeddict(
                node_info,
                observations,
                current_idx_dict['receive_idx'],
                current_idx_dict['send_idx'],
                current_idx_dict['node_type_idx'],
                current_idx_dict['inverse_node_type_idx'],
                current_idx_dict['output_type_idx'],
                current_idx_dict['inverse_output_type_idx'],
                current_idx_dict['last_batch_size']
            )
        # from util import fpdb; fpdb.fpdb().set_trace()
    else:
        assert rollout_data is not None
        # preprocess the episodic information

        graph_obs, graph_parameters, _, _, _, _, _, _, _ = \
            graph_data_util.construct_graph_input_feeddict(
                node_info, observations,
                -1, -1, -1, -1, -1, -1, -1,
                request_data=['ob'])
        nervenetplus_batch_pos, total_size = \
            nervenetplus_util.nervenetplus_step_assign(
                rollout_data, gnn_num_prop_steps
            )
        _, _, receive_idx, send_idx, \
            node_type_idx, inverse_node_type_idx, \
            output_type_idx, inverse_output_type_idx, \
            last_batch_size = \
            graph_data_util.construct_graph_input_feeddict(
                node_info,
                np.empty([int(total_size / gnn_num_prop_steps)]),
                current_idx_dict['receive_idx'],
                current_idx_dict['send_idx'],
                current_idx_dict['node_type_idx'],
                current_idx_dict['inverse_node_type_idx'],
                current_idx_dict['output_type_idx'],
                current_idx_dict['inverse_output_type_idx'],
                current_idx_dict['last_batch_size'],
                request_data=['idx']
            )

    return {
        'batch_size_int_placeholder': int(last_batch_size),
        'raw_obs_placeholder': observations,

        'last_batch_size': last_batch_size,
        'receive_idx': receive_idx,
        'send_idx': send_idx,

        'node_type_idx': node_type_idx,
        'inverse_node_type_idx': inverse_node_type_idx,

        'output_type_idx': output_type_idx,
        'inverse_output_type_idx': inverse_output_type_idx,

        'graph_obs': graph_obs,
        'graph_parameters': graph_parameters,
    }, nervenetplus_batch_pos


def generate_advantage(data_dict, baseline_network):
    '''
        @brief: calculate the parameters for the advantage function
    '''

    for path in data_dict:
        # the predicted value function (baseline function)
        path["baseline"] = baseline_network.predict(path)

    advantage_method = baseline_network.args.advantage_method
    gamma = baseline_network.args.gamma
    gae_lam = baseline_network.args.gae_lam

    # esitmate the advantages
    if advantage_method == 'raw':
        for path in data_dict:
            # the gamma discounted rollout value function
            path["returns"] = utils.discount(path["rewards"], gamma)
            path["advantage"] = path["returns"] - path["baseline"]
            path['target_return'] = path['returns']
    else:
        assert advantage_method == 'gae', logger.error(
            'invalid advantage estimation method: {}'.format(advantage_method)
        )

        for path in data_dict:
            # the gamma discounted rollout value function
            path["returns"] = utils.discount(path["rewards"], gamma)

            # init the advantage
            path["advantage"] = np.zeros(path['returns'].shape)

            num_steps = len(path['returns'])

            # generate the GAE advantage
            for i_step in reversed(range(num_steps)):
                if i_step < num_steps - 1:
                    delta = path['rewards'][i_step] \
                        + gamma * path['baseline'][i_step + 1] \
                        - path['baseline'][i_step]

                    path['advantage'][i_step] = \
                        delta + gamma * gae_lam * path['advantage'][i_step + 1]
                else:
                    delta = path['rewards'][i_step] - path['baseline'][i_step]
                    path['advantage'][i_step] = delta

            path['target_return'] = path['advantage'] + path['baseline']

    # standardized advantage function
    advant_n = np.concatenate([path["advantage"] for path in data_dict])
    advant_n -= advant_n.mean()
    advant_n /= (advant_n.std() + 1e-8)  # standardize to mean 0 stddev 1
    return advant_n
