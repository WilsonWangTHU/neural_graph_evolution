# ------------------------------------------------------------------------------
#   @brief:
#       The optimization agent is responsible for doing the updates.
#   @author:
#       modified from the code of kvfran, modified by Tingwu Wang
# ------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import init_path
from util import utils
from util import logger
# from util import model_saver
from util import parallel_util
# import os
from .agent import base_agent
from graph_util import graph_data_util
from graph_util import gen_gnn_param
from graph_util import gnn_util
from network import pruning_network
from env import model_gen
from lxml import etree


class pruning_agent(base_agent):

    def __init__(self, args, task_q, result_q,
                 name_scope='pruning_agent_policy'):
        super(pruning_agent, self).__init__(args, -1, -1, task_q, result_q,
                                            name_scope=name_scope)
        self.args = args
        self.base_path = init_path.get_abs_base_dir()
        self.task_q = task_q
        self.result_q = result_q
        self._name_scope = name_scope
        self.current_iteration = 0
        self._seed = 1234

        self.data_dict = {}  # species id: reward

    def run(self):
        '''
            @brief:
                this is the standard function to be called by the
                "multiprocessing.Process"

            @NOTE:
                check the parallel_util.py for definitions
        '''
        self.build_models()

        # load the model if needed
        if self.args.ckpt_name is not None:
            self.restore_all()

        # the main training process
        while True:
            next_task = self.task_q.get()

            # Kill the learner
            if next_task is None or next_task[0] == parallel_util.END_SIGNAL:
                self.task_q.task_done()
                break

            else:
                self.reset = next_task[2]
                species = next_task[1]
                filtered_species = self.update_and_filter(species)
                self.task_q.task_done()
                self.result_q.put(filtered_species)

    def update_and_filter(self, species):
        self.current_iteration += 1

        old_species = [i_species for i_species in species if i_species['SpcID'] != -1]
        new_species = [i_species for i_species in species if i_species['SpcID'] == -1]

        if self.reset:
            num_species_left = self.args.maximum_num_species - len(old_species)
            assert num_species_left >= 0
            old_species.extend(new_species[:num_species_left])

            # resetting the network
            self.set_policy(self.initial_weights)
            self.data_dict = {}
            return old_species

        # update the database
        for i_species in old_species:
            self.data_dict[i_species['SpcID']] = i_species

        # sample the data
        num_samples = min(len(self.data_dict), self.args.pruning_batch_size)
        key_list = [key for key in self.data_dict]
        sample_train_species = [
            key_list[i_id] for i_id in
            np.random.randint(0, len(self.data_dict), int(num_samples))
        ]

        # preprocess the last reward
        reward_list = np.array([self.data_dict[i_species_id]['LastRwd']
                               for i_species_id in self.data_dict])
        reward_mean = reward_list.mean()
        reward_std = reward_list.std()  # standardize to mean 0 stddev 1

        # fetch the gradient separately
        grad_list, loss_list = [], []
        for i_species_id in sample_train_species:
            feed_dict = self.get_feed_dict(self.data_dict[i_species_id])
            feed_dict[self._target_returns] = np.reshape(
                (self.data_dict[i_species_id]['LastRwd'] - reward_mean) /
                reward_std, [-1]
            )
            loss, grad, vpred = self.session.run(
                [self.vf_loss, self.grads, self.baseline_network._vpred],
                feed_dict=feed_dict
            )
            grad_list.append(grad)
            loss_list.append(loss)

        logger.info('Pruning network error: {}'.format(np.mean(loss_list)))

        # average the gradients
        var_grads = []
        for var_id in range(len(self.tvars)):
            var_grads.append(
                np.mean(
                    [i_species_grad[var_id] for i_species_grad in grad_list],
                    axis=0
                )
            )
        # apply the gradient
        grad_feed_dict = {}
        for i_id in range(len(self.tvars)):
            grad_feed_dict[self.gradient_placeholder[i_id]] = var_grads[i_id]
        self.session.run(self.update_op, feed_dict=grad_feed_dict)

        # score the species
        if self.args.bayesian_pruning:
            scores = []

            # construct the dropout mask
            test_dropout_mask = []
            for layer_shape in self.dropout_mask_shape:
                mask = np.concatenate(
                    [np.zeros([int(layer_shape / 2)]),
                     np.ones([int(layer_shape / 2)])]
                )
                np.random.shuffle(mask)
                test_dropout_mask.append(mask.reshape([1, -1]))

            for i_species in new_species:
                feed_dict = self.get_feed_dict(i_species)
                for i_layer in range(len(self.dropout_mask_shape)):
                    feed_dict[self.baseline_network.test_dropout_mask[i_layer]] \
                        = test_dropout_mask[i_layer]

                predicted_value = float(
                    np.reshape(
                        self.session.run(self.baseline_network.thompson_vpred,
                                         feed_dict=feed_dict), [-1]
                    )
                )
                # i.e. rank(scores/temperature + z),
                # where z = -\log(-\log(u)), u\sim Uniform(0,1)
                gumble_noise = -np.log(-np.log(np.random.uniform(0, 1, 1)))[0]
                predicted_value = gumble_noise + \
                    predicted_value / self.args.gumble_temperature
                scores.append(predicted_value)

            # pass thourgh a softmax
            '''
            current_temperature = self.args.start_temp + \
                (self.args.end_temp - self.args.start_temp) / \
                self.args.temp_iteration * \
                min(self.current_iteration, self.args.temp_iteration)
            scores = np.array(scores) / current_temperature
            scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
            '''

        else:
            scores = []
            for i_species in new_species:
                feed_dict = self.get_feed_dict(i_species)
                predicted_value = float(
                    np.reshape(
                        self.session.run(self._vpred, feed_dict=feed_dict), [-1]
                    )
                )
                gumble_noise = -np.log(-np.log(np.random.uniform(0, 1, 1)))[0]
                predicted_value = gumble_noise + \
                    predicted_value / self.args.gumble_temperature
                scores.append(predicted_value)

            '''
            # pass thourgh a softmax
            current_temperature = self.args.start_temp + \
                (self.args.end_temp - self.args.start_temp) / \
                self.args.temp_iteration * \
                min(self.current_iteration, self.args.temp_iteration)
            scores = np.array(scores) / current_temperature
            scores = np.exp(scores) / np.sum(np.exp(scores), axis=0)
            '''

        # filter the species
        num_species_left = self.args.maximum_num_species - len(old_species)
        assert num_species_left >= 0
        '''
        if self.args.softmax_pruning:
            new_species = np.random.choice(new_species, num_species_left,
                                           replace=False, p=scores)
        else:
            # rank the species
            score_order = np.argsort(scores)[::-1]
            new_species = new_species[score_order][:num_species_left]
        '''
        score_order = np.argsort(scores)[::-1]
        new_species = [new_species[i_order]
                       for i_order in score_order[:num_species_left]]

        old_species.extend(new_species)
        return old_species

    def build_models(self):
        '''
            @brief:
                this is the function where the rollout agents and optimization
                agent build their networks, set up the placeholders, and gather
                the variable list.
        '''
        # make sure that the agent has a session
        self.build_session()

        # the baseline function to reduce the variance
        self.build_baseline_network()

        # init the network parameters (xavier initializer)
        self.session.run(tf.global_variables_initializer())

        # prepare the feed_dict info for the ppo minibatches
        self._receive_idx, self._send_idx, self._node_type_idx, \
            self._inverse_node_type_idx, self._batch_size_int = \
            self.baseline_network.get_gnn_idx_placeholder()
        self._num_nodes_ph = self.baseline_network._num_nodes_placeholder

        self._input_parameters = \
            self.baseline_network.get_input_parameters_placeholder()
        self._target_returns = \
            self.baseline_network.get_target_return_placeholder()
        self._vpred = self.baseline_network.get_vpred_placeholder()

        # prepared the init kl divergence if needed
        self.current_kl_lambda = 1

        self.policy_var_list, self.all_policy_var_list = \
            self.baseline_network.get_var_list()

        self.get_policy = \
            utils.GetPolicyWeights(self.session, self.policy_var_list)
        self.set_policy = \
            utils.SetPolicyWeights(self.session, self.policy_var_list)
        self.initial_weights = self.get_policy()

    def build_baseline_network(self):
        '''
            @brief:
                Build the baseline network, and fetch the baseline variable list
        '''

        adj_matrix, node_attr, xml_str = model_gen.get_initial_settings()
        xml_str = etree.tostring(xml_str, pretty_print=True)

        self._node_info = gen_gnn_param.gen_gnn_param(
            self.args.task_name,
            adj_matrix,
            node_attr,
            gnn_node_option=self.args.gnn_node_option,
            root_connection_option=self.args.root_connection_option,
            gnn_output_option=self.args.gnn_output_option,
            gnn_embedding_option='parameter'
        )
        self._node_info = gnn_util.construct_ob_size_dict(self._node_info, 64)
        self._node_info = \
            gnn_util.get_inverse_type_offset(self._node_info, 'node')
        self._node_info = \
            gnn_util.get_inverse_type_offset(self._node_info, 'output')
        self._node_info = gnn_util.get_receive_send_idx(self._node_info)
        self.init_node_info = self._node_info

        self.baseline_network = pruning_network.GGNN(
            args=self.args,
            session=self.session,
            name_scope=self.name_scope + '_baseline',
            initial_node_info=self.init_node_info,
            bayesian_op=self.args.bayesian_pruning
        )
        self.dropout_mask_shape = self.baseline_network.dropout_mask_shape
        self.dropout_mask_placeholder = self.baseline_network.test_dropout_mask

        # step 2: get the placeholders for the network
        # self.target_return_placeholder = \
        # self.baseline_network.get_target_return_placeholder()
        self.vf_loss = self.baseline_network.get_vf_loss()
        self.tvars = self.baseline_network._trainable_var_list

        # define the loss and gradient
        self.optimizer = tf.train.AdamOptimizer(self.args.lr)
        self.grads = tf.gradients(self.vf_loss, self.tvars)
        self.gradient_placeholder = []
        for i_id in range(len(self.tvars)):
            self.gradient_placeholder.append(
                tf.placeholder(tf.float32, shape=self.tvars[i_id].get_shape())
            )

        self.update_op = self.optimizer.apply_gradients(
            zip(self.gradient_placeholder, self.tvars)
        )

    def update_ppo_parameters(self, paths):
        '''
            @brief: update the ppo
        '''
        pass

    def save_all(self, current_iteration, ob_normalizer_info={}):
        '''
            @brief: save the model into several npy file.
        '''
        pass

    def restore_all(self):
        '''
            @brief: restore the parameters
        '''
        pass

    def get_output_path(self, save=True):
        pass

    def get_feed_dict(self, new_species):
        adj_matrix, node_attr, xml_str = new_species['adj_matrix'], \
            new_species['node_attr'], new_species['xml_str']

        node_info = gen_gnn_param.gen_gnn_param(
            self.args.task_name,
            adj_matrix,
            node_attr,
            gnn_node_option=self.args.gnn_node_option,
            root_connection_option=self.args.root_connection_option,
            gnn_output_option=self.args.gnn_output_option,
            gnn_embedding_option='parameter'
        )
        node_info = gnn_util.construct_ob_size_dict(node_info, 64)
        node_info = gnn_util.get_inverse_type_offset(node_info, 'node')
        node_info = gnn_util.get_inverse_type_offset(node_info, 'output')
        node_info = gnn_util.get_receive_send_idx(node_info)

        dummy_obs = np.zeros([1, 6 * node_info['num_nodes'] + 6])
        _, graph_parameters, receive_idx, send_idx, \
            node_type_idx, inverse_node_type_idx, _, _, _ = \
            graph_data_util.construct_graph_input_feeddict(
                node_info, dummy_obs, -1, -1, -1, -1, -1, -1, -1
            )

        feed_dict = {
            self._receive_idx: receive_idx,
            # self._send_idx: send_idx,
            # self._node_type_idx: node_type_idx,
            self._inverse_node_type_idx: inverse_node_type_idx,
            self._batch_size_int: 1,
            # self._input_parameters: graph_parameters,
            # self._target_returns: self.data_dict[i_species_id]['LastRwd']
        }
        for i_edge in node_info['edge_type_list']:
            feed_dict[self._send_idx[i_edge]] = send_idx[i_edge]

        # append the node type idx
        for i_node_type in node_info['node_type_dict']:
            feed_dict[self._node_type_idx[i_node_type]] = \
                node_type_idx[i_node_type]

        for i_node_type in node_info['node_type_dict']:
            feed_dict[self._input_parameters[i_node_type]] = \
                graph_parameters[i_node_type]

        feed_dict[self._num_nodes_ph] = adj_matrix.shape[0]
        return feed_dict

    def add_species_copy(self, species):
        self.data_dict[species['SpcID']] = species
