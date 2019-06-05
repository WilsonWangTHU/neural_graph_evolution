# -----------------------------------------------------------------------------
#   @brief:
#       In this function, we define the base agent.
#       The base agent should be responsible for building the policy network,
#       fetch the io placeholders / tensors, and set up the variable list
#   @author:
#       code originally from kvfran, modified by Tingwu Wang
# -----------------------------------------------------------------------------
import tensorflow as tf
import init_path
import multiprocessing
from network import policy_network
from network import gated_graph_policy_network
from network import nervenet_policy
from network import nervenetplus_policy
from network import treenet_policy
from util import logger
from graph_util import graph_data_util
from graph_util import nervenetplus_util
from util import utils
import numpy as np


class base_agent(multiprocessing.Process):

    def __init__(self,
                 args,
                 observation_size,
                 action_size,
                 task_q,
                 result_q,
                 name_scope='trpo_agent',
                 is_rollout_agent=False):

        # the multiprocessing initialization
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.gnn_placeholder_list = None
        self.obs_placeholder = None

        # the configurations for the agent
        self.args = args

        # the network parameters
        self.name_scope = name_scope
        self.observation_size = observation_size
        self.action_size = action_size
        self.is_rollout_agent = is_rollout_agent

        # the variables and networks to be used, init them before use them
        self.policy_network = None
        self.step_policy_network = None
        self.policy_var_list = None

        self.tf_var_list = None
        self.iteration = None

        # the gnn parameters
        if self.args.use_gnn_as_policy:
            self.gnn_parameter_initialization()

        self.base_path = init_path.get_base_dir()

    def build_session(self):
        if self.args.use_gpu:
            config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
        config.gpu_options.allow_growth = True  # don't take full gpu memory
        self.session = tf.Session(config=config)

    def fetch_policy_info(self):
        assert self.policy_network is not None, \
            logger.error('Init the policy network before using it')

        # input placeholders to the policy networks
        if self.args.use_gnn_as_policy:
            # index placeholders
            self.receive_idx_placeholder, self.send_idx_placeholder, \
                self.node_type_idx_placeholder, \
                self.inverse_node_type_idx_placeholder, \
                self.output_type_idx_placeholder, \
                self.inverse_output_type_idx_placeholder, \
                self.batch_size_int_placeholder = \
                self.policy_network.get_gnn_idx_placeholder()

            # the graph_obs placeholders and graph_parameters_placeholders
            self.graph_obs_placeholder = \
                self.policy_network.get_input_obs_placeholder()
            self.graph_parameters_placeholder = \
                self.policy_network.get_input_parameters_placeholder()

            self.gnn_placeholder_list = [
                self.receive_idx_placeholder,
                self.send_idx_placeholder,
                self.node_type_idx_placeholder,
                self.inverse_node_type_idx_placeholder,
                self.output_type_idx_placeholder,
                self.inverse_output_type_idx_placeholder,
                self.batch_size_int_placeholder,
                self.graph_obs_placeholder,
                self.graph_parameters_placeholder
            ]
            if self.args.nervenetplus:

                self.step_receive_idx_placeholder, \
                    self.step_send_idx_placeholder, \
                    self.step_node_type_idx_placeholder, \
                    self.step_inverse_node_type_idx_placeholder, \
                    self.step_output_type_idx_placeholder, \
                    self.step_inverse_output_type_idx_placeholder, \
                    self.step_batch_size_int_placeholder = \
                    self.step_policy_network.get_gnn_idx_placeholder()

                # the graph_obs placeholders and graph_parameters_placeholders
                self.step_graph_obs_placeholder = \
                    self.step_policy_network.get_input_obs_placeholder()
                self.step_graph_parameters_placeholder = \
                    self.step_policy_network.get_input_parameters_placeholder()
        else:
            self.obs_placeholder = self.policy_network.get_input_placeholder()

        self.raw_obs_placeholder = None

        # output from the policy networks means for each action
        self.action_dist_mu = self.policy_network.get_action_dist_mu()

        # log std parameters of actions (all the same)
        self.action_dist_logstd_param = \
            self.policy_network.get_action_dist_logstd_param()
        self.action_dist_logstd = self.action_dist_logstd_param

        if self.args.nervenetplus:
            self.step_action_dist_mu = \
                self.step_policy_network.get_action_dist_mu()

            # log std parameters of actions (all the same)
            self.step_action_dist_logstd_param = \
                self.step_policy_network.get_action_dist_logstd_param()
            self.step_action_dist_logstd = self.step_action_dist_logstd_param
        # "policy_var_list": to be passed to the rollout agents
        # "all_policy_var_list": to be saved into the checkpoint
        self.policy_var_list, self.all_policy_var_list = \
            self.policy_network.get_var_list()
        self.iteration = self.policy_network.get_iteration_var()
        self.iteration_add_op = self.iteration.assign_add(1)

    def build_policy_network(self, adj_matrix=None, node_attr=None):
        if self.args.nervenetplus:
            assert self.args.use_gnn_as_policy and self.args.use_nervenet

        if self.args.use_gnn_as_policy:
            if self.args.use_nervenet:
                if self.args.nervenetplus:
                    if self.args.tree_net:
                        self.policy_network = treenet_policy.nervenet(
                            session=self.session,
                            name_scope=self.name_scope + '_policy',
                            input_size=self.observation_size,
                            output_size=self.action_size,
                            adj_matrix=adj_matrix,
                            node_attr=node_attr,
                            args=self.args,
                            is_rollout_agent=self.is_rollout_agent
                        )
                        self.step_policy_network = treenet_policy.nervenet(
                            session=self.session,
                            name_scope=self.name_scope + 'step_policy',
                            input_size=self.observation_size,
                            output_size=self.action_size,
                            adj_matrix=self.adj_matrix,
                            node_attr=self.node_attr,
                            args=self.args,
                            is_rollout_agent=True
                        )
                        self.node_info = self.policy_network.get_node_info()

                        self.step_policy_var_list, _ = \
                            self.step_policy_network.get_var_list()

                        self.set_step_policy = \
                            utils.SetPolicyWeights(self.session,
                                                   self.step_policy_var_list)
                    else:
                        self.policy_network = nervenetplus_policy.nervenet(
                            session=self.session,
                            name_scope=self.name_scope + '_policy',
                            input_size=self.observation_size,
                            output_size=self.action_size,
                            adj_matrix=adj_matrix,
                            node_attr=node_attr,
                            args=self.args,
                            is_rollout_agent=self.is_rollout_agent
                        )
                        self.step_policy_network = nervenetplus_policy.nervenet(
                            session=self.session,
                            name_scope=self.name_scope + 'step_policy',
                            input_size=self.observation_size,
                            output_size=self.action_size,
                            adj_matrix=self.adj_matrix,
                            node_attr=self.node_attr,
                            args=self.args,
                            is_rollout_agent=True
                        )
                        self.node_info = self.policy_network.get_node_info()

                        self.step_policy_var_list, _ = \
                            self.step_policy_network.get_var_list()

                        self.set_step_policy = \
                            utils.SetPolicyWeights(self.session,
                                                   self.step_policy_var_list)
                else:
                    self.policy_network = nervenet_policy.nervenet(
                        session=self.session,
                        name_scope=self.name_scope + '_policy',
                        input_size=self.observation_size,
                        output_size=self.action_size,
                        adj_matrix=adj_matrix,
                        node_attr=node_attr,
                        args=self.args
                    )
            else:
                self.policy_network = gated_graph_policy_network.GGNN(
                    session=self.session,
                    name_scope=self.name_scope + '_policy',
                    input_size=self.observation_size,
                    output_size=self.action_size,
                    ob_placeholder=None,
                    trainable=True,
                    build_network_now=True,
                    is_baseline=False,
                    placeholder_list=None,
                    args=self.args
                )
            self.raw_obs_placeholder = None
            self.node_info = self.policy_network.get_node_info()
        else:
            self.policy_network = policy_network.policy_network(
                session=self.session,
                name_scope=self.name_scope + '_policy',
                input_size=self.observation_size,
                output_size=self.action_size,
                ob_placeholder=None,
                trainable=True,
                build_network_now=True,
                define_std=True,
                is_baseline=False,
                args=self.args
            )

        # if use the nervenetplus model
        # if self.args.nervenetplus:
        #     # build the action model
        #     with tf.variable_scope('', reuse=True):
        #         self.step_policy_network = nervenetplus_policy.nervenet(
        #             session=self.session,
        #             name_scope=self.name_scope + '_policy',
        #             input_size=self.observation_size,
        #             output_size=self.action_size,
        #             adj_matrix=self.adj_matrix,
        #             node_attr=self.node_attr,
        #             args=self.args,
        #             is_rollout_agent=True
        #         )
        #     self.node_info = self.policy_network.get_node_info()
        self.fetch_policy_info()

    def gnn_parameter_initialization(self):
        '''
            @brief:
                the parameters for the gnn, see the gated_graph_network_policy
                file for details what these variables mean.
        '''
        self.receive_idx = None
        self.send_idx = None
        self.node_type_idx = None
        self.inverse_node_type_idx = None
        self.output_type_idx = None
        self.inverse_output_type_idx = None
        self.last_batch_size = -1
        self.nervenetplus_batch_pos = None

    def prepared_policy_network_feeddict(self, obs_n, rollout_data=None,
                                         step_model=False):
        '''
            @brief: prepare the feed dict for the policy network part
        '''
        nervenetplus_batch_pos = None

        if self.args.use_gnn_as_policy:

            if not self.args.nervenetplus or obs_n.shape[0] == 1:
                graph_obs, graph_parameters, \
                    self.receive_idx, self.send_idx, \
                    self.node_type_idx, self.inverse_node_type_idx, \
                    self.output_type_idx, self.inverse_output_type_idx, \
                    self.last_batch_size = \
                    graph_data_util.construct_graph_input_feeddict(
                        self.node_info,
                        obs_n,
                        self.receive_idx,
                        self.send_idx,
                        self.node_type_idx,
                        self.inverse_node_type_idx,
                        self.output_type_idx,
                        self.inverse_output_type_idx,
                        self.last_batch_size,
                        request_data=['ob', 'idx']
                    )
            else:
                assert rollout_data is not None

                # preprocess the episodic information
                graph_obs, graph_parameters, _, _, _, _, _, _, _ = \
                    graph_data_util.construct_graph_input_feeddict(
                        self.node_info, obs_n,
                        -1, -1, -1, -1, -1, -1, -1,
                        request_data=['ob']
                    )
                nervenetplus_batch_pos, total_size = \
                    nervenetplus_util.nervenetplus_step_assign(
                        rollout_data, self.args.gnn_num_prop_steps
                    )
                _, _, self.receive_idx, self.send_idx, \
                    self.node_type_idx, self.inverse_node_type_idx, \
                    self.output_type_idx, self.inverse_output_type_idx, \
                    self.last_batch_size = \
                    graph_data_util.construct_graph_input_feeddict(
                        self.node_info,
                        np.empty(
                            [int(total_size / self.args.gnn_num_prop_steps)]
                        ),
                        self.receive_idx,
                        self.send_idx,
                        self.node_type_idx,
                        self.inverse_node_type_idx,
                        self.output_type_idx,
                        self.inverse_output_type_idx,
                        self.last_batch_size,
                        request_data=['idx']
                    )

            if step_model:
                feed_dict = {
                    self.step_batch_size_int_placeholder:
                        int(self.last_batch_size),
                    self.step_receive_idx_placeholder:
                        self.receive_idx,
                    self.step_inverse_node_type_idx_placeholder:
                        self.inverse_node_type_idx,
                    self.step_inverse_output_type_idx_placeholder:
                        self.inverse_output_type_idx
                }

                # append the input obs and parameters
                for i_node_type in self.node_info['node_type_dict']:
                    feed_dict[self.step_graph_obs_placeholder[i_node_type]] = \
                        graph_obs[i_node_type]
                    feed_dict[self.step_graph_parameters_placeholder[i_node_type]] = \
                        graph_parameters[i_node_type]

                # append the send idx
                for i_edge in self.node_info['edge_type_list']:
                    feed_dict[self.step_send_idx_placeholder[i_edge]] = \
                        self.send_idx[i_edge]

                # append the node type idx
                for i_node_type in self.node_info['node_type_dict']:
                    feed_dict[self.step_node_type_idx_placeholder[i_node_type]] \
                        = self.node_type_idx[i_node_type]

                # append the output type idx
                for i_output_type in self.node_info['output_type_dict']:
                    feed_dict[self.step_output_type_idx_placeholder[i_output_type]] \
                        = self.output_type_idx[i_output_type]

                # if the raw_obs is needed for the baseline
                if self.raw_obs_placeholder is not None:
                    feed_dict[self.raw_obs_placeholder] = obs_n
            else:
                feed_dict = {
                    self.batch_size_int_placeholder:
                        int(self.last_batch_size),
                    self.receive_idx_placeholder:
                        self.receive_idx,
                    self.inverse_node_type_idx_placeholder:
                        self.inverse_node_type_idx,
                    self.inverse_output_type_idx_placeholder:
                        self.inverse_output_type_idx
                }

                # append the input obs and parameters
                for i_node_type in self.node_info['node_type_dict']:
                    feed_dict[self.graph_obs_placeholder[i_node_type]] = \
                        graph_obs[i_node_type]
                    feed_dict[self.graph_parameters_placeholder[i_node_type]] = \
                        graph_parameters[i_node_type]

                # append the send idx
                for i_edge in self.node_info['edge_type_list']:
                    feed_dict[self.send_idx_placeholder[i_edge]] = \
                        self.send_idx[i_edge]

                # append the node type idx
                for i_node_type in self.node_info['node_type_dict']:
                    feed_dict[self.node_type_idx_placeholder[i_node_type]] \
                        = self.node_type_idx[i_node_type]

                # append the output type idx
                for i_output_type in self.node_info['output_type_dict']:
                    feed_dict[self.output_type_idx_placeholder[i_output_type]] \
                        = self.output_type_idx[i_output_type]

                # if the raw_obs is needed for the baseline
                if self.raw_obs_placeholder is not None:
                    feed_dict[self.raw_obs_placeholder] = obs_n
        else:
            # it is the most easy case, nice and easy
            feed_dict = {self.obs_placeholder: obs_n}
        self.nervenetplus_batch_pos = nervenetplus_batch_pos

        return feed_dict, nervenetplus_batch_pos

    def build_update_op_preprocess(self):
        '''
            @brief: The preprocess that is shared by trpo, ppo and vpg updates
        '''
        # the input placeholders for the input
        self.action_placeholder = tf.placeholder(
            tf.float32, [None, self.action_size],
            name='action_sampled_in_rollout'
        )
        self.advantage_placeholder = tf.placeholder(
            tf.float32, [None], name='advantage_value'
        )
        self.oldaction_dist_mu_placeholder = tf.placeholder(
            tf.float32, [None, self.action_size], name='old_act_dist_mu'
        )
        self.oldaction_dist_logstd_placeholder = tf.placeholder(
            tf.float32, [None, self.action_size], name='old_act_dist_logstd'
        )
        self.batch_size_float_placeholder = tf.placeholder(
            tf.float32, [], name='batch_size_float'
        )

        # the adaptive kl penalty
        if self.args.use_kl_penalty:
            self.kl_lambda_placeholder = tf.placeholder(tf.float32, [],
                                                        name='kl_lambda')

        # what are the probabilities of taking self.action, given new and old
        # distributions
        self.log_p_n = utils.gauss_log_prob(
            self.action_dist_mu,
            self.action_dist_logstd,
            self.action_placeholder
        )
        self.log_oldp_n = utils.gauss_log_prob(
            self.oldaction_dist_mu_placeholder,
            self.oldaction_dist_logstd_placeholder,
            self.action_placeholder
        )

        self.ratio = tf.exp(self.log_p_n - self.log_oldp_n)

        # the kl divergence between the old and new action
        self.kl = utils.gauss_KL(
            self.oldaction_dist_mu_placeholder,
            self.oldaction_dist_logstd_placeholder,
            self.action_dist_mu,
            self.action_dist_logstd
        )
        self.kl = self.kl / self.batch_size_float_placeholder

        # the entropy
        self.ent = utils.gauss_ent(
            self.action_dist_mu, self.action_dist_logstd
        )
        self.ent = self.ent / self.batch_size_float_placeholder

    def build_update_op_postprocess(self):
        '''
            @brief: The postprocess that is shared by trpo, ppo and vpg updates
        '''
        raise NotImplementedError

    def run(self):
        '''
            @brief:
                This is the standard function to be called by the
                "multiprocessing.Process"
        '''
        raise NotImplementedError

    def build_models(self):
        '''
            @brief:
                This is the function where the rollout agents and trpo agent
                build their networks, set up the placeholders, and gather the
                variable list.
        '''
        raise NotImplementedError

    def prepared_network_feeddict(self, data_dict):
        '''
            @brief:
                For the general policy network and graph network, we have
                different format, as we batch the network in the policy network
            @return:
                The feed_dict structure
        '''
        raise NotImplementedError

    def get_sess(self):
        return self.session

    def get_iteration_count(self):
        return self.session.run(self.iteration)

    def get_experiment_name(self):
        '''
            @brief:
                this is the unique id of the experiments. it might be useful if
                we are running several tasks on the server
        '''
        return self.args.task + '_' + self.args.time_id
