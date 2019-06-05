# ------------------------------------------------------------------------------
#   @brief:
#       The nervenet plus! Where we use truncated bp to optimize the weights
#   @Input:
#       _input_obs:                [num_prop_steps * batch_size, ob_size]
#       _input_hidden_state:       [batch_size, hidden_dim],
#       _input_parameters:         [batch_size, param_size_dict]
#   @Output:
#       _action_mu_output:         [num_prop_steps * batch_size, ac_size]
#       _action_dist_logstd_param: [num_prop_steps * batch_size, ac_size]
# ------------------------------------------------------------------------------
import init_path
import tensorflow as tf
import numpy as np
from .policy_network import policy_network
# from util import logger
from graph_util import mujoco_parser
from graph_util import gnn_util
from graph_util import gen_gnn_param
from util import nn_cells as nn
from six.moves import xrange


class nervenet(policy_network):

    def __init__(self, session, name_scope,
                 input_size, output_size,
                 adj_matrix, node_attr,
                 weight_init_methods='orthogonal',
                 is_rollout_agent=True,
                 args=None):

        self._node_update_method = args.node_update_method
        self.adj_matrix = adj_matrix
        self.node_attr = node_attr

        policy_network.__init__(
            self,
            session,
            name_scope,
            input_size,
            output_size,
            ob_placeholder=None,
            trainable=True,
            build_network_now=False,
            define_std=True,
            args=args
        )

        self._base_dir = init_path.get_abs_base_dir()
        self._root_connection_option = args.root_connection_option
        self._num_prop_steps = args.gnn_num_prop_steps
        self._init_method = weight_init_methods
        self._gnn_node_option = args.gnn_node_option
        self._gnn_output_option = args.gnn_output_option
        self._gnn_embedding_option = args.gnn_embedding_option

        self.is_rollout_agent = is_rollout_agent

        self._nstep = 1 if self.is_rollout_agent else self._num_prop_steps

        # parse the network shape and do validation check
        self._network_shape = args.network_shape
        self._hidden_dim = args.gnn_node_hidden_dim
        self._input_feat_dim = args.gnn_input_feat_dim
        self._seed = args.seed
        self._npr = np.random.RandomState(args.seed)

        assert self._input_feat_dim == self._hidden_dim

        self._build_model()

    def _build_model(self):
        '''
            @brief: everything about the network goes here
        '''
        with tf.get_default_graph().as_default():
            tf.set_random_seed(self._seed)

            # record the iteration count
            self._iteration = tf.Variable(0, trainable=False, name='step')

            # read from the xml files
            self._parse_mujoco_template()

            # prepare the network's input and output
            self._prepare_placeholders()

            # define the network here
            self._build_network_weights()
            self._build_network_graph()

            # get the variable list ready
            self._set_var_list()

    def _prepare_placeholders(self):
        '''
            @brief:
                get the input placeholders ready. The _input placeholder has
                different size from the input we use for the general network.
        '''

        # step 1: build the input_obs and input_parameters
        self._input_obs = {
            node_type: tf.placeholder(
                tf.float32,
                [None, self._node_info['ob_size_dict'][node_type]],
                name='input_ob_placeholder_ggnn'
            )
            for node_type in self._node_info['node_type_dict']
        }

        self._input_hidden_state = {
            node_type: tf.placeholder(
                tf.float32,
                [None, self._hidden_dim],
                name='input_hidden_dim_' + node_type
            )
            for node_type in self._node_info['node_type_dict']
        }

        input_parameter_dtype = tf.int32 \
            if 'noninput' in self._gnn_embedding_option else tf.float32
        self._input_parameters = {
            node_type: tf.placeholder(
                input_parameter_dtype,
                [None, self._node_info['para_size_dict'][node_type]],
                name='input_para_placeholder_ggnn')
            for node_type in self._node_info['node_type_dict']
        }

        # step 2: the receive and send index
        self._receive_idx = tf.placeholder(
            tf.int32, shape=(None), name='receive_idx'
        )
        self._send_idx = {
            edge_type: tf.placeholder(
                tf.int32, shape=(None),
                name='send_idx_{}'.format(edge_type))
            for edge_type in self._node_info['edge_type_list']
        }

        # step 3: the node type index and inverse node type index
        self._node_type_idx = {
            node_type: tf.placeholder(
                tf.int32, shape=(None),
                name='node_type_idx_{}'.format(node_type))
            for node_type in self._node_info['node_type_dict']
        }
        self._inverse_node_type_idx = tf.placeholder(
            tf.int32, shape=(None), name='inverse_node_type_idx'
        )

        # step 4: the output node index
        self._output_type_idx = {
            output_type: tf.placeholder(
                tf.int32, shape=(None),
                name='output_type_idx_{}'.format(output_type)
            )
            for output_type in self._node_info['output_type_dict']
        }
        self._inverse_output_type_idx = tf.placeholder(
            tf.int32, shape=(None), name='inverse_output_type_idx'
        )

        # step 5: batch_size
        self._batch_size_int = tf.placeholder(
            tf.int32, shape=(), name='batch_size_int'
        )

    def _build_network_weights(self):
        '''
            @brief: build the network
            @weights:
                _MLP_embedding (1 layer)
                _MLP_ob_mapping (1 layer)
                _MLP_prop (2 layer)
                _MLP_output (2 layer)
        '''
        # step 1: build the weight parameters (mlp, gru)
        with tf.variable_scope(self._name_scope):
            # step 1_1: build the embedding matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            assert self._input_feat_dim % 2 == 0
            if 'noninput' not in self._gnn_embedding_option:
                self._MLP_embedding = {
                    node_type: nn.MLP(
                        [self._input_feat_dim / 2,
                         self._node_info['para_size_dict'][node_type]],
                        init_method=self._init_method,
                        act_func=['tanh'] * 1,  # one layer at most
                        add_bias=True,
                        scope='MLP_embedding_node_type_{}'.format(node_type)
                    )
                    for node_type in self._node_info['node_type_dict']
                    if self._node_info['ob_size_dict'][node_type] > 0
                }
                self._MLP_embedding.update({
                    node_type: nn.MLP(
                        [self._input_feat_dim,
                         self._node_info['para_size_dict'][node_type]],
                        init_method=self._init_method,
                        act_func=['tanh'] * 1,  # one layer at most
                        add_bias=True,
                        scope='MLP_embedding_node_type_{}'.format(node_type)
                    )
                    for node_type in self._node_info['node_type_dict']
                    if self._node_info['ob_size_dict'][node_type] == 0
                })
            else:
                embedding_vec_size = max(
                    np.reshape(
                        [max(self._node_info['node_parameters'][i_key])
                         for i_key in self._node_info['node_parameters']],
                        [-1]
                    )
                ) + 1
                embedding_vec_size = int(embedding_vec_size)
                self._embedding_variable = {}

                out = self._npr.randn(
                    embedding_vec_size, int(self._input_feat_dim / 2)
                ).astype(np.float32)
                out *= 1.0 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                self._embedding_variable[False] = tf.Variable(
                    out, name='embedding_HALF', trainable=self._trainable
                )

                if np.any([node_size == 0 for _, node_size
                           in self._node_info['ob_size_dict'].items()]):

                    out = self._npr.randn(
                        embedding_vec_size, self._input_feat_dim
                    ).astype(np.float32)
                    out *= 1.0 / np.sqrt(np.square(out).sum(axis=0,
                                                            keepdims=True))
                    self._embedding_variable[True] = tf.Variable(
                        out, name='embedding_FULL', trainable=self._trainable
                    )

            # step 1_2: build the ob mapping matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            self._MLP_ob_mapping = {
                node_type: nn.MLP(
                    [self._input_feat_dim / 2,
                     self._node_info['ob_size_dict'][node_type]],
                    init_method=self._init_method,
                    act_func=['tanh'] * 1,  # one layer at most
                    add_bias=True,
                    scope='MLP_embedding_node_type_{}'.format(node_type)
                )
                for node_type in self._node_info['node_type_dict']
                if self._node_info['ob_size_dict'][node_type] > 0
            }

            # step 1_4: build the mlp for the propogation between nodes
            MLP_prop_shape = self._network_shape + \
                [self._hidden_dim] + [self._hidden_dim]
            self._MLP_prop = {
                i_edge: nn.MLP(
                    MLP_prop_shape,
                    init_method=self._init_method,
                    act_func=['tanh'] * (len(MLP_prop_shape) - 1),
                    add_bias=True,
                    scope='MLP_prop_edge_{}'.format(i_edge)
                )
                for i_edge in self._node_info['edge_type_list']
            }

            # step 1_5: build the node update function for each node type
            if self._node_update_method == 'GRU':
                self._Node_update = {
                    i_node_type: nn.GRU(
                        self._hidden_dim * 2,  # for both the message and ob
                        self._hidden_dim,
                        init_method=self._init_method,
                        scope='GRU_node_{}'.format(i_node_type)
                    )
                    for i_node_type in self._node_info['node_type_dict']
                }
            else:
                assert False

            # step 1_6: the mlp for the mu of the actions
            # (l_1, l_2, ..., l_o, l_i)
            MLP_out_shape = self._network_shape + \
                [self.args.gnn_output_per_node] + [self._hidden_dim]
            MLP_out_act_func = ['tanh'] * (len(MLP_out_shape) - 1)
            MLP_out_act_func[-1] = None

            self._MLP_Out = {
                output_type: nn.MLP(
                    MLP_out_shape,
                    init_method=self._init_method,
                    act_func=MLP_out_act_func,
                    add_bias=True,
                    scope='MLP_out'
                )
                for output_type in self._node_info['output_type_dict']
            }

            # step 1_8: build the log std for the actions
            self._action_dist_logstd = tf.Variable(
                (0.0 * self._npr.randn(1, self._output_size)).astype(
                    np.float32
                ),
                name="policy_logstd",
                trainable=self._trainable
            )

    def _build_input_graph(self):
        if 'noninput' not in self._gnn_embedding_option:
            self._input_embedding = {
                node_type: self._MLP_embedding[node_type](
                    self._input_parameters[node_type]
                )[-1]
                for node_type in self._node_info['node_type_dict']
            }
        else:
            self._input_embedding = {
                node_type: tf.gather(
                    self._embedding_variable[
                        self._node_info['ob_size_dict'][node_type] == 0
                    ],
                    tf.reshape(self._input_parameters[node_type], [-1])
                )
                for node_type in self._node_info['node_type_dict']
            }

        # shape: [n_step, node_num, embedding_size + ob_size]
        self._ob_feat = {
            node_type: self._MLP_ob_mapping[node_type](
                self._input_obs[node_type]
            )[-1]
            for node_type in self._node_info['node_type_dict']
            if self._node_info['ob_size_dict'][node_type] > 0
        }
        self._ob_feat.update({
            node_type: self._input_obs[node_type]
            for node_type in self._node_info['node_type_dict']
            if self._node_info['ob_size_dict'][node_type] == 0
        })

        self._input_feat = {
            node_type: tf.concat([
                tf.reshape(
                    self._input_embedding[node_type],
                    [-1, self._nstep *
                     len(self._node_info['node_type_dict'][node_type]),
                     int(self._input_feat_dim / 2)],
                ),
                tf.reshape(
                    self._ob_feat[node_type],
                    [-1, self._nstep *
                     len(self._node_info['node_type_dict'][node_type]),
                     int(self._input_feat_dim / 2)],
                )
            ], axis=2)
            for node_type in self._node_info['node_type_dict']
        }

        split_feat_list = {
            node_type: tf.split(
                self._input_feat[node_type],
                self._nstep,
                axis=1,
                name='split_into_nstep' + node_type
            )
            for node_type in self._node_info['node_type_dict']
        }
        feat_list = []
        for i_step in range(self._nstep):
            # for node_type in self._node_info['node_type_dict']:
            feat_list.append(
                tf.concat(
                    [tf.reshape(split_feat_list[node_type][i_step],
                                [-1, self._input_feat_dim])
                     for node_type in self._node_info['node_type_dict']],
                    axis=0  # the node
                )
            )
        self._input_feat_list = [
            tf.gather(  # get node order into graph order
                i_step_data,
                self._inverse_node_type_idx,
                name='get_order_back_gather_init' + str(i_step),
            )
            for i_step, i_step_data in enumerate(feat_list)
        ]

        current_hidden_state = tf.concat(
            [self._input_hidden_state[node_type]
             for node_type in self._node_info['node_type_dict']],
            axis=0
        )
        current_hidden_state = tf.gather(  # get node order into graph order
            current_hidden_state,
            self._inverse_node_type_idx,
            name='get_order_back_gather_init'
        )
        return current_hidden_state

    def _build_network_graph(self):
        current_hidden_state = self._build_input_graph()

        # step 3: unroll the propogation
        self._action_mu_output = []  # [nstep, None, n_action_size]

        for tt in xrange(self._nstep):
            current_input_feat = self._input_feat_list[tt]
            self._prop_msg = []
            for ee, i_edge_type in enumerate(self._node_info['edge_type_list']):
                node_activate = \
                    tf.gather(
                        current_input_feat,
                        self._send_idx[i_edge_type],
                        name='edge_id_{}_prop_steps_{}'.format(i_edge_type, tt)
                    )
                self._prop_msg.append(
                    self._MLP_prop[i_edge_type](node_activate)[-1]
                )

            # aggregate messages
            concat_msg = tf.concat(self._prop_msg, 0)
            self.concat_msg = concat_msg
            message = tf.unsorted_segment_sum(
                concat_msg, self._receive_idx,
                self._node_info['num_nodes'] * self._batch_size_int
            )

            denom_const = tf.unsorted_segment_sum(
                tf.ones_like(concat_msg), self._receive_idx,
                self._node_info['num_nodes'] * self._batch_size_int
            )
            message = tf.div(message, (denom_const + tf.constant(1.0e-10)))
            node_update_input = tf.concat([message, current_input_feat], axis=1,
                                          name='ddbug' + str(tt))

            # update the hidden states via GRU
            new_state = []
            for i_node_type in self._node_info['node_type_dict']:
                new_state.append(
                    self._Node_update[i_node_type](
                        tf.gather(
                            node_update_input,
                            self._node_type_idx[i_node_type],
                            name='GRU_message_node_type_{}_prop_step_{}'.format(
                                i_node_type, tt
                            )
                        ),
                        tf.gather(
                            current_hidden_state,
                            self._node_type_idx[i_node_type],
                            name='GRU_feat_node_type_{}_prop_steps_{}'.format(
                                i_node_type, tt
                            )
                        )
                    )
                )
            self.output_hidden_state = {
                node_type: new_state[i_id]
                for i_id, node_type
                in enumerate(self._node_info['node_type_dict'])
            }
            new_state = tf.concat(new_state, 0)  # BTW, the order is wrong
            # now, get the orders back
            current_hidden_state = tf.gather(
                new_state, self._inverse_node_type_idx,
                name='get_order_back_gather_prop_steps_{}'.format(tt)
            )

            # self._action_mu_output = []  # [nstep, None, n_action_size]
            action_mu_output = []
            for output_type in self._node_info['output_type_dict']:
                action_mu_output.append(
                    self._MLP_Out[output_type](
                        tf.gather(
                            current_hidden_state,
                            self._output_type_idx[output_type],
                            name='output_type_{}'.format(output_type)
                        )
                    )[-1]
                )

            action_mu_output = tf.concat(action_mu_output, 0)
            action_mu_output = tf.gather(
                action_mu_output,
                self._inverse_output_type_idx,
                name='output_inverse'
            )

            action_mu_output = tf.reshape(action_mu_output,
                                          [self._batch_size_int, -1])
            self._action_mu_output.append(action_mu_output)

        self._action_mu_output = tf.reshape(
            tf.concat(self._action_mu_output, axis=1), [-1, self._output_size]
        )
        # step 4: build the log std for the actions
        self._action_dist_logstd_param = tf.reshape(
            tf.tile(
                tf.reshape(self._action_dist_logstd, [1, 1, self._output_size],
                           name='test'),
                [self._nstep, self._batch_size_int, 1]
            ), [-1, self._output_size]
        )

    def _parse_mujoco_template(self):
        '''
            @brief:
                In this function, we construct the dict for node information.
                The structure is _node_info
            @attribute:
                1. general informatin about the graph
                    @self._node_info['tree']
                    @self._node_info['debug_info']
                    @self._node_info['relation_matrix']

                2. information about input output
                    @self._node_info['input_dict']:
                        self._node_info['input_dict'][id_of_node] is a list of
                        ob positions
                    @self._node_info['output_list']

                3. information about the node
                    @self._node_info['node_type_dict']:
                        self._node_info['node_type_dict']['body'] is a list of
                        node id
                    @self._node_info['num_nodes']

                4. information about the edge
                    @self._node_info['edge_type_list'] = self._edge_type_list
                        the list of edge ids
                    @self._node_info['num_edges']
                    @self._node_info['num_edge_type']

                6. information about the index
                    @self._node_info['node_in_graph_list']
                        The order of nodes if placed by types ('joint', 'body')
                    @self._node_info['inverse_node_list']
                        The inverse of 'node_in_graph_list'
                    @self._node_info['receive_idx'] = receive_idx
                    @self._node_info['receive_idx_raw'] = receive_idx_raw
                    @self._node_info['send_idx'] = send_idx

                7. information about the embedding size and ob size
                    @self._node_info['para_size_dict']
                    @self._node_info['ob_size_dict']
                        self._node_info['ob_size_dict']['root'] = 10
                        self._node_info['ob_size_dict']['joint'] = 6
            '''
        # step 0: parse the mujoco xml
        if 'evo' in self.args.task:
            self._node_info = gen_gnn_param.gen_gnn_param(
                self._task_name,
                self.adj_matrix,
                self.node_attr,
                gnn_node_option=self._gnn_node_option,
                root_connection_option=self._root_connection_option,
                gnn_output_option=self._gnn_output_option,
                gnn_embedding_option=self._gnn_embedding_option
            )
        else:
            self._node_info = mujoco_parser.parse_mujoco_graph(
                self._task_name,
                gnn_node_option=self._gnn_node_option,
                root_connection_option=self._root_connection_option,
                gnn_output_option=self._gnn_output_option,
                gnn_embedding_option=self._gnn_embedding_option
            )

        # step 1: check that the input and output size is matched
        gnn_util.io_size_check(self._input_size, self._output_size,
                               self._node_info, self._is_baseline)

        # step 2: check for ob size for each node type, construct the node dict
        self._node_info = gnn_util.construct_ob_size_dict(self._node_info,
                                                          self._input_feat_dim)

        # step 3: get the inverse node offsets (used to construct gather idx)
        self._node_info = gnn_util.get_inverse_type_offset(self._node_info,
                                                           'node')

        # step 4: get the inverse node offsets (used to gather output idx)
        self._node_info = gnn_util.get_inverse_type_offset(self._node_info,
                                                           'output')

        # step 5: register existing edge and get the receive and send index
        self._node_info = gnn_util.get_receive_send_idx(self._node_info)

    def get_num_nodes(self):
        return self._node_info['num_nodes']

    def get_logstd(self):
        return self._action_dist_logstd

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

    def get_node_info(self):
        return self._node_info

    def get_gnn_idx_placeholder(self):
        '''
            @brief: return the placeholders to the agent to construct feed dict
        '''
        return self._receive_idx, self._send_idx, \
            self._node_type_idx, self._inverse_node_type_idx, \
            self._output_type_idx, self._inverse_output_type_idx, \
            self._batch_size_int

    def get_input_obs_placeholder(self):
        return self._input_obs

    def get_input_parameters_placeholder(self):
        return self._input_parameters

    def get_output_hidden_state_list(self):
        return [self.output_hidden_state[key]
                for key in self._node_info['node_type_dict']]

    def get_input_hidden_state_placeholder(self):
        '''
        self._input_hidden_state = {
            node_type: tf.placeholder(
                tf.float32,
                [None, self._hidden_dim],
                name='input_hidden_dim'
            )
            for node_type in self._node_info['node_type_dict']
        }
        '''
        return self._input_hidden_state
