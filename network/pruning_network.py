# -----------------------------------------------------------------------------
#   @brief:
#       For the baseline network, we only need three functions to be defined
#       @fit, @save_checkpoint, and @load_checkpoint
#   @author:
#       Tingwu Wang, modified from kvfran and ppo repository.
# -----------------------------------------------------------------------------
import init_path
import tensorflow as tf
from util import nn_cells as nn


class GGNN(object):
    '''
        @brief:
            Gated Graph Sequence Neural Networks.
            Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015.
            arXiv preprint arXiv:1511.05493.
    '''

    def __init__(self, args, session, name_scope, initial_node_info, seed=1234,
                 bayesian_op=False):
        '''
            @input: the same as the ones defined in "policy_network"
        '''
        self.bayesian_op = bayesian_op
        self.args = args
        self._num_prop_steps = self.args.gnn_num_prop_steps
        self._initial_node_info = self._node_info = initial_node_info
        self._seed = seed
        self._input_feat_dim = self._hidden_dim = self.args.gnn_node_hidden_dim
        self._network_shape = self.args.network_shape
        self._name_scope = name_scope

        self._init_method = 'orthogonal'
        self._node_update_method = 'GRU'

        self._build_model()
        self._base_path = init_path.get_abs_base_dir()

        # for elem in self._node_info['ob_size_dict']:
        #     assert self._node_info['ob_size_dict'][elem] > 0

    def get_gnn_idx_placeholder(self):
        '''
            @brief: return the placeholders to the agent to construct feed dict
        '''
        return self._receive_idx, self._send_idx, \
            self._node_type_idx, self._inverse_node_type_idx, \
            self._batch_size_int

    def get_input_parameters_placeholder(self):
        return self._input_parameters

    def get_target_return_placeholder(self):
        return self._target_returns

    def _build_model(self):
        '''
            @brief: everything about the network goes here
        '''
        with tf.get_default_graph().as_default():
            tf.set_random_seed(self._seed)

            # prepare the network's input and output
            self._prepare()

            # define the network here
            self._build_network_weights()
            self._build_network_graph()
            self._build_baseline_train_placeholders()
            self._build_baseline_loss()

            # get the variable list ready
            self._set_var_list()

    def _build_baseline_loss(self):
        self._baseline_loss = tf.reduce_mean(
            tf.square(self._vpred - self._target_returns)
        )

    def _build_baseline_train_placeholders(self):
        self._target_returns = tf.placeholder(tf.float32, shape=[None],
                                              name='target_returns')

    def get_vf_loss(self):
        return self._baseline_loss

    def get_vpred_placeholder(self):
        return self._vpred

    def predict(self, feed_dict):
        '''
            @brief:
                generate the baseline function. This is only usable for baseline
                function
        '''
        baseline = self._session.run(self._vpred, feed_dict=feed_dict)
        baseline = baseline.reshape([-1])
        return baseline

    def _prepare(self):
        '''
            @brief:
                get the input placeholders ready. The _input placeholder has
                different size from the input we use for the general network.
        '''
        # step 1: build the input_obs and input_parameters
        self._num_nodes_placeholder = tf.placeholder(
            tf.int32, shape=(), name='num_nodes'
        )
        input_parameter_dtype = tf.float32
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
        # from util import fpdb; fpdb.fpdb().set_trace()
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

        # step 5: batch_size
        self._batch_size_int = tf.placeholder(
            tf.int32, shape=(), name='batch_size_int'
        )

    def _build_network_weights(self):
        '''
            @brief: build the network
        '''
        # step 1: build the weight parameters (mlp, gru)
        with tf.variable_scope(self._name_scope):
            # step 1_1: build the embedding matrix (mlp)
            # tensor shape (None, para_size) --> (None, input_dim - ob_size)
            self._MLP_embedding = {
                node_type: nn.MLP(
                    [self._input_feat_dim,
                     self._node_info['para_size_dict'][node_type]],
                    init_method=self._init_method,
                    act_func=['tanh'] * 1,  # one layer at most
                    add_bias=True,
                    scope='MLP_embedding_node_type_{}'.format(node_type)
                )
                for node_type in self._node_info['node_type_dict']
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
                        self._hidden_dim,
                        self._hidden_dim,
                        init_method=self._init_method,
                        scope='GRU_node_{}'.format(i_node_type)
                    )
                    for i_node_type in self._node_info['node_type_dict']
                }
            else:
                assert self._node_update_method == 'MLP'
                hidden_MLP_update_shape = self._network_shape
                self._Node_update = {
                    i_node_type: nn.MLPU(
                        message_dim=self._hidden_dim,
                        embedding_dim=self._hidden_dim,
                        hidden_shape=hidden_MLP_update_shape,
                        init_method=self._init_method,
                        act_func_type='tanh',
                        add_bias=True,
                        scope='MLPU_node_{}'.format(i_node_type)
                    )
                    for i_node_type in self._node_info['node_type_dict']
                }

            # step 1_6: the mlp for the mu of the actions
            MLP_out_shape = self._network_shape + [1] + \
                [self._hidden_dim]  # (l_1, l_2, ..., l_o, l_i)
            MLP_out_act_func = ['tanh'] * (len(MLP_out_shape) - 1)

            if self.bayesian_op:
                self._MLP_Out = nn.MLP(
                    MLP_out_shape, init_method=self._init_method,
                    act_func=MLP_out_act_func, add_bias=True, scope='MLP_out',
                    use_dropout=True
                )
                self.test_dropout_mask = []
                for feat_shape in [self._hidden_dim] + self._network_shape:
                    # [64, 64, 1, 64]
                    self.test_dropout_mask.append(
                        tf.placeholder(tf.float32, shape=[1, feat_shape])
                    )
                self.dropout_mask_shape = [self._hidden_dim] + self._network_shape
            else:
                self.test_dropout_mask = []
                self.dropout_mask_shape = None
                self._MLP_Out = nn.MLP(
                    MLP_out_shape, init_method=self._init_method,
                    act_func=MLP_out_act_func, add_bias=True, scope='MLP_out'
                )

    def _build_network_graph(self):
        # step 2: gather the input_feature from obs and node parameters
        self._input_embedding = {
            node_type: self._MLP_embedding[node_type](
                self._input_parameters[node_type]
            )[-1]
            for node_type in self._node_info['node_type_dict']
        }

        self._input_node_hidden = self._input_feat = self._input_embedding

        self._input_node_hidden = tf.concat(
            [self._input_node_hidden[node_type]
             for node_type in self._node_info['node_type_dict']],
            axis=0
        )
        self._input_node_hidden = tf.gather(  # get node order into graph order
            self._input_node_hidden,
            self._inverse_node_type_idx,
            name='get_order_back_gather_init'
        )

        # step 3: unroll the propogation
        self._node_hidden = [None] * (self._num_prop_steps + 1)
        self._node_hidden[-1] = self._input_node_hidden  # trick to use [-1]
        self._prop_msg = [None] * self._node_info['num_edge_type']

        for tt in range(self._num_prop_steps):
            ee = 0
            for i_edge_type in self._node_info['edge_type_list']:
                node_activate = \
                    tf.gather(
                        self._node_hidden[tt - 1],
                        self._send_idx[i_edge_type],
                        name='edge_id_{}_prop_steps_{}'.format(i_edge_type, tt)
                    )
                self._prop_msg[ee] = \
                    self._MLP_prop[i_edge_type](node_activate)[-1]

                ee += 1

            # aggregate messages
            concat_msg = tf.concat(self._prop_msg, 0)
            self.concat_msg = concat_msg
            message = tf.unsorted_segment_sum(
                concat_msg, self._receive_idx,
                self._num_nodes_placeholder * self._batch_size_int
            )
            denom_const = tf.unsorted_segment_sum(
                tf.ones_like(concat_msg), self._receive_idx,
                self._num_nodes_placeholder * self._batch_size_int
            )
            message = tf.div(message, (denom_const + tf.constant(1.0e-10)))

            # update the hidden states via GRU
            new_state = []
            for i_node_type in self._node_info['node_type_dict']:
                new_state.append(
                    self._Node_update[i_node_type](
                        tf.gather(
                            message,
                            self._node_type_idx[i_node_type],
                            name='GRU_message_node_type_{}_prop_step_{}'.format(
                                i_node_type, tt
                            )
                        ),
                        tf.gather(
                            self._node_hidden[tt - 1],
                            self._node_type_idx[i_node_type],
                            name='GRU_feat_node_type_{}_prop_steps_{}'.format(
                                i_node_type, tt
                            )
                        )
                    )
                )
            new_state = tf.concat(new_state, 0)  # BTW, the order is wrong
            # now, get the orders back
            self._node_hidden[tt] = tf.gather(
                new_state, self._inverse_node_type_idx,
                name='get_order_back_gather_prop_steps_{}'.format(tt)
            )

        # step 3: get the output
        self._root_node_state = \
            tf.gather(self._node_hidden[-2], [0], name='get_root_node')
        if self.bayesian_op:
            self._vpred = tf.reshape(self._MLP_Out(self._root_node_state)[-1], [1])
            self.thompson_vpred = tf.reshape(
                self._MLP_Out(self._root_node_state,
                              dropout_mask=self.test_dropout_mask)[-1], [1]
            )
        else:
            self._vpred = tf.reshape(self._MLP_Out(self._root_node_state)[-1], [1])

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        '''
        self.w_out = [var for var in self._trainable_var_list
                      if 'MLP_out' in var.name and '']
        MLP_out_shape = self._network_shape + [2] + \
                    [self._hidden_dim]  # (l_1, l_2, ..., l_o, l_i)
                MLP_out_act_func = ['tanh'] * (len(MLP_out_shape) - 1)
            self._MLP_Out = nn.MLP(
                MLP_out_shape, init_method=self._init_method,
                act_func=MLP_out_act_func, add_bias=True, scope='MLP_out'
            )
        '''

    def get_var_list(self):
        return self._trainable_var_list, self._all_var_list
