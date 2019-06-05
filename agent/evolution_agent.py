# ------------------------------------------------------------------------------
#   @brief:
#       The evolutionary agent is responsible for everything. It takes the
#       @weights, and the @xml of the env And it output the updated weights
#   @author:
#       Tingwu Wang
# ------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import init_path
import random

from util import utils
from util import logger
from util import parallel_util
from util import ob_normalizer
import os
from .agent import base_agent
from graph_util import graph_data_util
from .rollout_master_agent import parallel_rollout_master_agent
# from env import env_wrapper
import time
from util import agent_util
from graph_util import gnn_io_util

from env import model_gen
from env import hierarchy_model

from lxml import etree
from evolution import species2species


class evolutionary_agent(base_agent, parallel_rollout_master_agent):
    '''
        @functions:
            see the @optimization_agent.py, @rollout_master_agent.py and
            @rollout_agent.py
    '''

    def __init__(self, args, task_q, result_q, agent_id,
                 name_scope='evolutionary_agent'):
        # the base agent
        self.agent_id = agent_id
        base_agent.__init__(
            self, args=args, observation_size=-1, action_size=-1,
            task_q=task_q, result_q=result_q, name_scope=name_scope
        )

        self.reset_running_mean_info()
        # evolutionary agent will never load from ckpt_name
        # self.load_running_means()
        self.ob_normalizer = ob_normalizer.normalizer()

        # the variables and networks to be used, init them before use them
        self.baseline_network = None
        self.env_info = None
        self.error_count = 0

        # used to save the checkpoint files
        self.best_reward = -np.inf
        self.timesteps_so_far = 0
        self._npr = np.random.RandomState(args.seed + self.agent_id)
        self.debug = 0

        # self.last_save_iteration = 0
        self.last_average_reward = np.nan
        self.end_average_reward = np.nan
        self.start_time = None
        self.is_dead_species = False
        self.brute_search_reward = []

        if self.args.nervenetplus:
            if self.args.fc_pruning:
                assert self.args.use_nervenet
            else:
                assert self.args.use_nervenet and self.args.use_gnn_as_policy

    def rollout_and_train(self):
        # step 0: reset the weights?
        start_count_reward_iteration = \
            self.args.evolutionary_sub_iteration * \
            (1 - self.args.reward_count_percentage)

        self.end_average_reward = []
        self.last_average_reward = 0

        for i_iteration in range(self.args.evolutionary_sub_iteration):
            if_record_reward = i_iteration > \
                np.floor(start_count_reward_iteration)
            # step 1: generating the rollots
            rollout_data = self.ask_for_rollouts()

            # step 2: update the runnning mean info
            self.update_running_means(rollout_data)

            # step 3: update the network
            stats = self.update_parameters(rollout_data)

            # step 4: set the step model if needed
            if self.step_policy_network is not None:
                self.set_step_policy(self.get_policy())

            if if_record_reward:
                self.end_average_reward.append(stats['avg_reward'])
            self.last_average_reward = stats['avg_reward']
            # logger.error(stats['avg_reward'])
            # logger.error("lr: {}".format(self.current_lr))
            if self.args.brute_force_search:
                self.brute_search_reward.append(stats['avg_reward'])
        stats['brute_reward'] = self.brute_search_reward

        return stats

    def ask_for_rollouts(self):
        '''
            @brief: similar to the rollout_master_agent
        '''

        # update the filters
        self.ob_normalizer.set_parameters(
            self.running_mean_info['mean'],
            self.running_mean_info['variance'],
            self.running_mean_info['step']
        )

        num_timesteps_received = 0
        rollout_data = []

        num_rollouts = 0
        while True:  # generate the rollouts
            num_rollouts += 1

            if self.args.test and self.args.test < num_rollouts:
                break
            if num_timesteps_received >= self.args.timesteps_per_batch:
                break

            # request episodes
            traj_episode = self.rollout_for_one_episode()

            # collect episodes
            rollout_data.append(traj_episode)
            num_timesteps_received += len(traj_episode['rewards'])

        return rollout_data

    def rollout_for_one_episode(self):

        # init the variables
        if self.args.nervenetplus:
            self.current_state = {
                node_type: np.zeros(
                    [len(self.node_info['node_type_dict'][node_type]),
                     self.args.gnn_node_hidden_dim]
                )
                for node_type in self.node_info['node_type_dict']
            }
            hidden_state = {
                node_type: [] for node_type in self.node_info['node_type_dict']
            }

        obs, actions, rewards, action_dists_mu, \
            action_dists_logstd, raw_obs = [], [], [], [], [], []
        path = dict()
        raw_ob = self.env.reset()
        ob = self.ob_normalizer.filter(raw_ob)

        while True:
            # generate the policy
            if self.args.nervenetplus:
                for node_type in self.node_info['node_type_dict']:
                    hidden_state[node_type].append(
                        self.current_state[node_type]
                    )
            action, action_dist_mu, action_dist_logstd = self.act(ob)

            # record the stats
            obs.append(ob)
            raw_obs.append(raw_ob)
            actions.append(action)
            action_dists_mu.append(action_dist_mu)
            action_dists_logstd.append(action_dist_logstd)

            try:
                result = self.env.step(action)
            except Exception as ex:
                mj_err = str(ex)
                logger.error(mj_err)
                if 'mjWARN_INERTIA' in mj_err:
                    self.error_count += 1
                    if self.error_count >= 10:
                        logger.info('Killing the species')
                        result = []
                        result.append(np.random.rand(*raw_ob.shape))
                        result.append(-2048.0)
                        result.append(1)
                        self.is_dead_species = True
                    else:
                        logger.info('Resetting a specific species')
                        obs, actions, rewards, action_dists_mu, \
                            action_dists_logstd, raw_obs = [], [], [], [], [], []
                        path = dict()
                        raw_ob = self.env.reset()
                        ob = self.ob_normalizer.filter(raw_ob)
                        continue

                elif 'mjWARN_BADQACC' in mj_err or \
                     'mjWARN_BADCTRL' in mj_err:
                    logger.info('Killing the species')
                    result = []
                    result.append(np.random.rand(*raw_ob.shape))
                    result.append(-2048.0)
                    result.append(1)
                    self.is_dead_species = True

                else:
                    result = self.env.step(action)

            raw_ob = result[0]
            ob = self.ob_normalizer.filter(result[0])
            # print(result[0])
            rewards.append((result[1]))

            if result[2]:  # terminated
                path = {
                    "obs": np.array(obs),
                    "raw_obs": np.array(raw_obs),
                    "action_dists_mu": np.concatenate(action_dists_mu),
                    "action_dists_logstd": np.concatenate(action_dists_logstd),
                    "rewards": np.array(rewards),
                    "actions":  np.array(actions)
                }

                if self.args.nervenetplus:
                    path['hidden_state'] = hidden_state
                    for node_type in self.node_info['node_type_dict']:
                        path['hidden_state'][node_type] = \
                            np.concatenate(hidden_state[node_type])
                break

        return path

    def act(self, obs):
        '''
            @brief:
                The function where the agent actually decide what it want to do
        '''
        obs = np.expand_dims(obs, 0)
        feed_dict, _ = self.prepared_policy_network_feeddict(
            obs, step_model=self.args.nervenetplus
        )

        if self.args.nervenetplus:
            # append the hidden state into the input
            step_input_hidden_state = \
                self.step_policy_network.get_input_hidden_state_placeholder()
            for node_type in self.node_info['node_type_dict']:
                feed_dict[step_input_hidden_state[node_type]] = \
                    self.current_state[node_type]

            results = self.session.run(
                [self.step_action_dist_mu, self.step_action_dist_logstd] +
                self.step_policy_network.get_output_hidden_state_list(),
                feed_dict=feed_dict
            )
            action_dist_mu = results[0]
            action_dist_logstd = results[1]
            self.current_state = {
                node_info: results[2 + iid]
                for iid, node_info in
                enumerate(self.node_info['node_type_dict'])
            }

        else:
            action_dist_mu, action_dist_logstd = self.session.run(
                [self.action_dist_mu, self.action_dist_logstd],
                feed_dict=feed_dict
            )

        # samples the guassian distribution
        act = action_dist_mu + np.exp(action_dist_logstd) * \
            self._npr.randn(*action_dist_logstd.shape)
        act = act.ravel()
        return act, action_dist_mu, action_dist_logstd

    def build_env(self, received_data=None):
        assert 'evo' in self.args.task

        allow_monitor = 0
        if received_data is None:
            if self.args.new_species_struct:
                if self.args.more_body_nodes_at_start:
                    body_num = random.randint(3, 6)
                else:
                    body_num = 3
                species = hierarchy_model.Species(self.args, body_num=body_num)
                adj_matrix, node_attr = species.get_gene()
                xml_struct, xml_str = species.get_xml()
            else:
                adj_matrix, node_attr, xml_str = model_gen.get_initial_settings()
                xml_str = etree.tostring(xml_str, pretty_print=True)

        else:
            # if we set the visualization open
            if 'rank_info' in received_data:
                generation, rank = [int(info) for info
                                    in received_data['rank_info'].split('_')]
                if generation % self.args.species_visualize_freq == 0 and \
                        rank <= self.args.visualize_top_species:
                    allow_monitor = True

            if self.args.new_species_struct:
                species = received_data['species']

            adj_matrix = received_data['adj_matrix']
            node_attr = received_data['node_attr']
            xml_str = received_data['xml_str']
            # self.current_lr = received_data['lr']

        # create the environment for this agent
        if 'fish' in self.args.task:
            from env import fish_env_wrapper
            self.env = fish_env_wrapper.dm_evofish3d_wrapper(
                self.args, self._npr.randint(0, 99999), allow_monitor,
                adj_matrix=adj_matrix, xml_str=xml_str
            )
        elif 'walker' in self.args.task or \
             'cheetah' in self.args.task or \
             'hopper' in self.args.task:
            from env import walker_env_wrapper
            self.env = walker_env_wrapper.dm_evowalker_wrapper(
                self.args, self._npr.randint(0, 99999), allow_monitor,
                adj_matrix=adj_matrix, xml_str=xml_str
            )
        else:
            raise NotImplementedError

        if allow_monitor and received_data is not None:
            self.env.set_output_dir(received_data['video_save_path'])

        self.is_dm_env = True
        self.action_size = self.env.get_action_size()
        self.observation_size = self.env.get_observation_size()

        if self.args.new_species_struct:
            self.species = species

        self.adj_matrix = adj_matrix
        self.node_attr = node_attr
        self.xml_str = xml_str

    def run(self):
        '''
            @brief:
                this is the standard function to be called by the
                "multiprocessing.Process"

            @NOTE:
                check the parallel_util.py for definitions
        '''
        # the main training process
        received_signal, received_data = self.task_q.get()

        if received_signal == parallel_util.END_SIGNAL:
            # kill the learner
            self.task_q.task_done()

        elif received_signal == parallel_util.AGENT_EVOLUTION_START:
            self.build_env(received_data)
            self.build_models()

            training_stats = self.rollout_and_train()
            self.task_q.task_done()
            species_data = self.get_species_info(training_stats, received_data)
            self.result_q.put(species_data)

        elif received_signal == parallel_util.AGENT_EVOLUTION_TRAIN:
            self.build_env(received_data)
            self.build_models(received_data)

            training_stats = self.rollout_and_train()
            self.task_q.task_done()
            species_data = self.get_species_info(training_stats, received_data)
            self.result_q.put(species_data)

    def build_models(self, received_data=None):
        '''
            @brief:
                this is the function where the rollout agents and optimization
                agent build their networks, set up the placeholders, and gather
                the variable list.
        '''
        # make sure that the agent has a session
        self.start_time = time.time()
        self.build_session()

        self.build_policy_network(adj_matrix=self.adj_matrix,
                                  node_attr=self.node_attr)

        self.baseline_network, self.target_return_placeholder, \
            self.raw_obs_placeholder = \
            agent_util.build_baseline_network(
                self.args, self.session, self.name_scope, self.observation_size,
                self.gnn_placeholder_list, self.obs_placeholder
            )
        # for key in tf.trainable_variables():
        #     print(key.name)
        # from util import fpdb; fpdb.fpdb().set_trace()

        # the training op and graphs
        self.build_ppo_update_op()
        self.update_parameters = self.update_ppo_parameters

        # init the network parameters (xavier initializer)
        self.session.run(tf.global_variables_initializer())

        # the set weight policy ops
        self.get_policy = \
            utils.GetPolicyWeights(self.session, self.policy_var_list)
        self.set_policy = \
            utils.SetPolicyWeights(self.session, self.policy_var_list)

        if received_data is not None:
            # set the running_mean and network weights here
            if received_data['reset']:
                # no inheriting from the parents!
                logger.info('Not Inheriting from parents!')
            else:
                if self.args.use_gnn_as_policy:
                    current_species_format = self.get_species_info()
                    processed_species_info = species2species.process_inherited_info(
                        raw_species_info=received_data,
                        current_species_format=current_species_format,
                        is_nervenet=self.args.use_gnn_as_policy
                    )
                    self.set_policy(processed_species_info['policy_weights'])
                    self.set_running_means(
                        processed_species_info['running_mean_info']
                    )
                    self.current_lr = received_data['lr']
                else:
                    # old species of the fc baselines
                    if received_data['SpcID'] > 0 and self.args.fc_amortized_fitness:
                        self.set_policy(received_data['policy_weights'])
                        self.set_running_means(
                            received_data['running_mean_info']
                        )

        # prepare the feed_dict info for the ppo minibatches
        self.prepare_feed_dict_map()

        # prepared the init kl divergence if needed
        self.current_kl_lambda = 1

    def build_ppo_update_op(self):
        '''
            @brief:
                The only difference from the vpg update is that here we clip the
                ratio
        '''
        self.build_update_op_preprocess()

        # step 1: the surrogate loss
        self.ratio_clip = tf.clip_by_value(self.ratio,
                                           1.0 - self.args.ppo_clip,
                                           1.0 + self.args.ppo_clip)

        # the pessimistic surrogate loss
        self.surr = tf.minimum(self.ratio_clip * self.advantage_placeholder,
                               self.ratio * self.advantage_placeholder)
        self.surr = -tf.reduce_mean(self.surr)

        # step 2: the value function loss. if not tf, the loss will be fit in
        # update_parameters_postprocess
        self.vf_loss = self.baseline_network.get_vf_loss()
        self.loss = self.surr

        # step 3: the kl penalty term
        if self.args.use_kl_penalty:
            self.loss += self.kl_lambda_placeholder * self.kl
            self.loss += self.args.kl_eta * \
                tf.square(tf.maximum(0.0, self.kl - 2.0 * self.args.target_kl))

        # step 4: weight decay
        self.weight_decay_loss = 0.0
        for var in tf.trainable_variables():
            self.weight_decay_loss += tf.nn.l2_loss(var)
        if self.args.use_weight_decay:
            self.loss += self.weight_decay_loss * self.args.weight_decay_coeff

        # step 5: build the optimizer
        self.lr_placeholder = tf.placeholder(tf.float32, [],
                                             name='learning_rate')
        self.current_lr = self.args.lr
        if self.args.use_gnn_as_policy:
            # we need to clip the gradient for the ggnn
            self.optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
            self.tvars = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.tvars)
            self.clipped_grads, _ = tf.clip_by_global_norm(
                self.grads,
                self.args.grad_clip_value,
                name='clipping_gradient'
            )

            self.update_op = self.optimizer.apply_gradients(
                zip(self.clipped_grads, self.tvars)
            )
        else:
            self.update_op = tf.train.AdamOptimizer(
                learning_rate=self.lr_placeholder, epsilon=1e-5
            ).minimize(self.loss)

            self.tvars = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.tvars)

        if self.args.shared_network:
            assert False
        self.update_vf_op = tf.train.AdamOptimizer(
            learning_rate=self.args.value_lr, epsilon=1e-5
        ).minimize(self.vf_loss)

    def update_ppo_parameters(self, paths):
        '''
            @brief: update the ppo
        '''
        # step 1: get the data dict
        feed_dict = self.prepared_network_feeddict(paths)

        # step 2: train the network
        self.timesteps_so_far += self.args.timesteps_per_batch
        for i_epochs in range(self.args.optim_epochs +
                              self.args.extra_vf_optim_epochs):

            if self.args.nervenetplus:
                minibatch_id_candidate = list(
                    range(len(self.nervenetplus_batch_pos))
                )
                self._npr.shuffle(minibatch_id_candidate)
                # make sure that only timesteps per batch is used
                minibatch_id_candidate = \
                    minibatch_id_candidate[: int(self.args.timesteps_per_batch /
                                                 self.args.gnn_num_prop_steps)]
            else:
                minibatch_id_candidate = list(
                    range(
                        feed_dict[self.action_placeholder].shape[0]
                    )
                )
                self._npr.shuffle(minibatch_id_candidate)
                minibatch_id_candidate[: self.args.timesteps_per_batch]
            current_id = 0

            surrogate_epoch, kl_epoch, entropy_epoch, vf_epoch, weight_epoch = \
                [], [], [], [], []
            while current_id + self.args.optim_batch_size <= \
                    len(minibatch_id_candidate) and current_id >= 0:

                # fetch the minidata batch
                sub_feed_dict, current_id, minibatch_id_candidate = \
                    self.construct_minibatchFeeddict_from_feeddict(
                        feed_dict, minibatch_id_candidate, current_id,
                        self.args.optim_batch_size,
                        is_all_feed=self.args.minibatch_all_feed,
                        nervenetplus_batch_pos=self.nervenetplus_batch_pos
                    )

                if i_epochs < self.args.optim_epochs:
                    # train for one iteration in this epoch
                    _, i_surrogate_mini, i_kl_mini, i_entropy_mini, \
                        i_weight_mini = self.session.run(
                            [self.update_op, self.surr, self.kl, self.ent,
                                self.weight_decay_loss],
                            feed_dict=sub_feed_dict
                        )

                    # train the value network with fixed network coeff
                    _, i_vf_mini = self.session.run(
                        [self.update_vf_op, self.vf_loss],
                        feed_dict=sub_feed_dict
                    )
                    surrogate_epoch.append(i_surrogate_mini)
                    kl_epoch.append(i_kl_mini)
                    entropy_epoch.append(i_entropy_mini)
                    vf_epoch.append(i_vf_mini)
                    weight_epoch.append(i_weight_mini)
                else:
                    # only train the value function, might be unstable if share
                    # the value network and policy network
                    _, i_vf_mini = self.session.run(
                        [self.update_vf_op, self.vf_loss],
                        feed_dict=sub_feed_dict
                    )
                    vf_epoch.append(i_vf_mini)

            if i_epochs < self.args.optim_epochs:
                surrogate_epoch = np.mean(surrogate_epoch)
                kl_epoch = np.mean(kl_epoch)
                entropy_epoch = np.mean(entropy_epoch)
                vf_epoch = np.mean(vf_epoch)
                weight_epoch = np.mean(weight_epoch)
            else:
                surrogate_epoch = -0.1
                kl_epoch = -0.1
                entropy_epoch = -0.1
                weight_epoch = -0.1
                vf_epoch = np.mean(vf_epoch)

            # if we use kl_penalty, we will do early stopping if needed
            if self.args.use_kl_penalty:
                assert self.args.minibatch_all_feed, logger.error(
                    'KL penalty not available for epoch minibatch training'
                )
                if kl_epoch > 4 * self.args.target_kl and \
                        self.args.minibatch_all_feed:
                    break
        if self.args.nervenetplus:
            i_surrogate_total = surrogate_epoch
            i_kl_total = kl_epoch
            i_entropy_total = entropy_epoch
            i_vf_total = vf_epoch
            i_weight_total = weight_epoch
        else:
            i_surrogate_total, i_kl_total, i_entropy_total, \
                i_vf_total, i_weight_total = self.session.run(
                    [self.surr, self.kl, self.ent,
                        self.vf_loss, self.weight_decay_loss],
                    feed_dict=feed_dict
                )

        # step 3: update the hyperparameters of updating
        self.update_adaptive_hyperparams(kl_epoch, i_kl_total)

        # step 4: record the stats
        stats = {}

        episoderewards = np.array(
            [path["rewards"].sum() for path in paths]
        )
        stats["avg_reward"] = episoderewards.mean()
        stats["entropy"] = i_entropy_total
        stats["kl"] = i_kl_total
        stats["surr_loss"] = i_surrogate_total
        stats["vf_loss"] = i_vf_total
        stats["weight_l2_loss"] = i_weight_total
        stats['learning_rate'] = self.current_lr

        if self.args.use_kl_penalty:
            stats['kl_lambda'] = self.current_kl_lambda

        self.session.run(self.iteration_add_op)

        return stats

    def update_adaptive_hyperparams(self, kl_epoch, i_kl_total):
        # update the lambda of kl divergence
        if self.args.use_kl_penalty:
            if kl_epoch > self.args.target_kl_high * self.args.target_kl:
                self.current_kl_lambda *= self.args.kl_alpha
                if self.current_kl_lambda > 30 and \
                        self.current_lr > 0.1 * self.args.lr:
                    self.current_lr /= 1.5
            elif kl_epoch < self.args.target_kl_low * self.args.target_kl:
                self.current_kl_lambda /= self.args.kl_alpha
                if self.current_kl_lambda < 1 / 30 and \
                        self.current_lr < 10 * self.args.lr:
                    self.current_lr *= 1.5

            self.current_kl_lambda = max(self.current_kl_lambda, 1 / 35.0)
            self.current_kl_lambda = min(self.current_kl_lambda, 35.0)

        # update the lr
        elif self.args.lr_schedule == 'adaptive':
            mean_kl = i_kl_total
            if mean_kl > self.args.target_kl_high * self.args.target_kl:
                self.current_lr /= self.args.lr_alpha
            if mean_kl < self.args.target_kl_low * self.args.target_kl:
                self.current_lr *= self.args.kl_alpha

            self.current_lr = max(self.current_lr, 3e-10)
            self.current_lr = min(self.current_lr, 1e-2)
        else:
            self.current_lr = self.args.lr * max(
                1.0 - float(self.timesteps_so_far) / self.args.max_timesteps,
                0.0
            )
        # logger.warning(self.current_lr)

    def restore_all(self):
        '''
            @brief: restore the parameters
        '''
        model_name = self.get_output_path(save=False)

        # load the parameters one by one
        self.policy_network.load_checkpoint(
            model_name + '_policy.npy',
            transfer_env=self.args.transfer_env,
            logstd_option=self.args.logstd_option,
            gnn_option_list=[self.args.gnn_node_option,
                             self.args.root_connection_option,
                             self.args.gnn_output_option,
                             self.args.gnn_embedding_option],
            mlp_raw_transfer=self.args.mlp_raw_transfer
        )
        self.baseline_network.load_checkpoint(model_name + '_baseline.npy')
        self.last_save_iteration = self.get_iteration_count()

    def get_output_path(self, save=True):
        if save:
            if self.args.output_dir is None:
                path = init_path.get_base_dir()
                path = os.path.abspath(path)
            else:
                path = os.path.abspath(self.args.output_dir)
            base_path = os.path.join(path, 'checkpoint')
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            model_name = os.path.join(base_path, self.get_experiment_name())
        else:
            path = self.args.ckpt_name
            model_name = path
        return model_name

    def prepared_network_feeddict(self, rollout_data):
        feed_dict = {}

        # fetch the feed_dict element for the policy network
        if self.args.use_gnn_as_policy:
            current_idx_dict = {
                'receive_idx': self.receive_idx,
                'send_idx': self.send_idx,
                'node_type_idx': self.node_type_idx,
                'inverse_node_type_idx': self.inverse_node_type_idx,
                'output_type_idx': self.output_type_idx,
                'inverse_output_type_idx': self.inverse_output_type_idx,
                'last_batch_size': self.last_batch_size
            }
            data_dict, self.nervenetplus_batch_pos = \
                gnn_io_util.prepare_network_feeddict(
                    rollout_data, self.args.use_gnn_as_policy,
                    self.baseline_network,
                    self.node_info, current_idx_dict,
                    nervenetplus=self.args.nervenetplus,
                    gnn_num_prop_steps=self.args.gnn_num_prop_steps
                )
            self.receive_idx = data_dict['receive_idx']
            self.send_idx = data_dict['send_idx']

            self.node_type_idx = data_dict['node_type_idx']
            self.inverse_node_type_idx = data_dict['inverse_node_type_idx']

            self.output_type_idx = data_dict['output_type_idx']
            self.inverse_output_type_idx = data_dict['inverse_output_type_idx']
            self.last_batch_size = data_dict['last_batch_size']

            # the feed dict element

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
                    data_dict['graph_obs'][i_node_type]
                feed_dict[self.graph_parameters_placeholder[i_node_type]] = \
                    data_dict['graph_parameters'][i_node_type]

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
                feed_dict[self.raw_obs_placeholder] = \
                    data_dict['raw_obs_placeholder']

            feed_dict[self.batch_size_float_placeholder] = \
                data_dict['batch_size_float_placeholder']

        else:
            data_dict, self.nervenetplus_batch_pos = gnn_io_util.prepare_network_feeddict(
                rollout_data, self.args.use_gnn_as_policy, self.baseline_network,
                None, None
            )
            feed_dict[self.obs_placeholder] = data_dict['obs_placeholder']

        # fetch the feed_dict element for the optimization
        feed_dict.update({
            self.batch_size_float_placeholder:
                data_dict['batch_size_float_placeholder'],
            self.action_placeholder: data_dict['action_placeholder'],
            self.advantage_placeholder: data_dict['advantage_placeholder'],
            self.oldaction_dist_mu_placeholder:
                data_dict['oldaction_dist_mu_placeholder'],
            self.oldaction_dist_logstd_placeholder:
                data_dict['oldaction_dist_logstd_placeholder'],
            self.target_return_placeholder:
                data_dict['target_return_placeholder'],
        })

        # lr rate feeddict
        if self.args.use_kl_penalty:
            feed_dict.update(
                {self.kl_lambda_placeholder: self.current_kl_lambda}
            )
        feed_dict.update({self.lr_placeholder: self.current_lr})

        if self.args.nervenetplus:
            self._input_hidden_state = \
                self.policy_network.get_input_hidden_state_placeholder()
            for node_type in self.node_info['node_type_dict']:
                feed_dict[self._input_hidden_state[node_type]] = \
                    np.concatenate([path['hidden_state'][node_type]
                                    for path in rollout_data])

        return feed_dict

    def construct_minibatchFeeddict_from_feeddict(self,
                                                  feed_dict,
                                                  minibatch_id_candidate,
                                                  current_id,
                                                  batch_size,
                                                  is_all_feed=False,
                                                  nervenetplus_batch_pos=None):
        '''
            @brief:
                construct minibatch feed_dict from the database. If we set
                minibatch_all_feed = 1, then we just feed the whole dataset
                into the feed_dict

            @elemenet to process:
                @self.batch_feed_dict_key:
                @self.graph_batch_feed_dict_key
                @self.static_feed_dict:
                @self.dynamical_feed_dict_key:
                @self.graph_index_feed_dict:
        '''
        '''
        sub_feed_dict = {}
        if not is_all_feed:

            # step 0: id validation and id picking
            assert batch_size <= len(minibatch_id_candidate), \
                logger.error('Please increse the rollout size!')
            if current_id + batch_size > len(minibatch_id_candidate):
                logger.warning('shuffling the ids')
                current_id = 0
                self._npr.shuffle(minibatch_id_candidate)
            candidate_id = minibatch_id_candidate[
                current_id: current_id + batch_size
            ]

            # step 1: gather sub feed dict by sampling from the feed_dict
            for key in self.batch_feed_dict_key:
                sub_feed_dict[key] = feed_dict[key][candidate_id]

            # step 2: update the graph_batch_feed_dict
            if self.args.use_gnn_as_policy:
                for node_type in self.node_info['node_type_dict']:
                    num_nodes = len(self.node_info['node_type_dict'][node_type])
                    graph_candidate_id = [
                        list(range(i_id * num_nodes, (i_id + 1) * num_nodes))
                        for i_id in candidate_id
                    ]

                    # flatten the ids
                    graph_candidate_id = sum(graph_candidate_id, [])
                    for key in self.graph_batch_feed_dict_key:
                        sub_feed_dict[key[node_type]] = \
                            feed_dict[key[node_type]][graph_candidate_id]

            # step 3: static elements which are invariant to batch_size
            sub_feed_dict.update(self.static_feed_dict)

            # step 4: update dynamical elements
            for key in self.dynamical_feed_dict_key:
                sub_feed_dict[key] = feed_dict[key]

            # step 6: get the graph index feed_dict
            if self.args.use_gnn_as_policy:
                sub_feed_dict.update(self.graph_index_feed_dict)

            # step 5: update current id
            current_id = current_id + batch_size  # update id
        else:
            # feed the whole feed_dictionary into the network
            sub_feed_dict = feed_dict
            current_id = -1  # -1 means invalid
        '''

        sub_feed_dict = {}
        if not is_all_feed or self.args.nervenetplus:

            if is_all_feed and self.args.nervenetplus:
                assert nervenetplus_batch_pos is None
                assert False
                candidate_id = nervenetplus_batch_pos

            elif not is_all_feed and not self.args.nervenetplus:
                assert batch_size <= len(minibatch_id_candidate), \
                    logger.error('Please increse the rollout size!')
                if current_id + batch_size > len(minibatch_id_candidate):
                    logger.warning('shuffling the ids')
                    current_id = 0
                    self._npr.shuffle(minibatch_id_candidate)
                candidate_id = minibatch_id_candidate[
                    current_id: current_id + batch_size
                ]

            elif not is_all_feed and self.args.nervenetplus:
                assert batch_size <= len(minibatch_id_candidate), \
                    logger.error('Please increse the rollout size!')
                if current_id + batch_size > len(minibatch_id_candidate):
                    logger.warning('shuffling the ids')
                    current_id = 0
                    self._npr.shuffle(minibatch_id_candidate)

                # from util import fpdb; fpdb.fpdb().set_trace()
                nervenetplus_candidate_id = [
                    nervenetplus_batch_pos[i_id] for i_id in
                    minibatch_id_candidate[
                        current_id:
                        current_id + int(batch_size /
                                         self.args.gnn_num_prop_steps)
                    ]
                ]
                # nervenetplus_candidate_id = []
                candidate_id = []

                for chosen_pos in nervenetplus_candidate_id:
                    candidate_id.extend(
                        range(chosen_pos,
                              chosen_pos + self.args.gnn_num_prop_steps)
                    )
                # from util import fpdb; fpdb.fpdb().set_trace()

            # step 0: nervenet util
            if self.args.nervenetplus:
                # from util import fpdb; fpdb.fpdb().set_trace()
                for node_type in self.node_info['node_type_dict']:
                    node_num = len(self.node_info['node_type_dict'][node_type])
                    node_type_candidate_id = []
                    for i_id in nervenetplus_candidate_id:
                        node_type_candidate_id.extend(
                            list(range(i_id * node_num, (i_id + 1) * node_num))
                        )
                    # print node_type_candidate_id
                    # from util import fpdb; fpdb.fpdb().set_trace()
                    self._input_hidden_state = \
                        self.policy_network.get_input_hidden_state_placeholder()
                    sub_feed_dict[self._input_hidden_state[node_type]] = \
                        feed_dict[self._input_hidden_state[node_type]][
                        node_type_candidate_id
                    ]
                    # from util import fpdb; fpdb.fpdb().set_trace()

            # step 1: gather sub feed dict by sampling from the feed_dict
            for key in self.batch_feed_dict_key:
                sub_feed_dict[key] = feed_dict[key][candidate_id]

            # step 2: update the graph_batch_feed_dict
            if self.args.use_gnn_as_policy:
                for node_type in self.node_info['node_type_dict']:
                    num_nodes = len(self.node_info['node_type_dict'][node_type])
                    graph_candidate_id = [
                        list(range(i_id * num_nodes, (i_id + 1) * num_nodes))
                        for i_id in candidate_id
                    ]

                    # flatten the ids
                    graph_candidate_id = sum(graph_candidate_id, [])
                    for key in self.graph_batch_feed_dict_key:
                        sub_feed_dict[key[node_type]] = \
                            feed_dict[key[node_type]][graph_candidate_id]

            # step 3: static elements which are invariant to batch_size
            sub_feed_dict.update(self.static_feed_dict)

            # step 4: update dynamical elements
            for key in self.dynamical_feed_dict_key:
                sub_feed_dict[key] = feed_dict[key]

            # step 6: get the graph index feed_dict
            if self.args.use_gnn_as_policy:
                sub_feed_dict.update(self.graph_index_feed_dict)

            # step 5: update current id
            if not self.args.nervenetplus:
                current_id = current_id + batch_size  # update id
            else:
                current_id = \
                    current_id + int(batch_size / self.args.gnn_num_prop_steps)

        else:
            # feed the whole feed_dictionary into the network
            sub_feed_dict = feed_dict
            current_id = -1  # -1 means invalid
        # from util import fpdb; fpdb.fpdb().set_trace()

        return sub_feed_dict, current_id, minibatch_id_candidate

    def prepare_feed_dict_map(self):
        '''
            @brief:
                When trying to get the sub diction in
                @construct_minibatchFeeddict_from_feeddict, some key are just
                directly transferable. While others might need some other work

                @1. feed_dict for trpo or ppo update

                    # baseline function

                @2. feed_dict for generating the policy action

                    # for ggnn only
                @3. feed_dict for baseline if baseline is a fc-policy and policy
                    is a ggnn policy

            @return:
                @self.batch_feed_dict_key:
                    Shared between the fc policy network and ggnn network.
                    Most of them are only used for the update.

                        @self.action_placeholder
                        @self.advantage_placeholder
                        @self.oldaction_dist_mu_placeholder
                        @self.oldaction_dist_logstd_placeholder

                        (if use fc policy)
                        @self.obs_placeholder

                        (if use_ggn and baseline not gnn)
                        @self.raw_obs_placeholder

                        (if use tf baseline)
                        @self.target_return_placeholder (for ppo only)

                @self.graph_batch_feed_dict_key
                    Used by the ggnn. This feed_dict key list is a little bit
                    different from @self.batch_feed_dict_key if we want to do
                    minibatch

                        @self.graph_obs_placeholder
                        @self.graph_parameters_placeholder

                @self.static_feed_dict:
                    static elements that are set by optim parameters. These
                    parameters are set differently between minibatch_all_feed
                    equals 0 / equals 1

                        @self.batch_size_float_placeholder
                        @self.batch_size_int_placeholder

                @self.dynamical_feed_dict_key:
                    elements that could be changing from time to time

                        @self.kl_lambda_placeholder
                        @self.lr_placeholder

                @self.graph_index_feed_dict:
                    static index for the ggnn.

                        @self.receive_idx_placeholder
                        @self.inverse_node_type_idx_placeholder
                        @self.output_idx_placeholder
                        @self.send_idx_placeholder[i_edge]
                        @self.node_type_idx_placeholder[i_node_type]
        '''
        # step 1: gather the key for batch_feed_dict
        self.batch_feed_dict_key = [
            self.action_placeholder,
            self.advantage_placeholder,
            self.oldaction_dist_mu_placeholder,
            self.oldaction_dist_logstd_placeholder
        ]

        if not self.args.use_gnn_as_policy:
            self.batch_feed_dict_key.append(self.obs_placeholder)

        if self.args.use_gnn_as_policy and not self.args.use_gnn_as_value:
            self.batch_feed_dict_key.append(self.raw_obs_placeholder)

        self.batch_feed_dict_key.append(self.target_return_placeholder)

        # step 2: gather the graph batch feed_dict
        self.graph_batch_feed_dict_key = []
        if self.args.use_gnn_as_policy:
            self.graph_batch_feed_dict_key.extend(
                [self.graph_obs_placeholder, self.graph_parameters_placeholder]
            )

        # step 2: gather the static feed_dictionary
        self.static_feed_dict = {
            self.batch_size_float_placeholder:
                np.array(float(self.args.optim_batch_size))
        }
        if self.args.use_gnn_as_policy:
            if not self.args.nervenetplus:
                self.static_feed_dict.update(
                    {self.batch_size_int_placeholder: self.args.optim_batch_size}
                )
            else:
                self.static_feed_dict.update(
                    {self.batch_size_int_placeholder:
                        int(self.args.optim_batch_size / self.args.gnn_num_prop_steps)}
                )

        # step 3: gather the dynamical feed_dictionary
        self.dynamical_feed_dict_key = []
        if self.args.use_kl_penalty:
            self.dynamical_feed_dict_key.append(self.kl_lambda_placeholder)
        self.dynamical_feed_dict_key.append(self.lr_placeholder)

        # step 4: gather the graph_index feed_dict
        if self.args.use_gnn_as_policy:
            # construct a dummy obs to pass the batch size info
            if self.args.nervenetplus:
                assert self.args.gnn_num_prop_steps % \
                    self.args.gnn_num_prop_steps == 0
                dummy_obs = np.zeros(
                    [int(self.args.optim_batch_size /
                         self.args.gnn_num_prop_steps),
                     10])
            else:
                dummy_obs = np.zeros([self.args.optim_batch_size, 10])
            # print dummy_obs.shape
            node_info = self.policy_network.get_node_info()

            # get the index for minibatches
            _, _, receive_idx, send_idx, \
                node_type_idx, inverse_node_type_idx, \
                output_type_idx, inverse_output_type_idx, _ = \
                graph_data_util.construct_graph_input_feeddict(
                    node_info,
                    dummy_obs, -1, -1, -1, -1, -1, -1, -1,
                    request_data=['idx']
                )

            self.graph_index_feed_dict = {
                self.receive_idx_placeholder:
                    receive_idx,
                self.inverse_node_type_idx_placeholder:
                    inverse_node_type_idx,
                self.inverse_output_type_idx_placeholder:
                    inverse_output_type_idx
            }

            # append the send idx
            for i_edge in node_info['edge_type_list']:
                self.graph_index_feed_dict[
                    self.send_idx_placeholder[i_edge]
                ] = send_idx[i_edge]

            # append the node type idx
            for i_node_type in node_info['node_type_dict']:
                self.graph_index_feed_dict[
                    self.node_type_idx_placeholder[i_node_type]
                ] = node_type_idx[i_node_type]

            # append the node type idx
            for i_output_type in node_info['output_type_dict']:
                self.graph_index_feed_dict[
                    self.output_type_idx_placeholder[i_output_type]
                ] = output_type_idx[i_output_type]

    def get_species_info(self, training_stats=None, received_data=None):
        if received_data is not None:
            prt_id = received_data['PrtID']
            spc_id = received_data['SpcID']
            '''
            debug_info = received_data['debug_info'] \
                if 'debug_info' in received_data else None
            '''
        else:
            prt_id = -1  # the first generation
            spc_id = -1
            # debug_info = None

        if self.args.use_gnn_as_policy:
            node_info = self.policy_network.get_node_info()
        else:
            node_info = None

        species_data = {
            'policy_weights': self.get_policy(),
            'baseline_weights': self.baseline_network.get_weight_dict(),
            'running_mean_info': self.running_mean_info,
            'var_name': [var.name for var in self.policy_var_list],

            'stats': training_stats,
            'start_time': self.start_time,
            'rollout_time': time.time() - self.start_time,

            'xml_str': self.xml_str,
            'adj_matrix': self.adj_matrix,
            'node_attr': self.node_attr,
            'node_info': node_info,

            'SpcID': spc_id,
            'PrtID': prt_id,
            'agent_id': self.agent_id,
            'LastRwd': self.last_average_reward,
            'AvgRwd': np.mean(self.end_average_reward),

            'env_name': self.args.task,
            'action_size': self.action_size,
            'observation_size': self.observation_size,

            'debug_info': None,
            'is_dead': self.is_dead_species,  # kill species with mjWARN_BADQACC
            'lr': self.current_lr
        }
        if self.args.new_species_struct:
            species_data['species'] = self.species
        if self.args.brute_force_search and training_stats is not None:
            species_data['brute_reward'] = training_stats['brute_reward']
        return species_data
