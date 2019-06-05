# -----------------------------------------------------------------------------
#   @brief:
#       the agent that do the sampling of trajectories.
#   @author:
#       modified from the code of kvfran, modified by Tingwu Wang
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import gym
import time
import init_path
import os
from util import utils
from util import ob_normalizer
from util import logger
from util import parallel_util
from .agent import base_agent
from env import env_wrapper


class rollout_agent(base_agent):

    def __init__(self,
                 args,
                 observation_size,
                 action_size,
                 task_q,
                 result_q,
                 actor_id,
                 monitor,
                 name_scope='policy-actor',
                 init_filter_parameters=None):

        # the base agent
        super(rollout_agent, self).__init__(args=args,
                                            observation_size=observation_size,
                                            action_size=action_size,
                                            task_q=task_q,
                                            result_q=result_q,
                                            name_scope=name_scope,
                                            is_rollout_agent=True)
        self.allow_monitor = monitor
        self.actor_id = actor_id
        self._npr = np.random.RandomState(args.seed + actor_id)

        if init_filter_parameters is not None:
            self.ob_normalizer = ob_normalizer.normalizer(
                mean=init_filter_parameters['mean'],
                variance=init_filter_parameters['variance'],
                num_steps=init_filter_parameters['step']
            )
        else:
            self.ob_normalizer = ob_normalizer.normalizer()

        logger.info('The sampler {} is online'.format(self.actor_id))

    def run(self):
        self.build_model()

        while True:

            next_task = self.task_q.get(block=True)

            # Collect rollouts
            if next_task[0] == parallel_util.START_SIGNAL:
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)

            # Set normalizer
            elif next_task[0] == parallel_util.AGENT_SYNCHRONIZE_FILTER:
                self.ob_normalizer.set_parameters(
                    next_task[1]['mean'],
                    next_task[1]['variance'],
                    next_task[1]['step']
                )
                time.sleep(0.001)  # yield the process
                self.task_q.task_done()

            # Set parameters of the actor policy
            elif next_task[0] == parallel_util.AGENT_SET_POLICY_WEIGHTS:

                self.set_policy(next_task[1])
                time.sleep(0.001)  # yield the process
                self.task_q.task_done()

            # Kill all the thread
            elif next_task[0] == parallel_util.END_ROLLOUT_SIGNAL or \
                    next_task[0] == parallel_util.END_SIGNAL:
                logger.info("kill message for sampler {}".format(self.actor_id))
                self.task_q.task_done()
                break
            else:
                logger.error(
                    'Invalid task type {} for agents'.format(next_task[0]))
        return

    def wrap_env_monitor(self):
        if self.allow_monitor:
            def video_callback(episode):
                return episode % self.args.video_freq < 6
            if self.args.output_dir is None:
                base_path = init_path.get_base_dir()
            else:
                base_path = self.args.output_dir

            path = os.path.join(
                base_path, 'video', self.args.task + '_' + self.args.time_id
            )
            path = os.path.abspath(path)
            if not os.path.exists(path):
                os.makedirs(path)
            self.env = gym.wrappers.Monitor(
                self.env, path, video_callable=video_callback)

    def build_env(self):

        if not self.args.dm:
            # the flow when using gym
            # init the environments
            self.env = gym.make(self.args.task)
            self.sub_task_list = self.args.task
            self.env.seed(self._npr.randint(0, 999999))
            self.wrap_env_monitor()  # wrap the environment with monitor if needed
        else:
            # using dm_control package
            env, is_dm_env = env_wrapper.make_env(self.args,
                                                  self._npr.randint(0, 999999),
                                                  self.allow_monitor)
            self.env = env
            self.is_dm_env = is_dm_env
        return

    def build_model(self):
        # build the environments for the agent
        self.build_env()
        self.build_session()
        self.build_policy_network()
        self.fetch_policy_info()

        self.session.run(tf.global_variables_initializer())

        self.set_policy = utils.SetPolicyWeights(self.session,
                                                 self.policy_var_list)
        self.get_policy = utils.GetPolicyWeights(self.session,
                                                 self.policy_var_list)
        if self.args.nervenetplus:
            self.node_info = self.policy_network.get_node_info()

    def rollout(self):

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

        # start the env (reset the environment) based on package used
        raw_ob = self.env.reset()
        # from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
        ob = self.ob_normalizer.filter(raw_ob)

        # run the game
        while True:
            # generate the policy
            if self.args.nervenetplus:
                for node_type in self.node_info['node_type_dict']:
                    hidden_state[node_type].append(
                        self.current_state[node_type]
                    )
                    # logger.error(node_type)
                    # logger.error(self.current_state[node_type].shape)

            action, action_dist_mu, action_dist_logstd = self.act(ob)

            # record the stats
            obs.append(ob)
            raw_obs.append(raw_ob)
            actions.append(action)
            action_dists_mu.append(action_dist_mu)
            action_dists_logstd.append(action_dist_logstd)

            # take the action
            res = self.env.step(action)
            raw_ob = res[0]
            reward = res[1]
            done = res[2]

            ob = self.ob_normalizer.filter(raw_ob)
            rewards.append((reward))

            if done:  # terminated
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
        feed_dict, _ = self.prepared_policy_network_feeddict(obs)

        if self.args.nervenetplus:
            # append the hidden state into the input
            self._input_hidden_state = \
                self.policy_network.get_input_hidden_state_placeholder()
            for node_type in self.node_info['node_type_dict']:
                feed_dict[self._input_hidden_state[node_type]] = \
                    self.current_state[node_type]

            results = self.session.run(
                [self.action_dist_mu, self.action_dist_logstd] +
                self.policy_network.get_output_hidden_state_list(),
                feed_dict=feed_dict
            )
            action_dist_mu = results[0]
            # print(action_dist_mu.shape)
            # (results[2])
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
