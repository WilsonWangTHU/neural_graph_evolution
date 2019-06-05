'''
'''

import init_path
from util import dm_control_util
from util import model_gen_util
import os
import numpy as np
from env.dm_env_wrapper import dm_env_wrapper
import collections
# from env.dm_env_wrapper import NUM_EPISODE_RECORED


class dm_walker_wrapper(dm_env_wrapper):
    '''
    '''

    def __init__(self, args, rand_seed, monitor, width=480, height=480):
        self.width = width
        self.height = height
        self.task = args.task
        assert 'walker' in self.task or 'hopper' in self.task or 'cheetah' in self.task
        self.args = args
        self.is_evo = 'evo' in self.task

        from dm_control import suite
        self.env = suite.load(
            domain_name='walker', task_name='walk',
            task_kwargs={'random': rand_seed}
        )
        self._base_path = init_path.get_abs_base_dir()

        self.load_xml(os.path.join(self._base_path, 'env', 'assets/walker.xml'))
        self.set_get_observation()  # overwrite the original get_ob function
        self.set_get_reward()       # overwrite the original reward function
        self._JOINTS = ['right_hip',
                        'right_knee',
                        'right_ankle',
                        'left_hip',
                        'left_knee',
                        'left_ankle']

        # save the video
        self._monitor = monitor
        self._current_episode = 0
        if self._monitor:
            self.init_save(args)

    def load_xml(self, xml_path=None, xml_str=None):
        ''' directly taken from fish_env_wrapper
        '''
        from dm_control.mujoco import engine
        if xml_str is None:
            self.env._physics = engine.Physics.from_xml_path(xml_path)
        else:
            from dm_control.suite import common
            # from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
            self.env._physics = \
                engine.Physics.from_xml_string(xml_str, common.ASSETS)
        pass

    def reset(self):

        self.env._reset_next_step = False
        if hasattr(self, '_qpos_data') and len(self._qpos_data) > 0:
            # save the videos (qpos actually)
            self.save_qpos()

        self._current_episode += 1

        with self.env.physics.reset_context():
            # randomize the joints
            if 'walker' in self.args.task and self.args.optimize_creature:
                pass
            elif 'hopper' in self.args.task and self.args.optimize_creature:
                pass
            else:
                # walker
                self.env.physics.named.data.qpos['rooty'] = 90

            for joint in self._JOINTS:
                self.env.physics.named.data.qpos[joint] = \
                    self.env.task.random.uniform(-.2, .2)

            # if 'walk' in self.task:
            #     self.env.physics.named.model.geom_rgba['target', 3] = 0
            # elif 'run' in self.task:
            #     # hide the objects
            #     self.env.physics.named.model.geom_rgba['target', 3] = 0
            # else:
            #     assert False  # invalid task

        self.env.physics.after_reset()

        # note that the observation is now changed
        reset_return = dm_control_util.vectorize_ob(
            self.env.task.get_observation(self.env.physics)
        )
        return reset_return

    def step(self, action):

        step_return = dm_env_wrapper.step(self, action)
        if 'evohopper-speed' in self.task:
            done = step_return[2]
            height = (self.env.physics.named.data.xipos['torso', 'z'] -
                      min(self.env.physics.named.data.xipos[1:, 'z']))
            # from util import fpdb; fpdb.fpdb().set_trace()
            if height < 0.6:
                # print(height)
                done = True

            reward = step_return[1]
            reward = reward - self.args.walker_ctrl_coeff * \
                np.square(action).sum()
        elif 'speed' in self.task:
            done = step_return[2]

            reward = step_return[1]
            reward = reward - self.args.walker_ctrl_coeff * \
                np.square(action).sum()
        elif 'walk' in self.task:
            raise NotImplementedError
            done = step_return[2]
        elif 'run' in self.task:
            raise NotImplementedError
            done = step_return[2]

        else:
            assert False  # invalid task

        return step_return[0], reward, done, step_return[3]

    def set_get_observation(self):
        def walker_get_observation(physics):
            obs = collections.OrderedDict()

            # from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
            # 1
            obs['height'] = self.env.physics.named.data.xpos['torso', 'z']

            # 2 * N
            obs['orientations'] = \
                self.env.physics.named.data.xmat[1:, ['xx', 'xz']].ravel()

            # 3 (root) + (N - 1) (other body parts)
            obs['velocity'] = self.env.physics.data.qvel[:]

            # 2 * (N - 1)
            obs['touch'] = np.log1p(self.env.physics.data.sensordata[:])

            return obs

        self.env.task.get_observation = walker_get_observation

    def set_get_reward(self):

        def walker_get_reward(physics):
            # the distance to the target
            if 'evohopper-speed' in self.task:
                from dm_control.utils import rewards
                reward = physics.named.data.subtree_linvel['torso', 'x']
                reward = np.clip(reward, -10, 10)
                if reward > 0:
                    # from util import fpdb; fpdb.fpdb().set_trace()
                    height = (physics.named.data.xipos['torso', 'z'] -
                              min(physics.named.data.xipos[1:, 'z']))
                    height_coeff = rewards.tolerance(height, (0.6, 2))
                    reward *= height_coeff
                    # print(height_coeff)

            elif 'speed' in self.task:
                reward = physics.named.data.subtree_linvel['torso', 'x']
                reward = np.clip(reward, -10, 10)

            else:
                raise NotImplementedError

            return reward

        self.env.task.get_reward = walker_get_reward


class dm_evowalker_wrapper(dm_walker_wrapper):

    def __init__(self, args, rand_seed, monitor, adj_matrix=None,
                 xml_str=None, xml_path=None, width=480, height=480):
        dm_walker_wrapper.__init__(
            self, args, rand_seed, monitor, width, height
        )
        if xml_str is None and xml_path is None:
            self.load_xml(
                os.path.join(self._base_path, 'env', 'assets/walker.xml')
            )
        else:
            assert not(xml_str is not None and xml_path is not None)
            self.load_xml(xml_path=xml_path, xml_str=xml_str)
            self._JOINTS = self.get_joints(adj_matrix)

        self.set_get_observation()  # overwrite the original get_ob function
        self.set_get_reward()  # overwrite the original reward function

    def get_joints(self, adj_matrix):

        connection_list = []
        connection_list = model_gen_util.dfs_order(adj_matrix)

        # should i be giving rootz for walk/run task
        joint_list = []

        for edge in connection_list:
            p, c = [int(x) for x in edge.split('-')]
            joint_type = adj_matrix[p, c]
            joint_axis = model_gen_util.get_encoding(joint_type)

            if joint_axis[0] == 1:
                joint_list += [edge + '_x']
            elif joint_axis[1] == 1:
                joint_list += [edge + '_y']
            elif joint_axis[2] == 1:
                joint_list += [edge + '_z']

        return joint_list

    def parse_walker_type(self, task_name):
        '''
        '''
        return

    def get_observation_size(self):
        return dm_control_util.get_ob_size(self.env)

    def get_action_size(self):
        return self.env.action_spec().shape[0]

    def new_env(self, xml_path=None, xml_str=None):
        self.load_xml(xml_path=xml_path, xml_str=xml_str)
        self.update_joint_information()
        self.set_get_reward()
        self.set_get_observation()

    def update_joint_information(self):
        # import pdb; pdb.set_trace()
        self._JOINTS = list(self.env.physics.named.qpos.axes[0].names)
        # remove the root joints
        self._JOINTS = [joint for joint in self._JOINTS if 'root' not in joint]
