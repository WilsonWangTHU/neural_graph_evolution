# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper for the fishes!
# -----------------------------------------------------------------------------
import init_path
from util import dm_control_util
from util import model_gen_util
import os
import numpy as np
import random
from env.dm_env_wrapper import dm_env_wrapper
import collections
from env.dm_env_wrapper import NUM_EPISODE_RECORED
# import pdb


class dm_fish3d_wrapper(dm_env_wrapper):
    '''
        @brief:
            Several types of tasks are allowed here:
            ['fish3d-easyswim', 'fish3d-swim', 'fish3d-speed',
             'evofish3d-easyswim', 'evofish3d-swim', 'evofish3d-speed']
    '''

    def __init__(self, args, rand_seed, monitor, width=480, height=480):
        self.width = width
        self.height = height
        self.task = args.task
        self.args = args
        assert 'fish3d' in self.task
        self.is_evo = 'evo' in self.task
        if 'easyswim' in self.task:
            self.target_angle = args.fish_target_angle

        from dm_control import suite
        self.env = suite.load(
            domain_name='fish', task_name='swim',
            task_kwargs={'random': rand_seed}
        )
        self._base_path = init_path.get_abs_base_dir()

        self.load_xml(os.path.join(self._base_path, 'env', 'assets/fish3d.xml'))
        self.set_get_observation()  # overwrite the original get_ob function
        self.set_get_reward()  # overwrite the original reward function
        self._JOINTS = ['tail1',
                        'tail_twist',
                        'tail2',
                        'finright_roll',
                        'finright_pitch',
                        'finleft_roll',
                        'finleft_pitch']

        # save the video
        self._monitor = monitor
        self._current_episode = 0
        if self._monitor:
            self.init_save(args)

    def load_xml(self, xml_path=None, xml_str=None):
        from dm_control.mujoco import engine
        if xml_str is None:
            self.env._physics = engine.Physics.from_xml_path(xml_path)
        else:
            from dm_control.suite import common
            # from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
            self.env._physics = \
                engine.Physics.from_xml_string(xml_str, common.ASSETS)

    def reset(self):
        self.env._reset_next_step = False
        if hasattr(self, '_qpos_data') and len(self._qpos_data) > 0:
            # save the videos (qpos actually)
            self.save_qpos()

        self._current_episode += 1
        self._timestep = 0

        with self.env.physics.reset_context():
            # randomize the joints
            for joint in self._JOINTS:
                self.env.physics.named.data.qpos[joint] = \
                    self.env.task.random.uniform(-.2, .2)

            if 'easyswim' in self.task:
                # 'y is the direction'
                # theta = self.env.task.random.uniform(-3.14 / 2, 3.14 / 2)
                # theta = self.env.task.random.uniform(-3.14 / 6, 3.14 / 6)
                theta = self.env.task.random.uniform(-self.target_angle,
                                                     self.target_angle)
                length = self.env.task.random.uniform(0.6, 0.8)
                self.env.physics.named.model.geom_pos['target', 'x'] = \
                    np.sin(theta) * length
                self.env.physics.named.model.geom_pos['target', 'y'] = \
                    np.cos(theta) * length

            elif 'curlswim' in self.task:
                ''' try to set the target
                '''
                self.sin_sign = random.choice([-1, 1])
                self.env.physics.named.model.geom_pos['target', 'x'] = 0
                self.env.physics.named.model.geom_pos['target', 'y'] = 0

            elif 'circleswim' in self.task:
                self.circle_sign = random.choice([-1, 1])
                theta = np.pi / 12
                self.target_y = np.sin(theta) * self.args.fish_circle_radius
                self.target_x = np.cos(theta) * self.args.fish_circle_radius + \
                    -1 ** self.circle_sign * self.args.fish_circle_radius

                self.env.physics.named.model.geom_pos['target', 'x'] = \
                    self.target_x
                self.env.physics.named.model.geom_pos['target', 'y'] = \
                    self.target_y

            elif 'swim' in self.task:
                self.env.physics.named.model.geom_pos['target', 'x'] = \
                    self.env.task.random.uniform(-.4, .4)
                self.env.physics.named.model.geom_pos['target', 'y'] = \
                    self.env.task.random.uniform(-.4, .4)
                self.env.physics.named.data.qpos['rootz'] = \
                    self.env.task.random.uniform(-3.14, 3.14)

            elif 'speed' in self.task:
                # hide the objects
                self.env.physics.named.model.geom_rgba['target', 3] = 0

            else:
                assert False  # invalid task

        self.env.physics.after_reset()

        # note that the observation is now changed
        reset_return = dm_control_util.vectorize_ob(
            self.env.task.get_observation(self.env.physics)
        )
        return reset_return

    def step(self, action):
        step_return = dm_env_wrapper.step(self, action)
        self._timestep += 1

        if 'curlswim' in self.task:
            x = self._timestep * self.args.fish_target_speed
            self.target_y = x + 0.15
            self.target_x = self.sin_sign * self.args.fish_target_bound * \
                np.sin(self.args.fish_target_var * x)

            self.env.physics.named.model.geom_pos['target', 'x'] = \
                self.target_x
            self.env.physics.named.model.geom_pos['target', 'y'] = \
                self.target_y

            done = step_return[2]

        elif 'circleswim' in self.task:
            theta = self._timestep * self.args.fish_target_speed + np.pi / 12
            self.target_y = np.sin(theta) * self.args.fish_circle_radius
            if self.circle_sign == 1:
                self.target_x = np.cos(theta) * \
                    self.args.fish_circle_radius - self.args.fish_circle_radius
            elif self.circle_sign == -1:
                self.target_x = np.cos(-theta + np.pi) * \
                    self.args.fish_circle_radius + self.args.fish_circle_radius

            self.env.physics.named.model.geom_pos['target', 'x'] = \
                self.target_x
            self.env.physics.named.model.geom_pos['target', 'y'] = \
                self.target_y

            done = step_return[2]

        elif 'swim' in self.task:
            data = self.env.physics.named.data
            mouth_to_target_global = \
                data.geom_xpos['target'] - data.geom_xpos['mouth']
            mouth_to_target = mouth_to_target_global.dot(
                data.geom_xmat['mouth'].reshape(3, 3)
            )

            in_target = np.linalg.norm(mouth_to_target)
            done = step_return[2] or in_target < 0.045
        elif 'speed' in self.task:
            done = step_return[2]
        else:
            assert False  # invalid task

        # save the position of the target
        if self._monitor and \
                self._current_episode % self._video_freq < NUM_EPISODE_RECORED:

            self._qpos_data[-1] = np.array(
                self._qpos_data[-1].tolist() +
                [self.env.physics.named.model.geom_pos['target', 'x'],
                 self.env.physics.named.model.geom_pos['target', 'y']]
            )
        return step_return[0], step_return[1], done, step_return[3]

    def set_get_observation(self):

        def fish3d_get_observation(physics):
            obs = collections.OrderedDict()

            # from util import fpdb; fpdb = fpdb.fpdb(); fpdb.set_trace()
            obs['joint_angles'] = self.env.physics.named.data.qpos[self._JOINTS]
            obs['velocity'] = self.env.physics.data.qvel[:]
            if 'swim' in self.task:
                # TODO: the root in JOINTS list
                data = self.env.physics.named.data
                mouth_to_target_global = \
                    data.geom_xpos['target'] - data.geom_xpos['mouth']
                obs['target'] = mouth_to_target_global.dot(
                    data.geom_xmat['mouth'].reshape(3, 3)
                )
            else:
                assert 'speed' in self.task
                # TODO: provide rotation angle of Z axis
            return obs

        self.env.task.get_observation = fish3d_get_observation

    def set_get_reward(self):

        def fish3d_get_reward(physics):
            # the distance to the target

            if 'swim' in self.task:
                data = physics.named.data
                mouth_to_target_global = \
                    data.geom_xpos['target'] - data.geom_xpos['mouth']
                mouth_to_target = mouth_to_target_global.dot(
                    data.geom_xmat['mouth'].reshape(3, 3)
                )
                # distance reward in [-0.8, 0.0]
                distance_reward = - np.linalg.norm(mouth_to_target)

                '''
                # direction reward in [-3.14/6, 0]
                target_pos = [
                    physics.named.model.geom_pos['target', 'x'],
                    physics.named.model.geom_pos['target', 'y']
                ]
                target_pos = target_pos / (1e-8 + np.linalg.norm(target_pos))
                fish_pos = [
                    np.sin(physics.named.data.qvel['rooty'][0]),
                    np.cos(physics.named.data.qvel['rooty'][0])
                ]
                cosine = \
                    fish_pos[0] * target_pos[0] + fish_pos[1] * target_pos[1]
                direction_reward = -np.abs(np.arccos(cosine))
                '''
                direction_reward = 0.0
                reward = direction_reward + distance_reward
            elif 'speed' in self.task:
                reward = physics.named.data.qvel['rooty'][0]

            return reward

        self.env.task.get_reward = fish3d_get_reward

    '''
    def parse_fish_type(self, task_name):
        return

    def get_obervation_size(self):
        return

    def get_action_size(self):
        return
    '''


class dm_evofish3d_wrapper(dm_fish3d_wrapper):
    '''
    '''

    def __init__(self, args, rand_seed, monitor, adj_matrix=None,
                 xml_str=None, xml_path=None, width=480, height=480):
        dm_fish3d_wrapper.__init__(
            self, args, rand_seed, monitor, width, height
        )
        if xml_str is None and xml_path is None:
            self.load_xml(
                os.path.join(self._base_path, 'env', 'assets/fish3d.xml')
            )
        else:
            assert not(xml_str is not None and xml_path is not None)
            self.load_xml(xml_path=xml_path, xml_str=xml_str)
            self._JOINTS = self.get_joints(adj_matrix)

        self.set_get_observation()  # overwrite the original get_ob function
        self.set_get_reward()  # overwrite the original reward function

    def get_joints(self, adj_matrix):
        ''' based on the current connection, create the corresponding _JOINTS
        '''
        connection_list = []
        connection_list = model_gen_util.dfs_order(adj_matrix)

        if 'swim' in self.task:
            joint_list = []
        elif 'speed' in self.task:
            joint_list = ['rootz']
        else:
            raise NotImplementedError

        for item in connection_list:
            joint_list = joint_list + [item + '_x', item + '_y', item + '_z']

        return joint_list

    def parse_fish_type(self, task_name):
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


FISH_TYPE = {
    'fish3d-swim': dm_fish3d_wrapper,  # only Rz, dx, dy
    'fish5d-swim': None,  # Rz, dx, dy and Ry, dz
    'fish6d-swim': None,  # Rz, Ry, Rz, dx, dy, dz

    'fish3d-easyswim': dm_fish3d_wrapper,  # only Rz, dx, dy
    'fish5d-easyswim': None,  # Rz, dx, dy and Ry, dz
    'fish6d-easyswim': None,  # Rz, Ry, Rz, dx, dy, dz

    'fish3d-speed': dm_fish3d_wrapper,  # only Rz, dx, dy
    'fish5d-speed': None,  # Rz, dx, dy and Ry, dz
    'fish6d-speed': None,  # Rz, Ry, Rz, dx, dy, dz

    'evofish3d-swim': dm_evofish3d_wrapper,  # only Rz, dx, dy
    'evofish5d-swim': None,  # Rz, dx, dy and Ry, dz
    'evofish6d-swim': None,  # Rz, Ry, Rz, dx, dy, dz

    'evofish3d-easyswim': dm_evofish3d_wrapper,  # only Rz, dx, dy
    'evofish5d-easyswim': None,  # Rz, dx, dy and Ry, dz
    'evofish6d-easyswim': None,  # Rz, Ry, Rz, dx, dy, dz
}
