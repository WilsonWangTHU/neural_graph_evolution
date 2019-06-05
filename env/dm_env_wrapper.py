# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper for dm controller
# -----------------------------------------------------------------------------
import init_path
from util import dm_control_util
import os
import numpy as np
from util import logger
# import matplotlib.pyplot as plt

NUM_EPISODE_RECORED = 6


def get_output_path(args, env_str='deepmind'):
    if args.output_dir is None:
        base_path = init_path.get_base_dir()
    else:
        base_path = args.output_dir

    is_test_env = 'test_' if args.test_env else ''
    path = os.path.join(
        base_path,
        'video',
        is_test_env + env_str,
        args.task + '_' + args.time_id
    )
    path = os.path.abspath(path)

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            logger.error(str(e))

    return path


class dm_env_wrapper(object):

    def __init__(self, args, rand_seed, monitor, width=480, height=480):
        self.width = width
        self.height = height
        task_name = dm_control_util.get_env_names(args.task)
        from dm_control import suite
        self.env = suite.load(
            domain_name=task_name[0], task_name=task_name[1],
            task_kwargs={'random': rand_seed}
        )
        self._base_path = init_path.get_abs_base_dir()
        self.NUM_EPISODE_RECORED = NUM_EPISODE_RECORED
        self._is_dirname = True

        # save the video
        self._monitor = monitor
        self._current_episode = 0
        if self._monitor:
            self.init_save(args)

    def step(self, action):
        if self._monitor and \
                self._current_episode % self._video_freq < NUM_EPISODE_RECORED:
            self._qpos_data.append(np.array(self.env.physics.data.qpos))

        step_return = self.env.step(action)
        return dm_control_util.vectorize_ob(step_return.observation), \
            step_return.reward, \
            step_return.last(), \
            step_return

    def reset(self):
        if hasattr(self, '_qpos_data') and len(self._qpos_data) > 0:
            # save the videos (qpos actually)
            self.save_qpos()

        self._current_episode += 1
        return dm_control_util.vectorize_ob(self.env.reset().observation)

    def save_qpos(self):
        # save the data
        if self._is_dirname:
            data_path = \
                os.path.join(self._output_dir, str(self._current_episode))
        else:
            data_path = \
                os.path.join(self._output_dir + str(self._current_episode))
        np.save(data_path, self._qpos_data)

        # reset the qpos
        self._qpos_data = []

    def init_save(self, args):
        self._output_dir = get_output_path(args)
        self._video_freq = args.video_freq
        self._current_episode = 0
        self._qpos_data = []

    def render(self):
        pass
        '''
        images = np.hstack(
            [self.env.physics.render(self.height, self.width, camera_id=0),
             self.env.physics.render(self.height, self.width, camera_id=1)]
        )
        # plt.imshow(images)
        # plt.pause(0.0001)  # Need min display time > 0.0.
        # plt.draw()
        '''

    def set_output_dir(self, new_output_dir):
        self._output_dir = new_output_dir
        self._is_dirname = False
