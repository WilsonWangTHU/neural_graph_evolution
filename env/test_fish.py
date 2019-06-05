# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper
# -----------------------------------------------------------------------------
import init_path
from config.config import get_config
import os
# os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
import fish_env_wrapper
# from util import dm_control_util
import numpy as np


if __name__ == '__main__':
    '''
        @brief: test the environments
        @example:
            1. test the gym environment
                python test_env_wrapper.py
                    --task Reacher-v1
                    --monitor 0
                    --test_env 1

            1. test the dm environment
                python test_env_wrapper.py
                    --task Reacher-v1
                    --monitor 0
                    --test_env 1
    '''
    # os.environ['DISABLE_MUJOCO_RENDERING'] = '1'

    args = get_config()
    print('Base dir: {}'.format(init_path.get_abs_base_dir()))
    if not args.monitor:
        os.environ['DISABLE_MUJOCO_RENDERING'] = '1'

    # make the environment
    env = fish_env_wrapper.dm_fish3d_wrapper(
        args, 1, args.monitor
    )
    action_size = env.env.action_spec().shape[0]

    for num_episode in range(30):
        # test for one episode
        ob = env.reset()
        print("reset return - ob size: {}".format(ob.shape))

        for _ in range(1000):
            # import pdb; pdb.set_trace()
            ob, reward, done, _ = \
                env.step((np.random.rand(action_size) - 0.5) * 2)
            print(
                "action_size:{}, ob_size:{}, reward:{}, done:{}".format(
                    action_size, ob.shape, reward, done
                )
            )
            print("ob: {}\n".format(ob))
            if args.monitor:
                env.render()
            if done:
                break
            break

    ob = env.reset()
