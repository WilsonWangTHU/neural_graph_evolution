# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       The environment wrapper
# -----------------------------------------------------------------------------
import init_path
from util import dm_control_util
import os
# os.environ['DISABLE_MUJOCO_RENDERING'] = '1'  # disable headless rendering
from env import dm_env_wrapper
from env import fish_env_wrapper

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
        os.makedirs(path)
    return path


def make_env(args, rand_seed, allow_monitor):
    task = args.task
    if task in fish_env_wrapper.FISH_TYPE:
        # deep-mind environments
        env = fish_env_wrapper.FISH_TYPE[task](args, rand_seed, allow_monitor)
        is_deepmind_env = True
    elif task in dm_control_util.DM_ENV_INFO:
        env = dm_env_wrapper.dm_env_wrapper(args, rand_seed, allow_monitor)
        is_deepmind_env = True

    # gym environments
    else:
        import gym
        env = gym.make(task)
        env.seed(rand_seed)

        if allow_monitor:
            def video_callback(episode):
                return episode % args.video_freq < NUM_EPISODE_RECORED
            path = get_output_path(args, env_str='gym')
            env = gym.wrappers.Monitor(env, path, video_callable=video_callback)

        is_deepmind_env = False

    return env, is_deepmind_env


def get_input_output_size(task, env=None):

    if task in fish_env_wrapper.FISH_TYPE or \
            task in dm_control_util.DM_ENV_INFO:
        # dm environments
        return dm_control_util.DM_ENV_INFO[task]

    else:
        # gym environments
        if env is None:
            import gym
            env = gym.make(task)
        return {'ob_size': env.observation_space.shape[0],
                'ac_size': env.action_space.shape[0]}
