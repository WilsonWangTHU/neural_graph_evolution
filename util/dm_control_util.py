# -----------------------------------------------------------------------------
#   @brief:
#       wrapper for the dm_control ob dictionary
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
# import pdb
# import sys
import init_path
import numpy as np
# import multiprocessing
# import agent
# import pdb; pdb.set_trace()
# from util import parallel_util


BASE_PATH = init_path.get_abs_base_dir()
DM_ENV_INFO = {
    # fishes
    'fish3d-swim': {'ob_size': 20, 'ac_size': 5, 'length': 1000},
    'fish3d-easyswim': {'ob_size': 20, 'ac_size': 5, 'length': 1000},
    'fish3d-speed': {'ob_size': 17, 'ac_size': 5, 'length': 1000},

    'acrobot-swingup_sparse': {'ob_size': 6, 'ac_size': 1, 'length': 1000},
    'acrobot-swingup': {'ob_size': 6, 'ac_size': 1, 'length': 1000},

    'ball_in_cup-catch': {'ob_size': 8, 'ac_size': 2, 'length': 1000},

    'cartpole-swingup_sparse': {'ob_size': 5, 'ac_size': 1, 'length': 1000},
    'cartpole-balance': {'ob_size': 5, 'ac_size': 1, 'length': 1000},
    'cartpole-balance_sparse': {'ob_size': 5, 'ac_size': 1, 'length': 1000},
    'cartpole-swingup': {'ob_size': 5, 'ac_size': 1, 'length': 1000},

    'cheetah-run': {'ob_size': 17, 'ac_size': 6, 'length': 1000},

    'finger-turn_easy': {'ob_size': 12, 'ac_size': 2, 'length': 1000},
    'finger-spin': {'ob_size': 9, 'ac_size': 2, 'length': 1000},
    'finger-turn_hard': {'ob_size': 12, 'ac_size': 2, 'length': 1000},

    'fish-upright': {'ob_size': 21, 'ac_size': 5, 'length': 1000},
    'fish-swim': {'ob_size': 24, 'ac_size': 5, 'length': 1000},

    'hopper-stand': {'ob_size': 15, 'ac_size': 4, 'length': 1000},
    'hopper-hop': {'ob_size': 15, 'ac_size': 4, 'length': 1000},

    'humanoid-run': {'ob_size': 67, 'ac_size': 21, 'length': 1000},
    'humanoid-stand': {'ob_size': 67, 'ac_size': 21, 'length': 1000},
    'humanoid-walk': {'ob_size': 67, 'ac_size': 21, 'length': 1000},

    'manipulator-bring_ball': {'ob_size': 37, 'ac_size': 5, 'length': 1000},

    'pendulum-swingup': {'ob_size': 3, 'ac_size': 1, 'length': 1000},

    'point_mass-easy': {'ob_size': 4, 'ac_size': 2, 'length': 1000},

    'reacher-hard': {'ob_size': 7, 'ac_size': 2, 'length': 1000},
    'reacher-easy': {'ob_size': 7, 'ac_size': 2, 'length': 1000},

    'swimmer-swimmer6': {'ob_size': 25, 'ac_size': 5, 'length': 1000},
    'swimmer-swimmer15': {'ob_size': 61, 'ac_size': 14, 'length': 1000},

    'walker-run': {'ob_size': 24, 'ac_size': 6, 'length': 1000},
    'walker-stand': {'ob_size': 24, 'ac_size': 6, 'length': 1000},
    'walker-walk': {'ob_size': 24, 'ac_size': 6, 'length': 1000},

    'humanoid_CMU-run': {'ob_size': 137, 'ac_size': 56, 'length': 1000},
    'humanoid_CMU-stand': {'ob_size': 137, 'ac_size': 56, 'length': 1000},
    'humanoid_CMU-walk': {'ob_size': 137, 'ac_size': 56, 'length': 1000},
}


def render(pixel):
    ''' the function used to render the simulation
    '''

    return


def vectorize_ob(ob_dictionary):
    # element = [np.array(element).flatten() for element in ob_dictionary]
    element = [np.array(element[1]).flatten() for element in ob_dictionary.items()]
    return np.concatenate(element)


def get_ob_size(env):

    ob_size = 0
    for ob_type in env.observation_spec().values():
        if len(ob_type.shape) == 0:
            ob_size += 1
        else:
            ob_size += ob_type.shape[0]
    return ob_size


def get_env_names(task_name):
    return task_name.split('-')


def io_information(task_name):
    info = DM_ENV_INFO[task_name]
    return info['ob_size'], info['ac_size'], info['length']


if __name__ == '__main__':
    from dm_control import suite

    for domain_name, task_name in suite.BENCHMARKING:
        env = suite.load(domain_name, task_name)
        max_pathlength = int(env._time_limit / env.control_timestep())
        print(
            "'" + domain_name + '-' + task_name + "' "
            ": 'ob_size': {}, 'ac_size': {}, 'length': {},".format(
                get_ob_size(env), env.action_spec().shape[0], max_pathlength
            )
        )

    env = suite.load('humanoid_CMU', 'run')
    max_pathlength = int(env._time_limit / env.control_timestep())
    print(
        "'" + 'humanoid_CMU' + '-' + 'run ' + "' "
        ": 'ob_size': {}, 'ac_size': {}, 'length': {},".format(
            get_ob_size(env), env.action_spec().shape[0], max_pathlength
        )
    )
