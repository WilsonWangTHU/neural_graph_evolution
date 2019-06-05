# -----------------------------------------------------------------------------
#   @brief: main fucntion
# -----------------------------------------------------------------------------

import os
import gym
# import pdb
import init_path
from agent import optimization_agent
from agent import rollout_master_agent
from config.config import get_config
from util import parallel_util
from util import dm_control_util
from util import logger
from util import visdom_util
import multiprocessing
import time
from environments import register
init_path.bypass_frost_warning()
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'  # disable headless rendering


def make_optimizer(ob_size, action_size, args):
    ''' dm_control optimizer requires different treatment
    '''
    optimizer_tasks = multiprocessing.JoinableQueue()
    optimizer_results = multiprocessing.Queue()
    optimizer_agent = optimization_agent.optimization_agent(
        args,
        ob_size, action_size,
        optimizer_tasks, optimizer_results
    )
    optimizer_agent.start()

    optimizer_tasks.put(parallel_util.START_SIGNAL)
    optimizer_tasks.join()
    starting_weights = optimizer_results.get()
    return optimizer_tasks, optimizer_results, optimizer_agent, starting_weights


def make_rollout_agent(ob_size, action_size, starting_weights, args):
    rollout_agent = rollout_master_agent.parallel_rollout_master_agent(
        args, ob_size, action_size
    )
    rollout_agent.set_policy_weights(starting_weights)
    return rollout_agent


if __name__ == '__main__':

    # get the configuration
    logger.info('New environments available : {}'.format(
        register.get_name_list()))
    args = get_config()
    # args.use_nervenet = 0

    if args.write_log:
        logger.set_file_handler(
            path=args.output_dir,
            prefix='mujoco_' + args.task, time_str=args.time_id
        )

    if args.task in dm_control_util.DM_ENV_INFO:
        args.dm = 1

    # optional visdom plotting
    if args.viz:
        viz_item = ['avg_reward', 'entropy', 'kl', 'surr_loss',
                    'vf_loss', 'weight_l2_loss', 'learning_rate']
        viz_win = {}
        for item in viz_item:
            viz_win[item] = None

    if not args.dm:
        args.max_pathlength = gym.spec(args.task).timestep_limit
        learner_env = gym.make(args.task)

        optimizer_tasks = multiprocessing.JoinableQueue()
        optimizer_results = multiprocessing.Queue()
        optimizer_agent = optimization_agent.optimization_agent(
            args,
            learner_env.observation_space.shape[0],
            learner_env.action_space.shape[0],
            optimizer_tasks,
            optimizer_results
        )
        optimizer_agent.start()

        # the rollouts agents
        rollout_agent = rollout_master_agent.parallel_rollout_master_agent(
            args,
            learner_env.observation_space.shape[0],
            learner_env.action_space.shape[0]
        )

        # start the training and rollouting process
        optimizer_tasks.put(parallel_util.START_SIGNAL)
        optimizer_tasks.join()
        starting_weights = optimizer_results.get()
        # Note that it is super hacky, if the number of threads is too big,
        # when the weights are being assigned, some rollout_agents might
        # not even be created.
        time.sleep(5)
        rollout_agent.set_policy_weights(starting_weights)
    else:
        # the case where invoking dm_control suite
        pass
        ob_size, action_size, max_pathlength = \
            dm_control_util.io_information(args.task)
        args.max_pathlength = max_pathlength
        optimizer_tasks, optimizer_results, optimizer_agent, starting_weights = \
            make_optimizer(ob_size, action_size, args)
        rollout_agent = \
            make_rollout_agent(ob_size, action_size, starting_weights, args)

    # some training stats
    start_time = time.time()
    totalsteps = 0

    while True:

        # step 1: collect rollout data
        rollout_start = time.time()
        paths = rollout_agent.rollout()
        rollout_time = (time.time() - rollout_start) / 60.0

        # step 2: optimize the network based on the collected data
        learn_start = time.time()
        optimizer_tasks.put(paths)
        optimizer_tasks.join()
        results = optimizer_results.get()
        totalsteps = results['totalsteps']
        learn_time = (time.time() - learn_start) / 60.0

        # step 3: update the policy
        rollout_agent.set_policy_weights(results['policy_weights'])

        # command line logger
        logger.info("---------- Iteration %d -----------" % results['iteration'])
        logger.info("total time: %.2f mins" % ((time.time() - start_time) / 60.0))
        logger.info("optimization agent spent : %.2f mins" % (learn_time))
        logger.info("rollout agent spent : %.2f mins" % (rollout_time))
        logger.info("%d total steps have happened" % totalsteps)

        if args.viz:
            for item in viz_item:
                viz_win[item] = visdom_util.visdom_plot_curve(
                    int(results['iteration']), float(results['stats'][item]),
                    viz_win=viz_win[item], title=item
                )

        if totalsteps > args.max_timesteps:
            break

    rollout_agent.end()
    optimizer_tasks.put(parallel_util.END_SIGNAL)  # kill the learner
    if args.test:
        logger.info(
            'Test performance ({} rollouts): {}'.format(
                args.test, results['avg_reward']
            )
        )

        logger.info(
            'max: {}, min: {}, median: {}'.format(
                results['max_reward'], results['min_reward'],
                results['median_reward']
            )
        )
