# -----------------------------------------------------------------------------
#   @brief: main function
# -----------------------------------------------------------------------------

import os
import init_path
from config.config import get_config
from util import parallel_util
from util import logger
from util import visdom_util
from evolution import species_bank as sbank
from evolution import agent_bank as abank
from evolution import evo_analysis as analysis
from agent import pruning_agent
import multiprocessing

os.environ['DISABLE_MUJOCO_RENDERING'] = '1'  # disable headless rendering
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'osmesa'


if __name__ == '__main__':
    # read the configs and set the logging path
    args = get_config(evolution=True)
    args.use_nervenet = 1

    base_path = init_path.get_abs_base_dir()
    if args.write_log:
        logger.set_file_handler(path=args.output_dir,
                                prefix='mujoco_' + args.task,
                                time_str=args.time_id)

    if args.viz:
        visdom_util.visdom_initialize(args)
        if not args.mute_info:
            analysis.print_info(args)

    agent_bank = abank.agent_bank(args.num_threads + 1, args)
    species_bank = sbank.species_bank(args.speciesbank_path,
                                      args.elimination_rate,
                                      args.maximum_num_species,
                                      args)

    if args.use_pruning:
        assert args.maximum_num_species < args.maximum_num_species_generated
        # build the pruning agent
        pruning_task = multiprocessing.JoinableQueue()
        pruning_result = multiprocessing.Queue()
        pruning_agent = pruning_agent.pruning_agent(
            args, pruning_task, pruning_result
        )
        pruning_agent.start()

    current_reset_freq = args.controller_reset_freq
    for i_generation in range(args.num_total_generation):
        logger.info('Running evolution at generation: %3d' % i_generation)
        species = species_bank.pop_all_species()
        if len(species) == 0:  # intial run
            species = [None] * (args.start_num_species + 1)
        else:
            # if resetting?
            if args.controller_reset_freq and \
                    i_generation % current_reset_freq == 0:
                # update the reset freq
                current_reset_freq = \
                    int(current_reset_freq * args.reset_progressive_multiplier)
                current_reset_freq = min(current_reset_freq,
                                         args.max_reset_freq)
                reset = True
            else:
                reset = False

            if args.use_pruning:
                pruning_task.put(('Pruning', species, reset))
                pruning_task.join()
                species = pruning_result.get()  # species are now pruned

            # append the generation information
            for i_species in species:
                i_species['reset'] = reset

        for i_species in species:
            if i_species is None:
                signal = parallel_util.AGENT_EVOLUTION_START
            else:
                signal = parallel_util.AGENT_EVOLUTION_TRAIN
            agent_bank.put((signal, i_species))

        for _ in range(len(species)):
            species_copy = species_bank.add_species(agent_bank.get())
            if args.use_pruning:
                pruning_agent.add_species_copy(species_copy)

        analysis.analyse_gen(args, i_generation, species_bank.get_all_species())

        species_bank.log_information()

        if args.controller_reset_freq and \
                i_generation + 1 % current_reset_freq == 0 and \
                args.use_disaster_elimination:
            # if next timesteps a reset will happen, kill at a big rate
            disaster = True
        else:
            disaster = False

        species_bank.natural_selection(disaster)
        species_bank.save_gene_map()

    agent_bank.end()  # kill the learner
