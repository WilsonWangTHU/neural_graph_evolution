# -----------------------------------------------------------------------------
#   @brief:
#       The species_bank save all the information of the species
#       written by Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np
import init_path
from util import logger
from . import species2species
import os
import glob

KEY_LIST = {
    'species_topology':
        ['xml_str', 'adj_matrix', 'node_attr',
         'PrtID', 'SpcID'],
    'species_data':
        ['policy_weights', 'baseline_weights', 'running_mean_info',
         'var_name',
         'stats', 'start_time', 'rollout_time',
         'xml_str', 'adj_matrix', 'node_attr', 'node_info',
         'PrtID', 'SpcID', 'LastRwd', 'AvgRwd', 'agent_id',
         'env_name', 'action_size', 'observation_size',
         'rank_info', 'debug_info']
}


class species_bank(object):
    '''
        @brief:
            ['policy_weights', 'baseline_weights', 'running_mean_info',
             'var_name']

            ['stats', 'start_time', 'rollout_time']
            ['xml_str', 'adj_matrix', 'node_attr', 'node_info']

            ['PrtID', 'SpcID' 'agent_id', 'LastRwd', 'AvgRwd']
            ['env_name', 'action_size', 'observation_size']

            ['rank_info', 'debug_info']
    '''

    def __init__(self, load_path, elimination_rate, maximum_num_species, args):
        self.num_total_species = 0
        self.current_generation = 1
        self.elimination_rate = elimination_rate
        self.maximum_num_species = maximum_num_species
        self.args = args

        self.species = []
        self.base_path = os.path.join(
            init_path.get_abs_base_dir(), 'evolution_data',
            self.args.task + '_' + self.args.time_id
        )
        self.video_base_path = os.path.join(self.base_path, 'species_video/')
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            os.makedirs(os.path.join(self.base_path, 'species_topology'))
            os.makedirs(os.path.join(self.base_path, 'species_data'))
            os.makedirs(self.video_base_path)

        if load_path is not None:
            self.load(load_path)
        self.gene_tree = {}

    def save(self):
        if self.current_generation % self.args.species_bank_save_freq != 0:
            return
        # the species in this generation and their rank
        path = os.path.join(self.base_path, 'species_data')
        rank_key_list = ['PrtID', 'SpcID', 'LastRwd', 'AvgRwd']
        rank_info = {key: [] for key in rank_key_list}
        rank_info['current_generation'] = self.current_generation
        rank_info['num_total_species'] = self.num_total_species

        for i_species in self.species[::-1]:
            for key in rank_key_list:
                rank_info[key].append(i_species[key])
        np.save(
            os.path.join(path, str(self.current_generation) + '_rank_info'),
            rank_info
        )

        # save the weights for the first species
        for i_species in self.species[::-1][:self.args.save_top_species]:
            np.save(
                os.path.join(
                    path, str(self.current_generation) + '_' +
                    str(i_species['SpcID'])
                ),
                {key: i_species[key] for key in KEY_LIST['species_data']}
            )

    def load(self, load_path):
        # the number of species, the number of generation
        load_path = os.path.abspath(os.path.join(self.base_path, load_path))
        species_list = glob.glob(load_path + '_*.npy')

        self.current_generation = os.path.basename(load_path).split('_')[0]
        rank_info = np.load(load_path + '_rank_info.npy').item()
        assert self.current_generation == rank_info['current_generation']
        self.num_total_species = rank_info['num_total_species']

        # load the species
        for i_species_path in species_list:
            i_species = np.load(i_species_path).item()
            self.add_species(i_species)
        num_species = len(self.species)
        logger.info('Load species bank with bank size: {}'.format(num_species))
        logger.info('Start with generation: {}'.format(self.current_generation))
        logger.info('{} Species existed'.format(self.num_total_species))

    def log_information(self):
        logging_items = ['SpcID', 'AvgRwd', 'LastRwd', 'PrtID']
        logger.info(('| %10s ' * len(logging_items) + '|')
                    % tuple(logging_items))
        for i_species in self.species:
            logging_value = tuple([i_species[item] for item in logging_items])
            logger.info(
                ('| %10.6f ' * len(logging_items) + '|') % logging_value
            )

    def pop_all_species(self):
        if len(self.species) > self.maximum_num_species and \
                (not self.args.use_pruning):
            logger.error('Error: Invalid bank size %d' % (len(self.species)))
        if len(self.species) > self.args.maximum_num_species_generated and \
                self.args.use_pruning:
            logger.error('Error: Invalid bank size %d' % (len(self.species)))
        return_species = self.species
        self.clear_all_species()

        return return_species

    def get_all_species(self):
        if len(self.species) > self.maximum_num_species:
            logger.error('Error: Invalid bank size %d' % (len(self.species)))
        return_species = self.species

        return return_species

    def clear_all_species(self):
        species_num = len(self.species)
        self.species = []
        return species_num

    def add_species(self, species):
        if 'is_dead' in species and species['is_dead'] and self.args.num_total_generation > 1:
            return species

        if species['SpcID'] < 0:  # the first generation
            self.num_total_species += 1
            species['SpcID'] = self.num_total_species
            # save the topology of the species if it's new
            if not self.args.brute_force_search:
                np.save(
                    self.base_path + '/species_topology/' + str(species['SpcID']),
                    {key: species[key]
                     for key in KEY_LIST['species_topology']}
                )
            else:
                np.save(
                    self.base_path + '/species_topology/' + str(species['SpcID']),
                    {key: species[key]
                     for key in KEY_LIST['species_topology'] + ['brute_reward']}
                )
        self.species.append(species)
        if species['PrtID'] >= 0:
            self.log_genealogy(species['PrtID'], species['SpcID'])
        return species

    def rank_species(self):
        average_reward = [i_species_data['AvgRwd']
                          for i_species_data in self.species]
        inverse_species_rank = np.argsort(average_reward)
        self.species = [self.species[i_rank] for i_rank in inverse_species_rank]

        # put the rank information
        for i_rank, i_species in enumerate(self.species[::-1]):
            i_species['rank_info'] = \
                str(self.current_generation) + '_' + str(i_rank)
            i_species['video_save_path'] = self.video_base_path + '/' + \
                str(self.current_generation) + '_' + str(i_species['SpcID']) + '_'

    def natural_selection(self, disaster=False):
        self.current_generation += 1

        # get_rid of the worst species
        self.rank_species()
        self.save()

        if disaster:
            num_species_to_kill = \
                np.floor(len(self.species) * self.reset_elimination_rate)
        elif self.args.brute_force_search:
            self.clear_all_species()
            num_species_to_kill = 0
        else:
            num_species_to_kill = \
                np.floor(len(self.species) * self.elimination_rate)
        for _ in range(int(num_species_to_kill)):
            self.species.pop(0)  # rip, the eliminated species

        # randomly perturb new species
        self.mutation_and_reproduction()

    def mutation_and_reproduction(self):
        # NOTE
        # return
        # assert len(self.species) != 0, 'Oops! All the species died out.'
        if len(self.species) != 0:
            # when there is still species left,
            # evolve the population based on the curent species_bank
            if self.args.use_pruning:
                self.num_new_species = \
                    self.args.maximum_num_species_generated - len(self.species)
            else:
                self.num_new_species = self.args.maximum_num_species - len(self.species)
            self.last_gen_num = len(self.species)
            logger.info('New species to be added %d' % (self.num_new_species))
            logger.info('Current species left in bank %d' % (self.last_gen_num))

            for _ in range(self.num_new_species):
                # randomly choose the mutation method
                mutation_method = np.random.choice(
                    ['add', 'delete', 'perturb'], 1,
                    p=[self.args.mutation_add_ratio,
                       self.args.mutation_delete_ratio,
                       1 - self.args.mutation_add_ratio -
                       self.args.mutation_delete_ratio]
                )

                '''
                mutation_method = np.random.choice(
                    ['add', 'delete', 'perturb'], 1, p=[0, 0, 1]
                )
                '''
                # randomly choose the parent
                p_species = self.species[np.random.randint(self.last_gen_num)]
                c_species = \
                    species2species.mutate_species(
                        p_species,
                        mutation_method,
                        new_spc_struct=self.args.new_species_struct
                    )

                self.species.append(c_species)
        else:
            # when there is no species left # reset the species bank
            self.clear_all_species()

    def save_gene_map(self):
        '''
        '''

        path = os.path.join(self.base_path, 'gene_tree')
        logger.info('Saving Genealogy...')
        np.save(path, self.gene_tree)

        return None

    def log_genealogy(self, parent_spc, child_spc):
        ''' log parent-child relationship in the overall gene tree
        update on the self.gene_tree structure
        a tree structure
        '''
        if parent_spc not in self.gene_tree:
            self.gene_tree[parent_spc] = []
        self.gene_tree[parent_spc].append(child_spc)
