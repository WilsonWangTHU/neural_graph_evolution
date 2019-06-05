'''
    @brief:
        analyze the population after each generation
'''

import init_path
from util import logger
from sklearn.decomposition import PCA
import numpy as np

# local imports
from util import visdom_util

__BASE_PATH = init_path.get_abs_base_dir()


def print_info(args):
    ''' print important training hyperparameters
    '''
    info_list = []

    info_list.append('Time id: %s' % (args.time_id))
    info_list.append('Task:%s' % args.task)
    info_list.append('Note:%s' % args.note)
    info_list.append('------')

    # some environment configuration
    if 'curlswim' in args.task:
        info_list.append('target v:%f' % (args.fish_target_speed))
        info_list.append('target omega:%f' % (args.fish_target_var))
        info_list.append('target A:%f' % (args.fish_target_bound))
    elif 'easyswim' in args.task:
        info_list.append('Target angle: %f' % (args.fish_target_angle))
    elif 'circleswim' in args.task:
        info_list.append('target v: %f' % (args.fish_target_speed))
        info_list.append('target r: %f' % (args.fish_circle_radius))
    
    info_list.append('ThreadNum %d; PopulationNum %d' \
        % (args.num_threads, args.maximum_num_species))
    info_list.append('Inner iteration: %d' % (args.evolutionary_sub_iteration))
    info_list.append('Batch size: %d' % (args.timesteps_per_batch))
    info_list.append('Using Nervenet+: %d' % (args.nervenetplus))
    info_list.append('Discrete mutate: %d' % (args.discrete_mutate))
    info_list.append('Start species num: %d' % (args.start_num_species))
    if args.optimize_creature:
        info_list.append('Using original creature structure')
    else:
        info_list.append('Using random creature structure')
    info_list.append('Elimination rate: %f' % (args.elimination_rate))
    info_list.append('Mutation add: %f delete: %f | node mutate ratio: %f' \
        % (args.mutation_add_ratio, args.mutation_delete_ratio, args.node_mutate_ratio))
    info_list.append('GNN embeding option: %s' % (args.gnn_embedding_option))
    info_list.append('------')
    info_list.append('Use pruning: %d' % (args.use_pruning))
    info_list.append('Prune BS: %d' % (args.pruning_batch_size))
    info_list.append('Max species generated: %d' % (args.maximum_num_species_generated))


    visdom_util.visdom_print_info(info_list, args.time_id)

    return None


def trans_PCA(orig_data, num_components=1):
    ''' convert original data to a PCA representation
        input:
            1. orig_data - (num_samples, num_features) numpy array
            2. num_conponents - number of features that is used as output
        output:
            1. pca_data - (num_samples, num_components)
    '''
    pca = PCA(n_components=num_components)
    pca.fit(orig_data)
    pca_data = pca.transform(orig_data)
    return pca_data, pca


def analyse_gen(args, i_gen, all_species):
    ''' @brief:
            perform various analysis on the current generation's species
            each element inside all_species has the following keys
            ['policy_weights', 'baseline_weights', 'running_mean_info',
             'training_stats', 'LastRwd', 'AvgRwd', 'start_time', 'end_time',
             'PrtID', 'agent_id', 'adj_matrix', 'node_attr', 'xml_str', 'SpcID']
        input:
            i_gen: current generation number
            all_species: a list of all the species
    '''
    if not args.viz:
        logger.error('Need to set up visdom to visualize analysis')
        return -1
    if i_gen == 0:
        plot_item = ['reward', 
                     'size_dist', 'pc_relationship', 'angle_range',
                     'pca_size_eigenvalue', 'size_v_dist', 'size_v_line',
                     'pca_pc_eigenvalue',
                     'body_num_line', 'body_size_line']
        analyse_gen.win_dict = {item: None for item in plot_item}

    title_module = '%s %s' % (args.time_id, args.task) + '-%s'

    required_info = ['AvgRwd', 'adj_matrix', 'node_attr']
    species_book = {}

    # parse the species list for required info
    for i, species in enumerate(all_species):
        species_book[i] = {}
        for item in required_info:
            species_book[i][item] = species[item]

    # reward
    gen_reward = [v['AvgRwd'] for k, v in species_book.items()]
    gen_reward = np.array(gen_reward)
    if len(gen_reward) == 0:
        return -1
    avg_r = float(gen_reward.mean())
    max_r = float(gen_reward.max())
    min_r = float(gen_reward.min())
    analyse_gen.win_dict['reward'] =  \
        visdom_util.viz_line(i_gen, [max_r, avg_r, min_r],
                             viz_win=analyse_gen.win_dict['reward'],
                             title=title_module % ('reward'),
                             xlabel='generation',
                             ylabel='reward',
                             legend=['Max', 'Avg', 'Min'])

    # get all the nodes' parameters and plot the histogram of their PCA
    size_mat = []
    pc_rel_mat = []
    range_mat = []
    body_num_mat = []
    species_size = []
    for k, v in species_book.items():

        body_num_mat.append(len(v['node_attr']))
        species_size.append( 
            [np.array([node['a_size'], node['b_size'], node['c_size']]) 
            for node in v['node_attr']] 
        )

        for node in v['node_attr']:
            if node['geom_type'] == -1:
                continue
            size_mat.append(
                np.array([node['a_size'], node['b_size'], node['c_size']])
            )
            pc_rel_mat.append(
                np.array([node['u'], node['v'], node['axis_x'], node['axis_y']])
            )
            range_mat.append(np.array([node['joint_range']]))

    size_mat = np.vstack(size_mat)
    pc_rel_mat = np.vstack(pc_rel_mat)
    range_mat = np.vstack(range_mat)
    body_num_mat = np.vstack(body_num_mat)
    species_size = np.array([ 
        # calculate size of each body part and sum up for each species 
        np.sum(np.prod(np.vstack(item), axis=1)) 
        for item in species_size 
    ])
    # import pdb; pdb.set_trace()

    size_pca, size_pca_info = trans_PCA(size_mat)
    pc_rel_pca, pc_rel_pca_info = trans_PCA(pc_rel_mat)
    range_pca, range_pca_info = trans_PCA(range_mat)

    # plotting pca coefficients
    if args.pca_size_dist:
        analyse_gen.win_dict['size_dist'] = \
            visdom_util.vis_hist_plot(i_gen, size_pca,
                                      viz_win=analyse_gen.win_dict['size_dist'],
                                      title=title_module % ('sizePCA'))
    if args.pca_pc_dist:
        analyse_gen.win_dict['pc_relationship'] = \
            visdom_util.vis_hist_plot(i_gen, pc_rel_pca,
                                      viz_win=analyse_gen.win_dict['pc_relationship'],
                                      title=title_module % ('pc-PCA'))
    if args.pca_range_dist:
        analyse_gen.win_dict['angle_range'] = \
            visdom_util.vis_hist_plot(i_gen, range_pca,
                                      viz_win=analyse_gen.win_dict['angle_range'],
                                      title=title_module % ('rangePCA'))
    
    # plot body num changes over generation
    if args.body_num_line:
        analyse_gen.win_dict['body_num_line'] = \
            visdom_util.viz_line(i_gen, 
                [body_num_mat.max(), body_num_mat.mean(), body_num_mat.min()],
                viz_win=analyse_gen.win_dict['body_num_line'],
                title=title_module % ('body_num'),
                xlabel='generation', ylabel='body num in one species',
                legend=['Max', 'Mean', 'Min'])
                   
    if args.body_size_line:
        analyse_gen.win_dict['body_size_line'] = \
            visdom_util.viz_line(i_gen,
                [species_size.max(), species_size.mean(), species_size.min()],
                viz_win=analyse_gen.win_dict['body_size_line'],
                title=title_module % ('species_volume'),
                xlabel='generation', ylabel='species volume/weight',
                legend=['Max', 'Mean', 'Min'])


    # plotting PCA eigenvalues
    if args.pca_pc_eigenvalue:
        val = pc_rel_pca_info.components_.squeeze().tolist()
        analyse_gen.win_dict['pca_pc_eigenvalue'] = \
            visdom_util.viz_line(i_gen, val,
                                 viz_win=analyse_gen.win_dict['pca_pc_eigenvalue'],
                                 title=title_module % ('pc_eigenvalue'),
                                 xlabel='generation', ylabel='Eigenvalue',
                                 legend=['u', 'v', 'axis_x', 'axis_y'])
    if args.pca_size_eigenvalue:
        raise NotImplementedError

    # plotting volume
    if args.size_v_dist: # plotting volume histogram
        size_vol = np.prod(size_mat, axis=1).reshape(-1,1)
        analyse_gen.win_dict['size_v_dist'] = \
            visdom_util.vis_hist_plot(i_gen, size_vol,
                                      viz_win=analyse_gen.win_dict['size_v_dist'],
                                      title=title_module % ('volumePCA'))
    if args.size_v_line: # plotting volume line
        size_vol = np.prod(size_mat, axis=1).reshape(-1,1)
        max_v = size_vol.max()
        mean_v = size_vol.mean()
        min_v = size_vol.min()
        analyse_gen.win_dict['size_v_line'] = \
            visdom_util.viz_line(i_gen, [max_v, mean_v, min_v],
                                 viz_win=analyse_gen.win_dict['size_v_line'],
                                 title=title_module % ('body volume'),
                                 xlabel='generation', ylabel='volume (in unit square)',
                                 legend=['Max', 'Mean', 'Min'])



    # plot pca variance



    return None


if __name__ == '__main__':
    pass
