'''
    visualizing the tree structure specific to one species
'''

import init_path

import os
import numpy as np
import argparse
import json
import glob
from pprint import pprint

# local imports
from util import visdom_util


def get_config():
    ''' parse the arguments for visualizing the evolution process
    '''
    def post_process(args):
        '''
        '''
        if args.test:
            args.phy_pth = '../evolution_data/test_tree_vis/gene_tree.npy'
            args.tree_pth = '../evolution_data/test_tree_vis/gene_tree.npy'
        return args

    parser = argparse.ArgumentParser(description='Set up gene tree visual')

    # 
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--world_path', type=str, default=None)
    parser.add_argument('--species', type=int, default=1)

    parser.add_argument('--tree_pth', type=str, default=None)
    parser.add_argument('--species_filter_pth', type=str, default=None)
    parser.add_argument('--image_pth', type=str, default=None)
    parser.add_argument('--evo_topk', type=int, default=5)

    args = parser.parse_args()
    args = post_process(args)

    return args

def generate_full_tree(pth):
    ''' brief:
            create the full evolution relationship.
            equivalent to using top-<max species num> in visualizing evolution
    '''
    from html_visual import vis_tree

    data_path = os.path.join(pth, 'species_data')

    rank_info_list = []
    for fn in glob.glob(data_path + '/*'):
        if fn.endswith('_rank_info.npy') == False:
            continue
        rank_info_list.append(fn)

    rank_info_list = sorted(rank_info_list, 
                            key=lambda x: int(x.split('/')[-1].split('_')[0]))

    last_gen = np.load(rank_info_list[-1]).item()
    max_species_num = len(last_gen['PrtID'])

    all_species_evolution = vis_tree.evolution_graph(data_path + '/', k=max_species_num)
    return all_species_evolution

def parent_child_performance(pth):
    ''' plotting parent-child and their performance difference
    '''
    data_path = os.path.join(pth, 'species_data')

    rank_info_list = []
    for fn in glob.glob(data_path + '/*'):
        if fn.endswith('_rank_info.npy') == False:
            continue
        rank_info_list.append(fn)

    rank_info_list = sorted(rank_info_list, 
                            key=lambda x: int(x.split('/')[-1].split('_')[0]))

    pc_performance_list = []

    for i, fn in enumerate(rank_info_list):
        if i == 0: continue
        curr_data = np.load(fn).item()
        prev_data = np.load(rank_info_list[i-1]).item()


        # getting all the parent's id
        p_id_list = curr_data['PrtID']
        prev_gen_spc = prev_data['SpcID']

        species_needed = [(curr_data['SpcID'][j], x) \
            for j, x in enumerate(p_id_list) \
            if x in prev_gen_spc 
        ]

        # create the reward
        p_r, c_r = [], []
        for pc_pair in species_needed:
            child, parent = pc_pair
            p_r.append(prev_data['AvgRwd'][prev_data['SpcID'].index(parent)])
            c_r.append(curr_data['AvgRwd'][curr_data['SpcID'].index(child)])

        pc_performance_list.append( (sum(p_r)/len(p_r), sum(c_r)/len(c_r)) )

    # use visdom to plot the line
    win1 = None
    win2 = None
    for i, val in enumerate(pc_performance_list):
        v1, v2 = val
        win1 = visdom_util.viz_line(i, [v1, v2], viz_win=win1,
            title='Avg Parent-child performance comparison',
            xlabel='Generation', ylabel='Average Reward',
            legend=['Parent', 'Child']
        )
        win2 = visdom_util.viz_line(i, [ (v2 - v1) / v1 ], viz_win=win2,
            title='Drop comparing to parent in percentile',
            xlabel='Generation', ylabel='Drop in percentile',
            legend=['Drop'])


    return None



def filter_spc(evolution_graph, spc_id):
    ''' get all the related nodes wrt spc_id
    includes: 1. direct parent
              2. all their children
    '''

    return evolution_graph




if __name__ == '__main__':

    args = get_config()

    parent_child_performance(args.world_path)

    all_evolution = generate_full_tree(args.world_path)

