''' create visualization all the training sessions in ../evolution_data/
'''
import os
import init_path

import glob
import json
import argparse
import subprocess
from shutil import copyfile


# local imports
from html_visual import vis_tree
from html_visual import vis_spc_tree

GENEALOGY_HTML = \
    os.path.join(init_path.get_base_dir(), 'html_visual/genealogy.html')
EVOLUTION_HTML = \
    os.path.join(init_path.get_base_dir(), 'html_visual/evolution.html')
GENEALOGY_HTML = \
    os.path.join(init_path.get_base_dir(), 'html_visual/expand_genealogy.html')

def get_config():
    '''
    '''
    def post_process(args):
        '''
        '''

        return args

    parser = argparse.ArgumentParser(description='Set up genealogy visualization')
    parser.add_argument('--use_image', action='store_true', default=False)
    parser.add_argument('--topk', type=int, default=5)

    args = parser.parse_args()
    args = post_process(args)
    return args

def process_one_trn_sess(args, trn_sess_path):
    '''
    '''
    # some sanity check for those invalid training sessions
    if os.path.isfile(trn_sess_path): return
    _, _, trn_sess_name = trn_sess_path.split('/')
    try:
        task_name = trn_sess_name.split('_')[0]
        env_name, task_name = task_name.split('-')
    except:
        return
    if 'evo' not in env_name: return

    # first burst out all the images
    print('Try to burst %s images' % trn_sess_path) 

    command_template = 'python ../env/visualize_species.py -i %s -v 0'
    # process = subprocess.Popen(command_template % \
    #     (os.path.join(trn_sess_path, 'species_data')),
    #     shell=True, stdout=subprocess.PIPE
    # )
    # process.wait()
    # if process.returncode:
    #     print('%s: Cannot burst species images.' % trn_sess_path)
    #     return
    # else:
    #     print('%s: Success.' % trn_sess_path)

    # step 0: initialize the 'visual' directory for the training session
    visual_path = os.path.join(trn_sess_path, 'visual')
    if os.path.exists( visual_path ) == False:
        os.makedirs( visual_path ) 

    copyfile(GENEALOGY_HTML, os.path.join(visual_path, 'genealogy.html'))
    copyfile(GENEALOGY_HTML, os.path.join(visual_path, 'evolution.html'))
    copyfile(GENEALOGY_HTML, os.path.join(visual_path, 'expand_genealogy.html'))

    # step 1: generate the evolution json
    species_data_path = os.path.join(trn_sess_path, 'species_data')
    evolution = vis_tree.evolution_graph(species_data_path, k=args.topk)

    # step 2: generate the genealogy json
    gene_path = os.path.join(trn_sess_path, 'gene_tree.npy')
    candidate = vis_tree.prune_species(species_data_path)
    genealogy = vis_tree.parse_tree_file(gene_path, candidate)
    genealogy = vis_tree.genealogy_with_style(genealogy)

    if args.use_image:
        evolution = vis_tree.evolution_add_iamge(args.)

    evolution_json = os.path.join(visual_path, 'evolution.json')
    with open(evolution_json, 'w') as fh:
        json.dump(evolution, fh, indent=2)

    genealogy_json = os.path.join(visual_path, 'genealogy.json')
    with open(genealogy_json, 'w') as fh:
        json.dump(genealogy, fh, indent=2)

    return None

if __name__ == '__main__':
    '''
    '''
    args = get_config()

    for trn_sess in glob.glob('../evolution_data/*'):
        process_one_trn_sess(args, trn_sess)

