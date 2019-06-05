''' this file implements the functionality to visualize
gene tree (possibly with physical structure visualization) 

preliminary settings:
1. set up local host server because of the d3 library being used.
   need to have the directory to be correct (preferably use the root
   directory of the project).
   python -m http.server <port number>

usage:
1. run the script to create the corresponding json file that 
   describes the tree structure
   python vis_tree.py <other arguments>
2. visualize by opening 
   - genealogy2.html (for genealogy tree)
   - evolution.html (for the progress of evolution over generation)
'''

import numpy as np
import argparse
import init_path
import json
import glob
from pprint import pprint
from colour import Color

def prune_species(pth):
    '''
    '''
    if pth == None:
        return -1

    # filter out the species not existing in pth
    import glob
    filter_spc = {}
    for filename in glob.glob(pth + '/*'):
        if filename.endswith('.npy') == False: continue
        # if filename.endswith('rank_info.npy') == False: continue
        fn = filename.split('/')[-1].split('.')
        try:
            # handle <gen>_<spc>
            gen, spc = fn[0].split('_')
            spc_info = np.load(filename).item()
            continue
            if spc not in filter_spc:
                filter_spc[spc] = []

            filter_spc[spc].append(spc_info)
            continue
        except:
            # handle rank info
            # import pdb; pdb.set_trace()
            rank_info = np.load(filename).item()
            for i, spc in enumerate(rank_info['SpcID']):
                if spc not in filter_spc: filter_spc[str(spc)] = []
                filter_spc[str(spc)].append({'AvgRwd': rank_info['AvgRwd'][i]})

    return filter_spc


good_creature_path = []
creature_path = {}

def add_image(pth, genealogy):
    ''' traverse through the genealogy tree
    '''
    def find_image(pth, name):
        '''
        '''
        if name not in good_creature_path: return None
        # search in the pth for the corresponding image
        for filename in glob.glob(pth + '/*'):
            abs_pth = filename
            if filename.endswith('.png') == False: continue
            if 'species_data' in filename:
                gen, spc = filename.split('/')[-1].split('.')[0].split('_')
            if 'species_topology' in filename:
                spc = filename.split('/')[-1].split('.')[0]

            if int(spc) != name: continue


            return abs_pth

        return None

    if pth is None:
        print('Not adding images')
        return genealogy

    cur_ptr = genealogy
    parent_list = [genealogy]
    while len(parent_list) > 0:
        item = parent_list.pop()
        image_path = find_image(pth, item['name'])
        item['icon'] = image_path

        parent_list += item['children']    

    with open('genealogy.json', 'w') as fh:
        json.dump(genealogy, fh, indent=4)

    return genealogy

def find_path(tree, target):

    def not_a_child_in_tree(tree, node):

        for k, v in tree.items():
            if node in v:
                return False
        return True

    path = []
    curr = target
    if tree == None: import pdb; pdb.set_trace()
    while not not_a_child_in_tree(tree, curr):
        path.insert(0, curr)
        try:
            for k, v in tree.items():
                if curr in v:
                    curr = k
                    break
        except: 
            import pdb; pdb.set_trace()
            pass

    return path



def parse_tree_file(pth, candidate_list=None):
    '''
    '''

    def build_tree(parent, adj_list, candidate_list=None):
        '''
        '''
        # import pdb; pdb.set_trace()
        child_list = set(adj_list[ int(parent['name']) ])
        if len(child_list) == 0: return parent

        if len(set(child_list).intersection(good_creature_path)) != 0:
            pass
        else:
            return parent
            pass

        for child in child_list:
            if candidate_list is None:
                pass
            elif str(child) in candidate_list:
                pass
            else:
                continue

            # trying to skip ahead 
            if child not in good_creature_path:
                import random
                if random.randint(0, 3) != 0:
                    continue

            else:
                import random
                k = [k if child in set(v) else -233 for k, v in creature_path.items()]
                k = max(k)
                idx = creature_path[k].index(child)

                new_idx = idx + 4 # random.randint(0, 3)
                if new_idx < len(creature_path[k]):
                    child = creature_path[k][new_idx]
                pass

            # global used_gene
            # used_gene[str(child)] = 0
            child_node = build_node(child)
            
            # add other information
            if candidate_list is not None:
                info_list = candidate_list[str(child)]
                r_list = [gen_info['AvgRwd'] for gen_info in info_list]
                child_node['best_r'] = max(r_list)
                child_node['info'] = 'R:%.2f' % \
                    (child_node['best_r'])
                pass

            if child in adj_list:
                child_node = build_tree(child_node, adj_list, 
                                        candidate_list=candidate_list)
            parent['children'].append(child_node)

        return parent

    def build_node(name):
        '''
        '''
        node = {}
        node['name'] = name
        node['children'] = []
        return node

    def genealogy_by_topology(tree_pth):
        topo_pth = tree_pth.replace('gene_tree.npy', 'species_topology')
        gene_tree = {}
        for fn in glob.glob(topo_pth + '/*'):
            if fn.endswith('.npy') == False: continue
            data = np.load(fn).item()

            if data['PrtID'] not in gene_tree:
                gene_tree[data['PrtID']] = []
            gene_tree[data['PrtID']].append(data['SpcID'])

        return gene_tree


    gene_tree = np.load(pth)[()]
    gene_tree = genealogy_by_topology(pth)
    genealogy = {}

    global good_creature_path
    global creature_path
    # for the good walker
    # creature_path[1164] = find_path(gene_tree, 1164)
    # creature_path[2017] = find_path(gene_tree, 2017)
    # creature_path[735] = find_path(gene_tree, 735)
    # creature_path[1908] = find_path(gene_tree, 1908)

    # for the good fish
    # creature_path[8369] = find_path(gene_tree, 8369)
    # creature_path[1603] = find_path(gene_tree, 1603)
    creature_path[8369] = find_path(gene_tree, 8369)
    creature_path[8044] = find_path(gene_tree, 8044)
    
    flatten = lambda l: [x for values in l.values() for x in values]
    good_creature_path = flatten(creature_path)

    print(good_creature_path)

    # find a list of origin nodes
    # keys in gene_tree that don't exist in items
    all_parent = []
    all_childs = []
    for k, v in gene_tree.items():
        all_parent += [k]
        all_childs += v
    all_parent = set(all_parent)
    all_childs = set(all_childs)

    origins = [] # a list of origin species 
    for item in all_parent:
        is_origin = True
        if item in all_childs:
            is_origin = False
            break
        if is_origin: origins.append(item)

    # construct the genealogy
    # gene_tree[-1] = gene_tree[-1]
    gene_tree[-1] = gene_tree[-1][len(gene_tree[-1]) // 2:]
    genealogy = build_node('-1')
    genealogy = build_tree(genealogy, gene_tree, candidate_list=candidate_list)

    with open('genealogy.json', 'w') as fh:
        json.dump(genealogy, fh, indent=4)

    return genealogy

def genealogy_with_style(genealogy):
    ''' add more styles such as coloring the connection link 
    in the visualization
    '''
    def process_node(node):
        '''
        '''
        if len(node['children']) == 0:
            return node


        child_r_list = [c['best_r'] for c in node['children']]

        sorted_r = sorted(child_r_list)
        sorted_color = list(Color('black').range_to(Color('black'), len(sorted_r)))

        max_c = max(child_r_list)
        min_c = min(child_r_list)
        for i, child in enumerate(node['children']):
                
            try:
                color_idx = sorted_r.index(child['best_r'])
                red, green, blue = (x * 255 for x in sorted_color[color_idx].rgb)
                child['level'] = 'rgb(%f %f %f)' % (red, green, blue)
            except:
                child['level'] = 'rgb(128, 128, 128)'
                
            child = process_node(child)
        return node


    genealogy = process_node(genealogy)

    with open('genealogy.json', 'w') as fh:
        json.dump(genealogy, fh, indent=2)

    return genealogy

def evolution_graph(pth, k=5):
    ''' create a graph for visualizing how evolution takes place
    input:
        1. pth should be 
           .../evolution_data/<some training session>/species_data
           where all the <gen num>_rank_info.npy are stored
    output:
        1. a dictionary that can be served as a json for describing 
           the evolution progress
    '''
    def build_node(spc_id, reward, info=None):
        '''
        '''
        node = {}
        node['name'] = spc_id
        node['reward'] = reward
        node['children'] = []
        node['info'] = info
        return node

    def build_tree(parent):

        return 

    evolution = {}

    # get all the generation ranking info
    gen_list = []
    for filename in glob.glob(pth + '/*'):
        if filename.endswith('rank_info.npy') == False: continue
        gen = int(filename.split('/')[-1].split('_')[0])
        data = np.load(filename).item()
        gen_list.append( (gen, data) )
    gen_list = sorted(gen_list, key=lambda x: x[0])

    # create the actual graph
    evolution = build_node(-1, -233, info='other')
    evolution['rank'] = 0

    prev_gen = [evolution]

    gen_pointer = 0
    # for gen_data in gen_list:
    while gen_pointer < len(gen_list):
        gen_data = gen_list[gen_pointer]
        gen_pointer += 1

        
        i_gen, gen_data = gen_data

        other_node = build_node(-2, -233, info='other@gen:%d' % i_gen)
        prev_gen[0]['children'].append(other_node)
        
        cur_gen = [other_node]
        for i in range(k):
            if i >= len(gen_data['SpcID']):
                continue

            spc_id = gen_data['SpcID'][i]
            reward = gen_data['AvgRwd'][i]
            p_spc = gen_data['PrtID'][i]

            node = build_node(spc_id, reward)
            cur_gen.append(node)

            try:
                prev_gen_spc = [x['name'] for x in prev_gen]
                p_pos = prev_gen_spc.index(spc_id)
                # node['level'] = 'yellow'
            except:
                try:
                    prev_gen_spc = [x['name'] for x in prev_gen]
                    p_pos = prev_gen_spc.index(p_spc)
                except:
                    p_pos = 0 # the 'other' node
            # import pdb; pdb.set_trace()
            prev_gen[p_pos]['children'].append(node)

        # add more styles to the current generation
        sorted_r = sorted(x['reward'] for x in cur_gen[1:])
        sorted_color = list(Color('black').range_to(Color("black"), len(sorted_r)))

        max_r = max([x['reward'] for x in cur_gen[1:]])
        min_r = min([x['reward'] for x in cur_gen[1:]])
        for i, x in enumerate(cur_gen):
            if x['info'] is None:
                x['info'] = 'Spc:%d-R:%.2f' % (x['name'], x['reward'])
            x['rank'] = i # rank for plotting the tree no actual meaning
            if 'level' in x: continue

            try:
                color_idx = sorted_r.index(x['reward'])                
                red, green, blue = (x * 255 for x in sorted_color[color_idx].rgb)
                x['level'] = 'rgb(%f %f %f)' % (red, green, blue)
            except:
                x['level'] = 'rgb(128, 128, 128)'

        # representing all the other nodes
        # other_node = build_node(-2, -233, info='other')
        # prev_gen[-1]['children'].append(other_node)
        # cur_gen.append(other_node)

        # import pdb; pdb.set_trace()
        prev_gen = cur_gen
        pass


    with open('evolution.json', 'w') as fh:
        json.dump(evolution, fh, indent=2)

    return evolution

def evolution_add_image(pth, evolution):
    ''' add image for evolution
    input: 
        1. pth contains all the images
        2. evolution tree without image
    '''
    def find_image(pth, name):
        '''
        '''
        # search in the pth for the corresponding image
        found = False
        for filename in glob.glob(pth + '/*'):
            abs_pth = filename
            if filename.endswith('.png') == False: continue
            if 'species_data' in filename:
                gen, spc = filename.split('/')[-1].split('.')[0].split('_')
            elif 'species_topology' in filename:
                spc = filename.split('/')[-1].split('.')[0]

            if int(spc) != name: continue
            found = True
            return abs_pth

        if found != True:
            return None
        else:
            assert 0
            return None

    if pth is None: 
        print('Not adding images for evolution')
        return evolution


    cur_ptr = evolution
    parent_list = [evolution]
    while len(parent_list) > 0:
        item = parent_list.pop()

        for child in item['children']:
            if child['name'] == item['name']:
                child['icon'] = None 
        if 'icon' not in item:
            image_path = find_image(pth, item['name'])
            item['icon'] = image_path

        parent_list += item['children']    

    with open('evolution.json', 'w') as fh:
        json.dump(evolution, fh, indent=4)

    return evolution

def get_config():
    ''' parse the arguments for visualizing the evolution process
    '''
    def post_process(args):
        '''
        '''
        if args.test:
            args.tree_pth = '../evolution_data/test_tree_vis/gene_tree.npy'
        return args

    parser = argparse.ArgumentParser(description='Set up gene tree visual')

    # 
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tree_pth', type=str, default=None)
    parser.add_argument('--species_filter_pth', type=str, default=None)
    parser.add_argument('--image_pth', type=str, default=None)
    parser.add_argument('--evo_topk', type=int, default=5)

    args = parser.parse_args()
    args = post_process(args)

    return args


if __name__ == '__main__':
    
    args = get_config()

    
    if args.species_filter_pth is not None:
        candidate = prune_species(args.species_filter_pth)
    else:
        candidate = None
    
    print('Generating genealogy tree...')
    genealogy = parse_tree_file(args.tree_pth, candidate)
    genealogy = genealogy_with_style(genealogy)

    if args.image_pth is not None:
        genealogy = add_image(args.image_pth, genealogy)
    
    print('Generating evolution tree...')
    evolution = evolution_graph(args.species_filter_pth, k=args.evo_topk)
    evolution = evolution_add_image(args.image_pth, evolution)

    # import pdb; pdb.set_trace()
    
