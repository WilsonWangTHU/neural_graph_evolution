'''
    @brief: introduct hiearchical structure in the model
    @author: henry zhou
'''

import init_path
import numpy as np
import random
from copy import deepcopy
import lxml.etree as etree

# local import
from env import model_gen
from env import model_perturb
from util import model_gen_util
from util import logger

from env import species_info

'''
    implementation of a generic tree structure
    with some modifications to fit the project's purpose
'''
class Node:
    def __init__(self, node_id, attr):
        self.id = node_id
        self.attr = attr
        self.child_list = []

    def set_id(self, new_id):
        '''
        '''
        self.id = new_id
        return new_id + 1

    def get_all_node_id(self):
        ''' return a list of node_ids that are part of the node-defined
        subtree
        NOTE: this method returns the node in breadth first search (BFS) order!
        '''
        # get all descendents list
        all_nodes = self.get_all_descendents()
        all_id_list = [x.get_node_id() for x in all_nodes]

        # add its own id
        # all_id_list = [self.get_node_id()] + all_descendents_id
        return all_id_list

    def get_all_descendents(self):
        '''
        '''
        all_nodes = []
        p_queue = [self]
        while len(p_queue) != 0:
            node = p_queue.pop(0)
            for c_node in node.get_child_list():
                p_queue.append(c_node)

            all_nodes.append(node)

        return all_nodes

    def get_node_id(self):
        return self.id 

    def get_attr(self):
        return self.attr

    def set_attr(self, new_attr):
        self.attr = new_attr
        return 
    
    def get_child_list(self):
        return self.child_list
        
    def add_child(self, node):
        self.child_list.append(node)

    def is_same_size(self, other_node):
        ''' return true if the two nodes have the same
        'a_size', 'b_size', 'c_size' in its attribute
        '''
        if np.abs(self.attr['a_size'] - \
            other_node.get_attr()['a_size']) > 1e-7:
            return False
        if np.abs(self.attr['b_size'] - \
            other_node.get_attr()['b_size']) > 1e-7:
            return False
        if np.abs(self.attr['c_size'] - \
            other_node.get_attr()['c_size']) > 1e-7:
            return False
        return True

    def make_symmetric(self, task='fish', discrete=True):
        ''' make the symmetric node w.r.t the self node
        symmetric node and return the new node
        '''
        new_attr = model_gen.gen_test_node_attr(
            task=task,
            node_num=2,
            discrete_rv=discrete
        )[-1]

        same_attr = ['a_size', 'b_size', 'c_size', 'geom_type',
                     'joint_range']
        for item in same_attr:
            new_attr[item] = self.attr[item]

        # set certain attributes
        new_attr['u'] = np.pi - self.attr['u']
        while new_attr['u'] < 0: new_attr['u'] += 2 * np.pi
        new_attr['v'] = self.attr['v']
        new_attr['axis_x'] = -self.attr['axis_x']
        new_attr['axis_y'] = self.attr['axis_y']

        # create the new node
        new_node = Node(-1, new_attr)

        return new_node

    def remove_child(self, node_id):
        child_id_list = [child.get_node_id() for child in self.child_list]
        try:
            idx = child_id_list.index(node_id)
        except:
            # the wanted node_id is not the child of the current node
            return -1

        deleted_node = self.child_list.pop(idx)
        return deleted_node

    def printinfo(self):
        child_id_list = [child.get_node_id() for child in self.child_list]
        print('NodeID:  %3d' % self.id)
        print('ChildIDs:  ', child_id_list)
        return 

    def __eq__(self, other):

        return self.__dict__ == other.__dict__

def node_count(node, counter_start=0):
    ''' count the number of nodes in a tree
    '''
    total_num = counter_start
    parent_list = [node]
    while len(parent_list) != 0:
        node = parent_list.pop(0)
        child_list = node.get_child_list()
        for child in child_list:
            parent_list.append(child)

        total_num = node.set_id(total_num)
    return total_num

class Tree:
    def __init__(self, node=None):
        if node is None: assert 0, 'Tree root is None.'
        self.root = node
        self.total_num = 0
        self.total_num = node_count(self.root, self.total_num)

    def get_root(self):
        return self.root

    def sample_node(self, p=0.5):
        ''' randomly sample a node
        '''
        def check_node(node, p=0.5):
            if np.random.binomial(1, p) == 1:
                return node
            res = None
            for child in node.child_list:
                res = check_node(child)
                if res is not None: return res
            return res

        # WARNING: 
        # this method has different weights for nodes at different depth
        # sample_node = None
        # while sample_node is None:
        #     sample_node = check_node(self.root, p=p)
        all_nodes = self.get_all_nodes()
        sample_node = random.choice(all_nodes)

        return sample_node

    def sample_leaf(self, p=0.5):
        ''' traverse the tree and flip coins on the leaf
        returns a list of leaf nodes that are sampled
        '''
        def find_leaf(node, total_res):
            '''
            '''
            child_list = node.get_child_list()
            if len(child_list) == 0:
                total_res.append(node)
                return total_res
            else:
                for child in child_list:
                    total_res = find_leaf(child, total_res)
            return total_res

        all_leaf = find_leaf(self.root, [])
        sampled_leaf = []
        for node in all_leaf:
            if np.random.binomial(1, p) == 1:
                sampled_leaf.append(node)
        return sampled_leaf

    def get_lvl_node(self, lvl=0):
        ''' return a list of node that belongs to the same lvl
        in the tree
        '''
        lvl_node = []
        cur_lvl_node = [self.root]

        for i in range(lvl):
            new_lvl_node = []
            for node in cur_lvl_node:
                new_lvl_node += node.get_child_list()
            cur_lvl_node = new_lvl_node
        return lvl_node

    def get_all_nodes(self):
        ''' return all the nodes in the tree
        (order is not guaranteed)
        '''
        all_nodes = []
        node_list = [self.root]
        while len(node_list) != 0:
            node = node_list.pop(0)
            for c_node in node.get_child_list():
                node_list.append(c_node)
            all_nodes.append(node)
        return all_nodes

    def get_pc_relation(self):
        ''' return a list of strings '%d-%d'
        meaning parent-child connection
        '''
        pc_rel = []
        p_node_list = [self.root]
        while len(p_node_list) != 0:
            p_node = p_node_list.pop(0)
            for c_node in p_node.get_child_list():
                p_id = p_node.get_node_id()
                c_id = c_node.get_node_id()
                rel = '%d-%d' % (p_id, c_id)
                pc_rel.append(rel)
                p_node_list.append(c_node)
        return pc_rel

    def add_sub_tree(self, parent, child):
        ''' add the child (a node containing subtree)
        to the parent's child_list
        return: list of id values that is assigned to child
        '''
        # traverse child's subtree and update the id
        prev_node_num = self.total_num
        self.total_num = node_count(child, self.total_num)
        parent.add_child(child)

        return list(range(prev_node_num, self.total_num))

    def remove_sub_tree(self, parent, child):
        ''' remove the parent child relationship
        '''
        # figure out the place child within parent's child list
        child_list = parent.get_child_list()
        child_id_list = [item.get_node_id() for item in child_list]
        try: 
            idx = child_id_list.index(child.get_node_id())
        except:
            raise RuntimeError('child doesn\'t exist in parent')
        child_list.pop(idx)
        self.total_num = 0
        self.total_num = node_count(self.root, self.total_num)
        return

    def to_mujoco_format(self):
        ''' create the 1. adj_mat
                       2. node_attr
        to get ready for model_gen
        '''
        N = self.total_num

        pc_relation = self.get_pc_relation()
        all_nodes = self.get_all_nodes() 

        # generate the adj_matrix
        adj_mat = np.zeros((N, N))
        for rel in pc_relation:
            i, j = rel.split('-')
            i, j = int(i), int(j)
            adj_mat[i, j] = 1
        adj_mat = model_gen_util.mirror_mat(adj_mat)
        adj_mat[adj_mat > 0] = 7 

        # generate the attributes
        sorted_nodes = sorted(all_nodes, key=lambda x: x.get_node_id())
        node_attr = [item.get_attr() for item in sorted_nodes]
        return adj_mat.astype('int'), node_attr
    
'''
    an object for defining a mujoco structure
'''

NEW_STRUCT_OPTS = ['l1-basic', 'l2-symmetry']


class Species:
    '''
    '''
    def __init__(self, 
                 args,
                 body_num=3, 
                 discrete=True,
                 allow_hierarchy=True,
                 filter_ratio=0.5):
        # filter ratio
        self.args = args
        self.p = args.node_mutate_ratio
        self.allow_hierarchy = args.allow_hierarchy
        self.discrete = args.discrete_mutate
        
        if args.optimize_creature == False:
            # set up the tree (if no starting species is specified)
            r_node = Node(0, 
                model_gen.gen_test_node_attr(task=args.task, node_num=2, discrete_rv=discrete)[0]
            )
            self.struct_tree = Tree(r_node)

            if 'walker' in self.args.task:
                while len(self.struct_tree.get_root().get_all_descendents()) <= \
                            body_num:
                    self.perturb_add(no_selfcross=True)
            else:
                for i in range(1, body_num):
                    if self.args.force_symmetric:
                        node_list = self.generate_one_node()
                    else:
                        node_list = self.generate_one_node(only_one=True)
                    for c_node in node_list:
                        self.struct_tree.add_sub_tree(r_node, c_node)
        else:
            if 'fish' in args.task:
                self.struct_tree = Tree( species_info.get_original_fish() )
            
            elif 'walker' in args.task:
                self.struct_tree = Tree( species_info.get_original_walker() )
            
            elif 'cheetah' in args.task:
                self.struct_tree = Tree( species_info.get_original_cheetah() )
            
            elif 'hopper' in args.task:
                self.struct_tree = Tree( species_info.get_original_hopper() )

        return None

    def mutate(self):
        '''
        '''
        op = np.random.choice(
            ['add', 'remove', 'perturb'], 1,
            p=[self.args.mutation_add_ratio,
               self.args.mutation_delete_ratio,
               1-self.args.mutation_add_ratio-self.args.mutation_delete_ratio]
        )

        debug_info = {}
        debug_info['old_mat'] = self.get_gene()[0]
        debug_info['old_pc'] = self.struct_tree.get_pc_relation()

        # setting some hard constraint, 
        # fixing the maximum and minimum number of nodes allowed in the body
        if len(self.get_gene()[1]) >= 10 and op == 'add':
            op = 'perturb'
        if len(self.get_gene()[1]) == 2 and op == 'remove':
            op = 'perturb'

        if op == 'add':
            perturb_info = self.perturb_add()
        elif op == 'remove':
            perturb_info = self.perturb_remove()
        elif op == 'perturb':
            perturb_info = self.perturb_attr()

        debug_info = {**debug_info, **perturb_info}

        return debug_info

    def generate_one_node(self, only_one=False):
        ''' return a list of new nodes being generated
        '''
        # different policy for adding new hierarchical structure
        if self.allow_hierarchy:
            choice = random.choice(NEW_STRUCT_OPTS)
        else:
            choice = NEW_STRUCT_OPTS[0]

        if self.args.force_symmetric:
            choice = 'l2-symmetry'
        if only_one:
            choice = 'l1-basic'
        if self.args.walker_force_no_sym:
            choice = 'l1-basic'
        
        new_node_list = []
        if choice == 'l1-basic':
            # adding one basic node
            new_node = Node(-1,
                model_gen.gen_test_node_attr(task=self.args.task, 
                    node_num=2, discrete_rv=self.discrete
                )[-1]
            )
            new_node_list.append(new_node)
        elif choice == 'l2-symmetry':
            # adding 2 symmetric nodes
            attr1, attr2 = [
                model_gen.gen_test_node_attr(task=self.args.task,
                    node_num=2, 
                    discrete_rv=self.discrete
                )[-1]
                for i in range(2)
            ]
            same_attr = ['a_size', 'b_size', 'c_size', 'geom_type',
                         'joint_range']
            for item in same_attr:
                attr2[item] = attr1[item]
            # set certain attributes
            attr2['u'] = np.pi - attr1['u']
            while attr2['u'] < 0: attr2['u'] += 2 * np.pi
            attr2['v'] = attr1['v']
            attr2['axis_x'] = attr1['axis_x']
            attr2['axis_y'] = -attr1['axis_y']
            # create the new node with corresponding attributes
            node1 = Node(-1, attr1)
            node2 = Node(-1, attr2)
            new_node_list.append(node1)
            new_node_list.append(node2)            
        else:
            raise NotImplementedError

        if self.args.force_grow_at_ends:
            for node in new_node_list:
                node.attr['axis_x'] = 1 if node.attr['axis_x'] >= 0 else -1

        return new_node_list

    def perturb_add(self, no_selfcross=False):
        '''
        '''
        debug_info = {}
        debug_info['op'] = 'add_node'

        # prepare debug info -- for updating weights in GGNN
        edge_dfs = model_gen_util.dfs_order(self.get_gene()[0])
        dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
        debug_info['old_order'] = deepcopy(dfs_order)
        debug_info['old_nodes'] = sorted(self.struct_tree.root.get_all_node_id())
        debug_info['new_nodes'] = sorted(self.struct_tree.root.get_all_node_id())

        debug_info['parent'] = []
        
        all_nodes = self.struct_tree.get_all_nodes()
        one_node_added = False

        while one_node_added == False:
            # iterate through all the nodes
            for node in all_nodes:
                not_add_node = np.random.binomial(1, 1 - self.p)
                
                if not_add_node:
                    continue

                if 'walker' in self.args.task and self.args.walker_more_constraint:
                    # check the node type and childs
                    if len(node.get_child_list()) >= 2: continue
                    if node is not self.struct_tree.get_root():
                        if len(node.get_child_list()) >= 1: continue

                one_node_added = True

                # sample a place to add the new struct
                one_node = node
                debug_info['parent'].append(one_node.get_node_id())

                use_selfcross = np.random.binomial(1, self.args.self_cross_ratio)
                if no_selfcross or not use_selfcross:
                    new_node_list = self.generate_one_node()
                else:
                    debug_info['other'] = 'using self cross'
                    self_struct = random.choice(all_nodes)
                    while self_struct is self.struct_tree.get_root():
                        self_struct = random.choice(all_nodes)
                    self_struct = deepcopy(self_struct)
                    new_node_list = [self_struct]    

                for node in new_node_list:
                    self.struct_tree.add_sub_tree(one_node, node)
                    debug_info['new_nodes'].append(-1)

        # new order, still the dfs order with new element value being -1
        prev_node_num = len(debug_info['old_order'])
        edge_dfs = model_gen_util.dfs_order(self.get_gene()[0])
        dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
        debug_info['new_order'] = deepcopy(dfs_order)
        debug_info['new_order'] = [-1 if x >= prev_node_num else x
            for x in debug_info['new_order']
        ]

        return debug_info

    def perturb_remove(self):
        ''' 
        '''
        # randomly sample a node and remove all its subtree
        debug_info = {}
        debug_info['op'] = 'rm_node'
        edge_dfs = model_gen_util.dfs_order(self.get_gene()[0])
        dfs_order = [0] + [int(edge.split('-')[-1]) for edge in edge_dfs]
        debug_info['old_order'] = deepcopy(dfs_order)
        debug_info['new_order'] = deepcopy(dfs_order)
        debug_info['old_nodes'] = sorted(self.struct_tree.root.get_all_node_id())
        debug_info['new_nodes'] = sorted(self.struct_tree.root.get_all_node_id())

        # sample the parent node to remove from
        one_node = self.struct_tree.sample_node()
        while len(one_node.get_child_list()) == 0:
            one_node = self.struct_tree.sample_node()
        
        delete_list = []
        bak_list = []
        remove_node_id_list = []
        # sample a child to start with
        child_node = random.choice(one_node.get_child_list())
        delete_list.append(child_node)
        bak_list.append(deepcopy(child_node))
        remove_node_id_list += child_node.get_all_node_id()

        if self.args.force_symmetric:
            # also find the node of the same size among the child 
            for node in one_node.get_child_list():
                if node is child_node: 
                    continue
                if node.is_same_size(child_node):
                    delete_list.append(node)
                    bak_list.append(deepcopy(node))
                    remove_node_id_list += node.get_all_node_id()
                    break

        # remove subtree from the parent
        for node_to_delete in delete_list:
            self.struct_tree.remove_sub_tree(one_node, node_to_delete)

        # if there is only root left, add everything back
        if self.struct_tree.total_num == 1:
            debug_info['op'] = 'none'
            root = self.struct_tree.sample_node()
            for bak_node in bak_list:
                self.struct_tree.add_sub_tree(root, bak_node)
            return debug_info

        debug_info['delete_node'] = []
        for rm_node_id in remove_node_id_list:
            debug_info['new_nodes'].remove(rm_node_id)
            debug_info['new_order'].remove(rm_node_id)
            debug_info['delete_node'].append(rm_node_id)
        return debug_info    

    def perturb_attr(self):
        '''
        '''
        debug_info = {}
        debug_info['op'] = 'change_attr'
        all_nodes = self.struct_tree.get_all_nodes()

        one_node_mutate = False

        while one_node_mutate == False:
            for node in all_nodes:
                # no need to perturb the root
                if node is self.struct_tree.root: continue

                if np.random.binomial(1, self.p):
                    one_node_mutate = True

                    if self.args.force_symmetric:
                        # search for the one with the same size
                        node2 = None
                        for another_node in all_nodes:
                            if another_node is node: continue
                            if another_node.is_same_size(node):
                                node2 = another_node
                        if node2 is None: 
                            import pdb; pdb.set_trace()
                    # regardless of forcing symmetry or not, we need to make the 
                    # perturbation
                    node.set_attr(
                        model_perturb.perturb_one_attr(node.get_attr(),
                            task=self.args.task,
                            perturb_discrete=self.discrete
                        )
                    )

                    if self.args.force_symmetric:
                        node2.set_attr(
                            node.make_symmetric(self.args.task,
                                self.discrete
                            ).get_attr()
                        )

                else: pass

        return debug_info

    def get_gene(self):
        '''
        '''
        adj_mat, node_attr = self.struct_tree.to_mujoco_format()
        
        if 'fish' in self.args.task:
            adj_mat[adj_mat > 0] = 7
        elif 'walker' in self.args.task or \
             'cheetah' in self.args.task or \
             'hopper' in self.args.task:
            adj_mat[adj_mat > 0] = 2

        return adj_mat, node_attr

    def get_xml(self):
        '''
        '''
        adj_mat, node_attr = self.get_gene()
        
        if 'fish' in self.args.task:
            xml_struct = model_gen.fish_xml_generator(adj_mat, node_attr, options=None)
        elif 'walker' in self.args.task:
            xml_struct = model_gen.walker_xml_generator(adj_mat, node_attr, options=None)
        elif 'hopper' in self.args.task:
            xml_struct = model_gen.hopper_xml_generator(adj_mat, node_attr, options=None)
        elif 'cheetah' in self.args.task:
            xml_struct = model_gen.cheetah_xml_generator(adj_mat, node_attr, options=None)

        xml_str = etree.tostring(xml_struct, pretty_print=True)
        return xml_struct, xml_str






if __name__ == '__main__':
    from pprint import pprint
    pprint(';')
    unit1 = UnitStruct()
    unit2 = UnitStruct(opts='l2-symmetry')
    model = ModelStruct(init_body_num=3, allow_hierarchy=True)

    # creating a tree of 5 nodes
    mynodes = [Node(i, i**2) for i in range(5)]
    mynodes[4].add_child(mynodes[0])
    mynodes[4].add_child(mynodes[2])
    mynodes[0].add_child(mynodes[3])
    mynodes[2].add_child(mynodes[1])

    # create a tree out of this
    mytree = Tree(node=mynodes[4])
    subtree = deepcopy(mynodes[4])
    one_node = mytree.sample_node()
    # testing add
    mytree.add_sub_tree(one_node, subtree)
    # testing remove
    mytree.remove_sub_tree(one_node, subtree)
    mat, attr = mytree.to_mujoco_format()

    for i in range(5):
        print('-----')
        spc = Species()
        adj, attr = spc.get_gene()
        print(adj)
        spc.perturb_add()
        adj, attr = spc.get_gene()
        print(adj)

    import pdb; pdb.set_trace()
    pass
