'''
    @brief
        utility functions can be used for model gen
        (mostly just some analytical geometry)
    @author:
        Henry Zhou, April 1st, 2018
'''

import pdb
import numpy as np
import math
import random

def gaussian_noise(mu, std, discrete, step_size=None,):
    '''
    '''
    if discrete and (step_size is None):
        raise RuntimeError('Gaussian noise cannot handle discrete without a step')

    if not discrete:
        noise = float(np.random.normal(mu, std, 1))
        return noise

    # sample by a step size if discrete
    noise = int(np.around( np.random.normal(0, 1, 1) ))
    return noise * step_size

def get_uniform(low, high, discrete=True, total_lvl=6, avoid_zero=True):
    ''' uniformly sample from [low, high]
    if given discrete, it will sample from [low, high] at total_lvl
    steps
    '''
    assert total_lvl > 0, 'Invalid number of step %d' % total_lvl
    
    if discrete:
        step = (high - low) / total_lvl
        val = low + step * random.randint(1, total_lvl)
    else:
        val = float( np.random.uniform(low, high, 1) )
    
    if avoid_zero:
        val += 1e-9
    return val

def xml_string_to_file(xml, filename):
    '''
    '''
    if filename == None:
        return 

    with open(filename, 'wb') as fd:
        fd.write(xml)
    return


def leaf_list(adj_mat):
    ''' @brief:
            given an adjacent matrix,
            return a list of all the nodes ordered by the node id
    '''
    N, _ = adj_mat.shape

    leaf_list = []
    for i in range(N):
        isleaf = True
        # check its child
        for j in range(i, N):
            if adj_mat[i, j] != 0:
                isleaf = False
                break
        if isleaf: leaf_list.append(i)
    return leaf_list



def vectorize_ellipsoid(a, b, c, u, v):
    '''
        ellipsoid parameterization
        use 2 angles to determine a point on the ellipsoid
        return (x, y, z) coordinate of the point
    '''
    x = float(a * np.cos(u) * np.sin(v))
    y = float(b * np.sin(u) * np.sin(v))
    z = float(c * np.cos(v))
    return (x, y, z)

def ellipsoid_normal_vec(a, b, c, p_coord):
    '''
        input:
            given a numpy array p_coord = (x, y, z)
            find the normal vector to the ellipsoid at the point
            NOTE: the point must be on the ellipsoid
        output:
            a numpy array (vec_x, vec_y, vec_z)
            a vector that represents the line
    '''
    # verify the point is on the ellipsoid
    x, y, z = p_coord.tolist()
    assert np.isclose(1, x**2/a**2 + y**2/b**2 + z**2/c**2, atol=1e-6)
    # partial derivative of the ellipsoid function
    vec_x = 2 * x / a**2
    vec_y = 2 * y / b**2
    vec_z = 2 * z / c**2
    vec = np.array([vec_x, vec_y, vec_z])
    # normalize the vector at the end
    return vec / (np.linalg.norm(vec) + 1e-7)


def get_child_given_parent(adj_mat, parent):
    ''' @brief:
            given the adjacent matrix, and a node_id 
            find all the node ids that is the child of the parent node
    '''
    N, _ = adj_mat.shape
    assert parent >= 0, parent < N
    child_list = None

    p_connection = np.copy(adj_mat[parent])
    p_connection[:parent] = 0

    child_list = np.where(p_connection > 0)[0]
    return child_list.tolist()


def dfs_order(adj_mat):
    '''
        return the order of parent-child relationship in the tree structure
        described by the adj_matrix
        input:
            1. adj_mat an N x N matrix in which index 0 is the root
            2. the matrix is symmetric
        output:
            1. a list of '%d-%d's decribing the order of parent-child relationship
               run using dfs
    '''
    def dfs_order_helper(adj_mat, node_id, cur_order):
        '''
        '''
        # get child_list
        node_row = np.copy(adj_mat[node_id])
        for i in range(node_id + 1): node_row[i] = 0
        child_list = np.where(node_row)[0].tolist()

        for child_node in child_list:
            edge = '%d-%d' % (node_id, child_node)
            cur_order.append(edge)
            dfs_order_helper(adj_mat, child_node, cur_order)
        
        return cur_order

    # using recursion to solve the problem
    dfs_order = []
    dfs_order = dfs_order_helper(adj_mat, 0, dfs_order)
    return dfs_order 


def mirror_mat(mat):
    ''' input:  1. numpy array matrix of N x N size
        output: 1. matrix of N x N size
                   keep the other half of the original mat
                   and make it symmetric along the diagonal line
    '''
    N, _ = mat.shape
    assert N == _, 'Not a square matrix, it cannot be symmetric!'

    for i in range(N):
        for j in range(i, N):
            mat[j, i] = mat[i, j]
    return mat

def get_encoding(val):
    ''' from a value (0-7) to binary one-hot encoding
    eg. input 6
        output [1, 1, 0] 
    '''
    onehot = [int(item) for item in list(bin(val)[2:])]
    onehot = [0 for i in range(3-len(onehot))] + onehot
    onehot = list(reversed(onehot))
    return onehot

def point_in_ellipsoid(a, b, c, p):
    ''' check whether point p = (dx, dy, dz) is in the ellipsoid with param (a, b, c)
        output: True -- point in the ellipsoid
                False -- point outside the ellipsoid
    '''
    dx, dy, dz = p.tolist()
    val = dx**2/a**2 + dy**2/b**2 + dz**2/c**2
    if val > 1: return False
    else: return True