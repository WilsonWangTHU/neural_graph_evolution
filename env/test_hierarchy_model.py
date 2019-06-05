'''
    script to test the functionality of hierarchical model
'''

import init_path
from config.config import get_config

import os
from env import fish_env_wrapper

from env import model_gen
from util import model_gen_util
from env import hierarchy_model
from env import test_model_gen as test_func

import cv2

max_frame = 90
width = 480
height = 480

def test_hierarchy_model(args, ep_num=5, body_num=3):
    '''
    '''
    for i in range(ep_num):

        # video handler
        videoh = cv2.VideoWriter(
            str(i) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            40,
            (width * 2, height)
        )

        spc = hierarchy_model.Species(body_num=body_num)
        adj_mat, node_attr = spc.get_gene()
        xml_struct, xml_str = spc.get_xml()
        test_func.run_one_ep_given_model(args,
            adj_mat, xml_str,
            videoh=videoh
        )

    return None

def test_perturb_hierarchy(args, max_evo_step=7, body_part_num=3):
    '''
    '''
    spc = hierarchy_model.Species(args, body_num=body_part_num)

    for i in range(max_evo_step):
        print('Evolution @ %d' % i)
        videoh = cv2.VideoWriter(
            'evo' + str(i) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            40,
            (width * 2, height)
        )

        adj_mat, node_attr = spc.get_gene()
        xml_struct, xml_str = spc.get_xml()
        file_path = os.path.join(init_path.get_base_dir(), 
            'env/assets/gen/test_hierarchy_perturb.xml'
        )
        model_gen_util.xml_string_to_file(xml_str, file_path)

        test_func.run_one_ep_given_model(args,
            adj_mat, xml_str,
            videoh=videoh, max_time_step=30
        )
        debug_info = spc.mutate()
        import pdb; pdb.set_trace()
        print('Mutate option: %s' % debug_info['op'])
        pass


if __name__ == '__main__':
    args = get_config(evolution=True)

    max_ep = 5
    body_part_num = 2
    max_evo_step = 7

    # test_hierarchy_model(args, ep_num=max_ep, body_num=body_part_num)
    test_perturb_hierarchy(args, 
        max_evo_step=max_evo_step, 
        body_part_num=body_part_num
    )