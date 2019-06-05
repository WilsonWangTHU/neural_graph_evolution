'''
    script to test the functionality of model_gen
'''


# my imports
import init_path
from config.config import get_config
# import os
import pdb
from env import fish_env_wrapper
from env import walker_env_wrapper
# import lxml
import lxml.etree as etree
import numpy as np

# local imports
from env import model_gen
from env import model_perturb
from env import hierarchy_model

import cv2


max_frame = 90
width = 480
height = 480


def run_one_ep_given_model(args, adj_matrix, xml_str, videoh=None, max_time_step=100):
    ''' run one episode
    '''
    print('Start running 1 episode of the simulation')
    # load the environment
    if 'walker' in args.task or 'hopper' in args.task or 'cheetah' in args.task:
        env = walker_env_wrapper.dm_evowalker_wrapper(args, 1, args.monitor,
                                                      adj_matrix=adj_matrix, xml_str=xml_str)
    elif 'fish' in args.task:
        env = fish_env_wrapper.dm_evofish3d_wrapper(args, 1, args.monitor,
                                                    adj_matrix=adj_matrix, xml_str=xml_str)
    action_spec = env.env.action_spec()

    ob = env.reset()
    for _ in range(max_time_step):
        # perform random action
        action= np.random.uniform(action_spec.minimum, 
                                  action_spec.maximum,
                                  size=action_spec.shape)
        # try: 
        #     ob, reward, done, _ = env.step(action)
        # except Exception as e:
        #     pdb.set_trapce()
        #     pass
        # import pdb; pdb.set_trace()
        ob, reward, done, _ = env.step(action)
        print(reward)
        # print(ob, reward, done)

        if done: break

        # visualize the agent
        if videoh is not None:
            frame = np.hstack([env.env.physics.render(height, width, camera_id=0),
                               env.env.physics.render(height, width, camera_id=1)])
            image = frame[:,:,[2, 1, 0]]
            image = cv2.putText(image.copy(), args.task, (30,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255,255,255), 3, cv2.LINE_AA)
            cv2.imshow('Image', image)
            cv2.waitKey(4)
            videoh.write(image)
    print('')
    if videoh is not None: videoh.release()
    return None

def test_model_gen(args, body_part_num):
    ''' singularly test for the generation of the model based on the given adj_matrix
    '''
    for num_episode in range(30):
        
        adj_matrix = model_gen.gen_test_adj_mat(hinge_opt=7, shape=(body_part_num, body_part_num))
        node_attr  = model_gen.gen_test_node_attr(task=args.task, node_num=body_part_num)

        species = hierarchy_model.Species(args, body_num=body_part_num)

        if 'walker' in args.task:
            adj_matrix[adj_matrix > 0] = 2
            xml_struct = model_gen.walker_xml_generator(adj_matrix, node_attr, filename='walker_test')
        elif 'fish' in args.task:
            xml_struct = model_gen.fish_xml_generator(adj_matrix, node_attr, filename='fish_test')
        xml_str = etree.tostring(xml_struct)

        videoh = cv2.VideoWriter(
            str(num_episode) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            40,
            (width * 2, height)
        )

        run_one_ep_given_model(args, adj_matrix, xml_str, videoh, max_time_step=400)
    return None

def test_model_big(args, body_part_num, max_episode=100):
    ''' exhaustively generate and load the xml file
    '''
    for i_ep in range(max_episode):
        print('Generating %6d episode...' % (i_ep))
        
        adj_matrix = model_gen.gen_test_adj_mat(hinge_opt=7, shape=(body_part_num, body_part_num))
        node_attr  = model_gen.gen_test_node_attr(node_num=body_part_num)
        xml_struct = model_gen.fish_xml_generator(adj_matrix, node_attr, filename='fish_big_test')
        xml_str = etree.tostring(xml_struct)

        run_one_ep_given_model(args, adj_matrix, xml_str, max_time_step=1)

    return None


def test_perturb(args, max_evo_step, body_part_num):
    '''
    '''
    # initialize 
    adj_matrix = model_gen.gen_test_adj_mat(hinge_opt=7, shape=(body_part_num, body_part_num))
    node_attr  = model_gen.gen_test_node_attr(node_num=body_part_num)

    for num_evo in range(max_evo_step):
        print('Evolution @ ' + str(num_evo))
        xml_struct = model_gen.xml_generator(adj_matrix, node_attr, filename='evo'+str(num_evo))
        xml_str = etree.tostring(xml_struct)

        videoh = cv2.VideoWriter(
            'evo' + str(num_evo) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            40,
            (width * 2, height)
        )

        run_one_ep_given_model(args, adj_matrix, xml_str, videoh=videoh, max_time_step=30)

        # perturb the evolution
        adj_matrix, node_attr, debug_info = model_perturb.perturb_topology(
                adj_matrix, node_attr,
                np.random.randint(3)
        )
        import pdb; pdb.set_trace()
        pass

    return None

if __name__ == '__main__':
    
    body_part_num = 3
    max_evo_step = 7

    args = get_config(evolution=True)
    print('Base dir: {}'.format(init_path.get_abs_base_dir()))

    test_model_gen(args, body_part_num)
    # test_perturb(args, max_evo_step, body_part_num)
    # import pdb; pdb.set_trace()
    # test_model_big(args, body_part_num, max_episode=1000)
