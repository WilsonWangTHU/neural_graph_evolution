# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       generate the videos into the same directory
# -----------------------------------------------------------------------------
import numpy as np
import argparse
import glob
import cv2
import os
import init_path
from tqdm import tqdm
# import matplotlib.pyplot as plt


def get_candidates(args):
    # base_path for the base, candidate_list for the topology data
    # case one: plot all species, or one of the species
    #   XX/species_topology
    # case two: plot the top ranked_species
    #   XX/species_data
    # case three: plot the top ranked_species's video
    #   XX/species_video

    if args.file_name.endswith('.npy'):
        candidate_list = [args.file_name]
    else:
        candidate_list = glob.glob(os.path.join(args.file_name, '*.npy'))
        candidate_list = [i_candidate for i_candidate in candidate_list
                          if 'rank_info' not in i_candidate]

    if 'species_topology' in args.file_name:
        species_topology_list = candidate_list

    elif 'species_data' in args.file_name:
        species_topology_list = candidate_list
    else:
        assert 'species_video' in args.file_name
        species_topology_list = [
            os.path.join(
                os.path.dirname(i_candidate).replace('species_video',
                                                     'species_topology'),
                os.path.basename(i_candidate).split('_')[1] + '.npy'
            )
            for i_candidate in candidate_list
        ]

    task = os.path.abspath(candidate_list[0]).split(
        init_path.get_abs_base_dir()
    )[1].split('/')[2].split('_')[0]
    task = task.replace('/', '')


    return candidate_list, species_topology_list, task


if __name__ == '__main__':
    '''
        @brief:
            Either plot the directory, or simply one npy file
    '''

    parser = argparse.ArgumentParser(description="Plot results from a dir")
    parser.add_argument(
        "-i", "--file_name", type=str, required=True,
        help="The directory of the summary file"
    )

    parser.add_argument(
        "-s", "--size", type=int, required=False, default=480
    )
    # temporary fix
    parser.add_argument(
        '--fish_target_angle', type=float, required=False, default=np.pi / 6
    )
    parser.add_argument(
        '--fish_circle_radius', type=float, required=False, default=0.5
    )
    parser.add_argument(
        '--walker_ctrl_coeff', type=float, required=False, default=0.001
    )

    parser.add_argument('--fish_target_speed', type=float, default=0.002,
                        help='the speed of the target')

    parser.add_argument(
        "-v", "--video", type=int, required=True, default=0,
        help='whether to generate the videos or just images'
    )
    parser.add_argument(
        "-l", "--video_length", type=int, required=False, default=100,
    )

    args = parser.parse_args()


    candidate_list, species_topology_list, task = get_candidates(args)
    if 'finetune' in candidate_list[0]:
        args.optimize_creature = True
    else:
        args.optimize_creature = False
    args.task = task

    for i_id, candidate in enumerate(tqdm(candidate_list)):
        # process each environment
        data = np.load(candidate)
        if data.dtype != np.float:
            data = data.item()
        # TODO:
        try:
            if 'species_video' in args.file_name:
                
                topology_data = np.load(species_topology_list[i_id]).item()
                if candidate.endswith('5.npy') == False: continue
            elif 'species_data' in args.file_name:
                topology_data = np.load(candidate_list[i_id]).item()
            elif 'species_topology' in args.file_name:
                topology_data = np.load(candidate).item()
            else:
                assert 0, 'The input filename is not valid.'
        except Exception as e:
            print(e)
            continue

        if 'fish' in task:
            import fish_env_wrapper
            env = fish_env_wrapper.dm_evofish3d_wrapper(args=args, rand_seed=1, monitor=0, 
                adj_matrix=topology_data['adj_matrix'], xml_str=topology_data['xml_str'])
            # @HZ: a very hacky way to change the upper camera view
            # import pdb; pdb.set_trace()
            topology_data['xml_str'] = topology_data['xml_str'].decode('utf-8').replace(
                    '<camera mode=\"trackcom\" name=\"tracking_top\" pos=\"0 0 1\" xyaxes=\"1 0 0 0 1 0\"',
                    '<camera mode=\"trackcom\" name=\"tracking_top\" pos=\"-.1 .2 .2\" xyaxes=\"-2-1 0 -0 -.5 1\"')
    
            env = fish_env_wrapper.dm_evofish3d_wrapper(
                args=args, rand_seed=1, monitor=0,
                adj_matrix=topology_data['adj_matrix'],
                xml_str=topology_data['xml_str']
            )

        elif 'walker' in task or 'cheetah' in task or 'hopper' in task:
            from env import walker_env_wrapper
            env = walker_env_wrapper.dm_evowalker_wrapper(
                args=args, rand_seed=1, monitor=0,
                adj_matrix=topology_data['adj_matrix'],
                xml_str=topology_data['xml_str']
            )
        else:
            assert 0

        action_size = env.env.action_spec().shape[0]

        if args.video:

            # check generation
            gen_num = int(candidate.split('/')[-1].split('_')[0])
            if gen_num % 10 == 0 or gen_num <= 5:
                pass
            else:
                continue

            # save the videos
            if os.path.exists(candidate.replace('.npy', '.mp4')):
                continue
            video = cv2.VideoWriter(
                candidate.replace('.npy', '.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                40, (args.size * 2, args.size)
            )

            env.reset()
            if 'species_video' in candidate:
                # recover the videos
                for i_frame in range(min(len(data), args.video_length)):
                    with env.env.physics.reset_context():
                        if 'fish' in args.task:
                            # set the target position
                            env.env.physics.named.model.geom_pos['target', 'x'] = \
                                data[i_frame][-2]
                            env.env.physics.named.model.geom_pos['target', 'y'] = \
                                data[i_frame][-1]
                            env.env.physics.data.qpos[:] = data[i_frame][:-2]
                        else:
                            env.env.physics.data.qpos[:] = data[i_frame]

                    image = np.hstack(
                        [env.env.physics.render(args.size, args.size, camera_id=0),
                         env.env.physics.render(args.size, args.size, camera_id=1)]
                    )
                    # rgb to bgr
                    image = image[:, :, [2, 1, 0]]
                    video.write(image)
            else:
                # generate videos using random actions
                for i_frame in range(args.video_length):
                    env.step((np.random.rand(action_size) - 0.5) * 2)

                    image = np.hstack(
                        [env.env.physics.render(args.size, args.size, camera_id=0),
                         env.env.physics.render(args.size, args.size, camera_id=1)]
                    )
                    # rgb to bgr
                    image = image[:, :, [2, 1, 0]]
                    video.write(image)

            video.release()
        else:
            # save the screenshot of the species
            if os.path.exists(candidate.replace('.npy', '.png')):
                continue
            env.reset()
            for _ in range(30):
                env.step((np.random.rand(action_size) - 0.5) * 2)
            image = np.hstack(
                [env.env.physics.render(args.size, args.size, camera_id=0),
                 env.env.physics.render(args.size, args.size, camera_id=1)]
            )
            # rgb to bgr
            image = image[:, :, [2, 1, 0]]
            cv2.imwrite(candidate.replace('.npy', '.png'), image)
