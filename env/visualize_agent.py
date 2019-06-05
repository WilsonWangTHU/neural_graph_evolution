# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       generate the videos into the same directory
# -----------------------------------------------------------------------------
import env_wrapper
import numpy as np
import argparse
import glob
import cv2
import os
# import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
        @brief:
            Either plot the directory, or simply one npy file
    '''

    parser = argparse.ArgumentParser(description="Plot results from a dir")
    parser.add_argument(
        "-i",
        "--file_name",
        type=str,
        required=True,
        help="The directory of the summary file"
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=False,
        default=480
    )

    args = parser.parse_args()

    # file list
    if args.file_name.endswith('.npy'):
        candidate_list = [args.file_name]
    else:
        candidate_list = glob.glob(os.path.join(args.file_name, '*.npy'))

    # make the environment
    env_name = os.path.abspath(candidate_list[0]).split('/')[-2].split('_20')[0]
    args.task = env_name.replace('IM', '')  # use the original environment
    env, is_deepmind_env = env_wrapper.make_env(
        args=args, rand_seed=1, allow_monitor=0
    )

    for candidate in candidate_list:
        # process each environment
        data = np.load(candidate)

        if os.path.exists(candidate.replace('.npy', '.mp4')):
            continue
        video = cv2.VideoWriter(
            candidate.replace('.npy', '.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            40,
            (args.size * 2, args.size)
        )

        env.reset()
        for i_frame in range(len(data)):
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

        video.release()
