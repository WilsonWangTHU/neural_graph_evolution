#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       Build the nervenet placeholders. Basically, we now have a different
#       setting as [nstep, minibatch_size / nstep]
#   @author:
#       Tingwu Wang, August. 13th, 2018
# -----------------------------------------------------------------------------

import numpy as np


def nervenetplus_step_assign(rollout_data, nstep):
    data_batch_id = []  # the start of nstep subsamples
    current_pos = 0
    for i_episode in rollout_data:
        episode_length = len(i_episode['rewards'])
        start_pos = np.random.randint(nstep) + current_pos

        # length = 6, start_pos = 1, nstep = 5 --> [1,2,3,4,5]: num_pos = 1
        # print((episode_length + current_pos - start_pos) / nstep)
        num_pos = int(
            np.floor((episode_length + current_pos - start_pos) / nstep)
        )

        if num_pos == 0:
            start_ids = [0]
        elif num_pos < 0:
            start_ids = []
        else:
            start_ids = [i_pos * nstep + start_pos
                         for i_pos in range(num_pos)]
        data_batch_id.extend(start_ids)
        current_pos += episode_length

    return data_batch_id, len(data_batch_id)
