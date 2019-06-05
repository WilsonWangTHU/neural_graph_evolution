''' this script takes in a remote directory
    henryzhou@<server>:<path to /evolution/trn_sess> 
    to local ../evolution_data
    select the top performing structures and bursts its images
'''
import init_path
import argparse

# system import
import os
import time
import glob
import subprocess

# computation
import numpy as np


def get_config():
    '''
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--task', required=True, default=None,
                        choices=['visual', 'graph'])
    parser.add_argument('--remote_dir', type=str, required=True, default='')
    parser.add_argument('--video', action='store_true', default=False)

    args = parser.parse_args()

    return args

def sync_dir(remote_dir, synced_flag=False):
    '''
    '''
    trn_sess_name = remote_dir.split('/')[-1]
    local_dir = os.path.join( init_path.get_base_dir(), 
                              'evolution_data')

    print('Local directory', local_dir)
    command = 'rsync -avz %s %s/ --exclude=\'*mp4\' --exclude=\'*png\' --delete' % \
        (remote_dir, local_dir)
    # command = 'sshfs %s %s' % (remote_dir, local_dir)
    print(command)
    cur_time = time.time()
    if not synced_flag:
        pass
        # momentarily commenting it off
        # process = subprocess.Popen(command,
        #     shell=True#, stdout=subprocess.PIPE
        # )
        # process.wait()
    
    # assert process.returncode, 'Command failure: \'%s\'' % command
    print('Syncing takes', time.time() - cur_time)
    local_dir = os.path.join(local_dir, trn_sess_name)
    return local_dir

def parse_rank_info(local_dir, topk=5):
    ''' parse the rank information and select the top-performing species
    return a list of species that we are interested in for generating images
    '''
    data_dir = os.path.join(local_dir, 'species_data')

    generation_num = 0
    selected_species = []
    for filename in glob.glob(data_dir + '/*'):
        if 'rank_info' not in filename: continue
        generation_num += 1
        rank_info = np.load(filename).item()

        for i in range(topk):
            selected_species.append( (rank_info['SpcID'][i], 
                                      rank_info['AvgRwd'][i]) )
    print('Generation examined %d' % generation_num)
    selected_species = sorted(selected_species, key=lambda x: x[1])
    selected_species = set([x[0] for x in selected_species])

    return list(selected_species)

def burst_images(args, local_dir, spc_candidates):
    '''
    '''

    
    visualize_script = os.path.join(init_path.get_base_dir(),
                                    'env',
                                    'visualize_species.py')

    if args.video == False:
        topology_dir = os.path.join(local_dir, 'species_topology')
        for spc_id in spc_candidates:
            topology_file = '%s/%d.npy' % (topology_dir, spc_id)
            burst_image_cmd = 'python %s -i %s -v 0' % \
                (visualize_script, topology_file)
            process = subprocess.Popen(burst_image_cmd,
                shell=True
            )
            process.wait()
    else:
        video_dir = os.path.join(local_dir, 'species_video')
        for filename in glob.glob(video_dir + '/*'):
            v_id = filename.split('/')[-1].split('.')[0]
            gen_id, spc_id, ep_num = [int(x) for x in v_id.split('_')]
            
            if ep_num != 5: continue
            if spc_id not in spc_candidates: continue

            burst_video_cmd = 'python %s -i %s -v 1 -l 1000' % \
                (visualize_script, filename)
            process = subprocess.Popen(burst_video_cmd,
                shell=True
            )
            process.wait()

    return None

if __name__ == '__main__':

    args = get_config()

    if args.task == 'visual':
        local_dir = sync_dir(args.remote_dir)
        species_list = parse_rank_info(local_dir)
        burst_images(args, local_dir, species_list)
    elif args.task == 'graph':
        raise NotImplementedError
    else:
        raise RuntimeError('task %s not supported' % args.task)

