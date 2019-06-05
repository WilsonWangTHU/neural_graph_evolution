'''
    this script generates the k-th best performing species in the generation
'''
import init_path

# system
import os
import glob
import argparse
import cv2
from tqdm import tqdm

# compute
import numpy as np

'''
    there are 2 training sessions serve as a good candidate
    ../evolution_data/evofish3d-speed_2018_05_08-18:06:38/
    ../evolution_data/evofish3d-speed_2018_05_08-04:17:08/
'''
max_frame = 90
width = 480
height = 480
MAX_LEN = 300

def k_th_evolution(k, rank_dir, vid_dir, vid_fn='test'):
    '''
        read the rank_info in dir
        pick the k_th performing 
    '''
    target_vid_list = []
    for fn in glob.glob(rank_dir + '/*'):

        if fn.endswith('rank_info.npy') == False: continue
        gen = fn.split('/')[-1].split('_')[0]

        data = np.load(fn).item()

        info = {}
        info['gen'] = gen
        info['SpcID'] = data['SpcID'][k]
        info['AvgRwd'] = data['AvgRwd'][k]
        target_vid_list.append( (gen, info['SpcID'], info) )

    target_vid_list = sorted(target_vid_list, key=lambda x: int(x[0]))
    
    # video handler
    videoh = cv2.VideoWriter(
            str(vid_fn) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            40,
            (width * 2, height)
        )

    # according to the order in target_vid_list, find the corresponding video
    print('Examine the video files available')
    for candidate in tqdm(target_vid_list):
        gen, spcid, info = candidate
        vid_text = 'Gen:%s-Spc:%d-R:%.2f' % (gen, spcid, info['AvgRwd'])

        video_exist = False
        for vid_file in glob.glob(vid_dir + '/*'):
            if vid_file.endswith('5.mp4') == False: continue
            file = vid_file.split('/')[-1]
            v_gen = file.split('_')[0]
            v_spc = file.split('_')[1]
            if int(gen) == int(v_gen) and int(v_spc) == int(spcid):
                video_exist = True
                break
            else: continue
        if video_exist == False: continue

        cap = cv2.VideoCapture(vid_file)
        i = 0
        while(cap.isOpened()) and i < MAX_LEN:
            ret, frame = cap.read()
            
            try:
                frame = cv2.putText(frame.copy(), vid_text, (30,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255,255,255), 3, cv2.LINE_AA
                    )
            except:
                if frame == None:
                    break
            i += 1
            videoh.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()



    videoh.release()
    return

def get_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--trn_sess', type=str, default=None)
    parser.add_argument('--k_rank', type=int, default=0)
    parser.add_argument('--vid_name', type=str, default='test')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = get_config()

    rank_dir = os.path.join(args.trn_sess, 'species_data')
    vid_dir = os.path.join(args.trn_sess, 'species_video')

    k_th_evolution(args.k_rank, rank_dir, vid_dir, vid_fn=args.vid_name)