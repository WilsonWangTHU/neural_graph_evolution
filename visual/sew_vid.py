'''
    utility file for sewing generated images
'''
import os
import sh
import subprocess

if __name__ == '__main__':

    for directory in os.listdir('../evolution_data'):
        print(directory)
        cmd = 'python evo_video.py --trn_sess ../evolution_data/%s --vid_name %s' \
                % (directory, directory)
        os.system(cmd)
