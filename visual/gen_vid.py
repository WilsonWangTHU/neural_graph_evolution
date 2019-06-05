'''
    utility file for generating videos in <evolution_data> directory
'''

import sh
import os
import subprocess

if __name__ == '__main__':
    for directory in os.listdir('../evolution_data'):
        print(directory)
        cmd = 'python ../env/visualize_species.py -i ../evolution_data/%s/species_video -v 1' % (directory)
        os.system(cmd)
