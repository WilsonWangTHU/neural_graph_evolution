# ------------------------------------------------------------------------------
#   @brief:
#       In this file we init the path
#   @author:
#       Written by Tingwu Wang, 2016/Sep/22
# ------------------------------------------------------------------------------


import os.path as osp
import sys
import datetime
import pdb


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# get current time
running_start_time = datetime.datetime.now()
time = str(running_start_time.strftime("%Y_%m_%d-%X"))

# get current file path
_this_dir = osp.dirname(__file__)
_base_dir = osp.join(_this_dir, '..')
add_path(_base_dir)


def init_path():
    ''' function to be called in the beginning of the file
    '''
    _this_dir = osp.dirname(__file__)
    _base_dir = osp.join(_this_dir, '..')
    add_path(_base_dir)
    return None


def bypass_frost_warning():
    return 0


def get_base_dir():
    return _base_dir


def get_time():
    return time


def get_abs_base_dir():
    return osp.abspath(_base_dir)
