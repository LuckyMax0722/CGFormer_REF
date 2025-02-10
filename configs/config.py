import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/u/home/caoh/projects/MA_Jiachen/CGFormer'  # TODO: Change path to your SGN-dir

# config
CONF.PATH.CONFIG_DIR = os.path.join(CONF.PATH.BASE, 'configs')
CONF.PATH.CONFIG = os.path.join(CONF.PATH.CONFIG_DIR, 'CGFormer-Efficient-Swin-SemanticKITTI_v3.py')

# log
CONF.PATH.LOG_DIR = os.path.join(CONF.PATH.BASE, 'output')

# ckpt
CONF.PATH.CKPT_DIR = os.path.join(CONF.PATH.BASE, 'ckpts')
CONF.PATH.CKPT = os.path.join(CONF.PATH.CKPT_DIR, 'CGFormer-Efficient-Swin-SemanticKITTI.ckpt')