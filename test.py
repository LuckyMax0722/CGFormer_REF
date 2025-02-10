import os
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model_v2 import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# python /u/home/caoh/projects/MA_Jiachen/CGFormer/main.py --eval --ckpt_path /u/home/caoh/projects/MA_Jiachen/CGFormer/ckpts/CGFormer-Efficient-Swin-SemanticKITTI.ckpt --config_path /u/home/caoh/projects/MA_Jiachen/CGFormer/configs/CGFormer-Efficient-Swin-SemanticKITTI.py --log_folder /u/home/caoh/projects/MA_Jiachen/CGFormer/output --seed 7240 --log_every_n_steps 100
# python /u/home/caoh/projects/MA_Jiachen/CGFormer/test.py --config_path /u/home/caoh/projects/MA_Jiachen/CGFormer/configs/CGFormer-Efficient-Swin-SemanticKITTI.py --log_folder /u/home/caoh/projects/MA_Jiachen/CGFormer/output --seed 7240 --log_every_n_steps 100

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()

    seed = config.seed
    pl.seed_everything(seed)

    model = pl_model(config)
    
    
    #print(model)
    

