import os
from easydict import EasyDict

cfg = EasyDict()

cfg.train_dir = 'train_log'
cfg.models_dir = 'models'
cfg.data_dir = 'datasets'

cfg.batch_size_per_gpu = 16
cfg.w1 = 112
cfg.h1 = 112
cfg.w2 = 48
cfg.h2 = 48
cfg.c1 = 1
cfg.c2 = 1
cfg.lmd = 5e-7
cfg.lr = 0.001
cfg.decay = 0.96
cfg.max_epoch = 30

if not os.path.exists(cfg.train_dir):
    os.makedirs(cfg.train_dir)

if not os.path.exists(cfg.models_dir):
    os.makedirs(cfg.models_dir)

if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)

