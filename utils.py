import torch.nn as nn
import torch
import math
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse
import json
import os
import wandb
import random

wandb_dict = {}

def init_wandb(args):
    wandb.init(config=args, dir=os.path.join(args.log_path, 'wandb/'))
    args = wandb.config
    return args

def wandb_log():
    global wandb_dict
    if len(wandb_dict) > 0:
        wandb.log(wandb_dict)
    wandb_dict = {}


def wandb_update_value(names_values_dict):
    for (name, value) in names_values_dict.items():
        wandb_dict[name] = value

# SHARING_STRATEGY = "file_system"
# torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
# def seed_worker(worker_id):
#     torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def seed_all(seed, cuda=False):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return True

BACKBONE_MODES = ['supervised',
                  'swav',
                  'vicreg',
                  'barlow',
                  'dino',
                  'simsiam',
                  'byol',
                  'simclr',
                  'unigrad']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config/cifar100.json', type=str, help='Path to config file')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--fraction', default=None, type=float, help='Learning rate')
    parser.add_argument('--loss_gl_coeff', default=None, type=float, help='Group Lasso coefficient')
    parser.add_argument('--backbone_mode', default='supervised', type=str, help='Path to config file', choices=BACKBONE_MODES)


    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        cfg_dict = json.load(f)

    if args.lr:
      cfg_dict['lr'] = args.lr
    
    if args.loss_gl_coeff:
      cfg_dict['loss_gl_coeff'] = args.loss_gl_coeff  

    args = argparse.Namespace(**cfg_dict)

    return args


def __create_namespace(cfg_dict):
    new_dict = {}
    for k, v in cfg_dict.items():
        if type(v) == dict:
            new_dict[k] = __create_namespace(v)
        else:
            new_dict[k] = v
    cfg_ns = argparse.Namespace(**new_dict)
    return cfg_ns

def flatten_and_concat(output_dict, pool_size=0, target_size=0,
                       cls_token_pool='normal'):

  """Summarizes a dict of outputs into single feature vector."""
  # If target_size is given pool_size is ignored.
  if cls_token_pool not in ('normal', 'only_cls', 'nopool_cls'):
    raise ValueError("%s must be one of 'normal', 'only_cls', 'nopool_cls'"
                     % cls_token_pool)
  all_features = []
  for k, output in output_dict.items():
    # output = output_dict[k]
    # TODO Make this more readable by making each branch a function.
    if len(output.shape) == 4:
      if target_size > 0:
        # Overwrite pool size so that final output matches target_size as close
        # as possible.
        _, channels, _, width = output.shape
        if channels >= target_size:
          # Global pool.
          pool_size = 0
        else:
          # Assuming square image.
          n_patches_per_row = int(math.sqrt(target_size // channels))
          pool_size = width // n_patches_per_row
      if pool_size > 0:
        output = nn.AvgPool2d(
            kernel_size=pool_size, stride=pool_size)(output)
        all_features.append(nn.Flatten()(output))
      else:
        # Global pool
        all_features.append(torch.mean(output, dim=(2, 3)))
    # elif len(output.shape) == 3:
    #   if cls_token_pool == 'only_cls':
    #     output = output[:, 0, :]
    #   else:
    #     if cls_token_pool == 'nopool_cls':
    #       # We will get the cls as it is and pool the rest.
    #       cls_output, output = output[:, 0, :], output[:, 1:, :]
    #     if target_size > 0:
    #       # Overwrite pool size so that final output matches target_size as
    #       # close as possible.
    #       _, n_token, channels = output.shape
    #       if channels >= target_size:
    #         # Global pool.
    #         pool_size = 0
    #       else:
    #         # Assuming square image.
    #         n_groups = target_size / channels
    #         pool_size = int(n_token / n_groups)
    #     if pool_size > 0:
    #       output = tf.keras.layers.AveragePooling1D(
    #           pool_size=pool_size, strides=pool_size)(output)
    #       output = tf.keras.layers.Flatten()(output)
    #     else:
    #       # Global pool
    #       output = tf.reduce_mean(output, axis=[1])
    #     if cls_token_pool == 'nopool_cls':
    #       output = tf.concat([cls_output, output], axis=1)
    #   all_features.append(output)
    elif len(output.shape) == 2:
      all_features.append(output)
    else:
      raise ValueError(
          f'Output tensor: {k} with shape {output.shape} not 2D or 4D.')
  return all_features


def get_dataset(args, mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
                                        # transforms.RandomResizedCrop(size=args.img_size),
                                        # transforms.RandomHorizontalFlip(p=0.5),
                                        # transforms.ColorJitter(),
                                        transforms.Resize(size=int(args.img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=args.normalize_param['mean'], std=args.normalize_param['std'])])
    else:
        transform = transforms.Compose([transforms.Resize(size=int(args.img_size * 1.15)),
                                        transforms.CenterCrop(size=args.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=args.normalize_param['mean'], std=args.normalize_param['std'])])

    data = DataLoader(dataset=datasets.get_dataset(args, transform=transform, mode=mode),
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    return data

def make_dirs(path):
  if os.path.exists(path):
    return
  else:
    os.makedirs(path)
    return

def save_np(f_importance, save_path):
  make_dirs(save_path)
  np.save(os.path.join(save_path, 'feature_importance.npy'), f_importance)