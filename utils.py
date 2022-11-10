import torch.nn as nn
import torch
import math
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='config/cifar100.json', type=str, help='Path to config file')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        cfg_dict = json.load(f)

    args = __create_namespace(cfg_dict)
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
        transform = transforms.Compose([transforms.RandomResizedCrop(size=args.data.img_size),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ColorJitter(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=args.data.normalize_param.mean, std=args.data.normalize_param.std)])
    else:
        transform = transforms.Compose([transforms.Resize(size=int(args.data.img_size * 1.15)),
                                        transforms.CenterCrop(size=args.data.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=args.data.normalize_param.mean, std=args.data.normalize_param.std)])

    data = DataLoader(dataset=datasets.get_dataset(args, transform=transform, mode=mode),
                        batch_size=args.env.batch_size,
                        shuffle=True,
                        num_workers=args.env.num_workers,
                        pin_memory=True)

    return data


def get_embedding(model, dataloader, cuda=True):
    model.eval()
    batch_embedding_lists = []
    labels = []
    with tqdm(total=len(dataloader), desc="Getting head2toe embeddings...") as t:
        for idx, batch in enumerate(dataloader):
            x, l = batch
            if cuda:
                x = x.cuda()
            out = model(x)
            batch_embedding_lists.append(list(out.values())) # should I detach?
            labels.append(l)

            t.update()

    labels = torch.concat(labels)
    output_embeddings = []
    for i in range(len(batch_embedding_lists[0])):
      embedding_i = [batch[i] for batch in batch_embedding_lists]
      output_embeddings.append(torch.concat(embedding_i, dim=0))

    return output_embeddings, labels