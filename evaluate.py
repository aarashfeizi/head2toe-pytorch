import torch
import utils
import datasets
from models import finetune
import wandb
import os


def main():
    args = utils.get_args()

    if args.wandb:
        args = utils.init_wandb(args)
    
    print(args)
    utils.seed_all(args.seed, args.cuda)

    model = finetune.FineTune(args=args, backbone='resnet50')
    
    if args.cuda:
        model.cuda()

    train_data = utils.get_dataset(args=args, mode='train')
    if args.train_to_val_ratio_split == 0:
        val_data = utils.get_dataset(args=args, mode='val')
    else:
        val_data = None


    model.optimize_finetune(train_loader=train_data,
                            val_loader=val_data)


    # import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    main()