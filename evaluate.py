import torch
import utils
import datasets
from models import finetune, finetune_fs
import wandb
import os


def main():
    args = utils.get_args()
    save_path = args.log_path
    if args.wandb:
        args = utils.init_wandb(args)
        r = wandb.run
        time = str(r.start_time)
        time = time[:time.find('.')]
        save_path = os.path.join(save_path, f'{r.id}, {r.name}_{time}')
        utils.make_dirs(save_path)
    

    
    print(args)
    utils.seed_all(args.seed, args.cuda)

    if args.select_features:
        model = finetune_fs.FineTune_FS(args=args, backbone='resnet50')
    else:
        model = finetune.FineTune(args=args, backbone='resnet50')

    train_data = utils.get_dataset(args=args, mode='train')
    if args.test:
        val_data = utils.get_dataset(args=args, mode='val')
    else:
        val_data = None


    f_importance_1 = model.evaluate(train_loader=train_data,
                            val_loader=val_data)
    
    f_importance_2 = model.get_feature_importance()
    
    utils.save_np(f_importance_1, save_path)    


    # import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    main()