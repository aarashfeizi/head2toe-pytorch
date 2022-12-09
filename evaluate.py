import torch
import utils
import datasets
from models import finetune, finetune_fs
import wandb
import os
import numpy as np


def main():
    args = utils.get_args()
    save_path = args.log_path
    if args.wandb:
        args = utils.init_wandb(args)
        r = wandb.run
        time = str(r.start_time)
        time = time[:time.find('.')]
        save_path = os.path.join(save_path, f'{r.sweep_id}', f'{r.name}')
        utils.make_dirs(save_path)
    

    
    print(args)
    utils.seed_all(args.seed, args.cuda)

    if args.vtab_5fold and not args.test:
        start_fold = 0
    else:
        start_fold = 4
    
    val_accs = []
    for fold_idx in range(start_fold, 5):
        print('Fold: ', fold_idx)
        if args.vtab:
            train_data = utils.get_dataset_tf(args, mode='train', eval_mode='valid', fold_idx=fold_idx)
            val_data = utils.get_dataset_tf(args, mode='eval', eval_mode='valid', fold_idx=fold_idx)
            if args.test:
                trainval_data = utils.get_dataset_tf(args, mode='train', eval_mode='test')
                test_data = utils.get_dataset_tf(args, mode='eval', eval_mode='test')
                nb_classes = utils.get_nb_classes(trainval_data)
            else:
                trainval_data = None
                test_data = None
                nb_classes = utils.get_nb_classes(train_data)
            
            print(f'train size: {len(train_data.dataset)}')
            print(f'val size: {len(val_data.dataset)}')
            
            if args.test:
                print(f'trainval size: {len(trainval_data.dataset)}')
                print(f'test size: {len(test_data.dataset)}')
            else:
                print('No testing being done!')

            print(f'class number: {nb_classes}')
        else:
            nb_classes = None
            train_data = utils.get_dataset(args=args, mode='train')
            trainval_data = train_data
            val_data = None
            if args.test:
                if args.dataset != 'rhotelid':
                    test_data = utils.get_dataset(args=args, mode='test')
                else:
                    test_data = utils.get_dataset(args=args, mode='test', extra_args={'classes': train_data.dataset.classes,
                                                                        'class_to_idx': train_data.dataset.class_to_idx})
            else:
                print('No testing being done!')
                test_data = None
                

        
        if args.select_features:
            model = finetune_fs.FineTune_FS(args=args, backbone='resnet50', nb_classes=nb_classes)
        else:
            model = finetune.FineTune(args=args, backbone='resnet50', nb_classes=nb_classes)

        

        f_importance_1, final_val_acc = model.evaluate(train_loader=train_data,
                                val_loader=val_data,
                                test_loader=test_data,
                                trainval_loader=trainval_data)
        
        val_accs.append(final_val_acc)
        # f_importance_2 = model.get_feature_importance()
        
        utils.save_np(f_importance_1, save_path, f'f_importance_{fold_idx}')    

    print('Final val accs: ', val_accs)
    print('Final validation acc (avg if 5fold):', np.mean(val_accs))
    utils.wandb_update_value({'val/acc': np.mean(val_accs)})
    utils.wandb_log()

    # import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    main()