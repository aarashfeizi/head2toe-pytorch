import torch
import utils
import datasets
from models import finetune


def main():
    args = utils.get_args()
    model = finetune.FineTune(args=args, backbone='resnet50')
    
    if args.env.cuda:
        model.cuda()

    train_data = utils.get_dataset(args=args, mode='train')
    val_data = utils.get_dataset(args=args, mode='val')


    model.optimize_finetune(train_loader=train_data,
                            val_loader=val_data)


    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    main()