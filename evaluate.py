from backbones import resnet
import torch
import utils
from torchvision.models import ResNet50_Weights
import datasets


def main():
    args = utils.get_args()
    model = resnet.resnet50(ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    if args.env.cuda:
        model.cuda()

    data = utils.get_dataset(args=args, mode='train')
    embeddings, labels = utils.get_embedding(model=model, dataloader=data, cuda=args.env.cuda)


    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    main()