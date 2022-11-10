import torch.nn as nn
import torch

from backbones import resnet
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import utils



class FineTune(nn.Module):
    def __init__(self, args, backbone):
        super(FineTune, self).__init__()
        self.cuda = args.env.cuda

        self.img_size = args.data.img_size
        self.nb_classes = args.data.nb_classes

        self.epochs = args.env.epochs
        self.gl_p = args.model.loss.gl_p
        self.gl_r = args.model.loss.gl_r
        self.gl_coeff = args.model.loss.gl_coeff
        self.emb_normalization = args.model.emb_normalization
        self.optimizer_name = args.model.optimizer
        self.optimizer = self._get_optimizer()

        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise Exception('Backbone not supported')

        if not args.finetune_backbone:
            self.backbone.eval()
        
        self.loss_fn = BCEWithLogitsLoss()

        self.classification_layer = None
        self.output_size = -1 # will be set in _prepare_fc()
        self.classification_layer = self._prepare_fc()
    
    def _get_optimizer(self, args):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(params=self.classification_layer.parameters, momentum=0.9, lr=args.model.lr)
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(params=self.classification_layer.parameters, lr=args.model.lr)
    
    def _prepare_fc(self):
        x = torch.rand((1, 3, self.img_size, self.img_size))
        if self.cuda:
            x = x.cuda()

        out = self.backbone(x)
        self.output_size = 0
        for k, v in out.items():
            self.output_size += v.shape[-1]

        print('Output feature size is:', self.output_size)

        return nn.Sequential(OrderedDict([('fc', nn.Linear(self.output_size, self.nb_classes))]))

    
    def __group_lasso_reg(self):
        # self.feature_importance = torch.norm(fc_weights, dim=0, p=2).detach().cpu()
        fc_weights = self.classification_layer.fc.weight
        return torch.norm(torch.norm(fc_weights, dim=0, p=self.gl_r), p=self.gl_r)

    def _get_embedding(self, dataloader):
        self.backbone.eval()
        batch_embedding_lists = []
        labels = []
        with tqdm(total=len(dataloader), desc="Getting head2toe embeddings...") as t:
            for idx, batch in enumerate(dataloader):
                x, l = batch
                if self.cuda:
                    x = x.cuda()
                out = self.backbone(x)
                batch_embedding_lists.append(utils.flatten_and_concat(output_dict=out, 
                                                target_size=8192)) # should I detach?
                labels.append(l)

                t.update()
            
        labels = torch.concat(labels)
        output_embeddings = []
        for i in range(len(batch_embedding_lists[0])):
            embedding_i = [batch[i] for batch in batch_embedding_lists]
            output_embeddings.append(torch.concat(embedding_i, dim=0))

        return output_embeddings, labels

    def _process_embeddings(self, embeddings, selected_features,
                            normalization='unit_vector'):
        """Processes embeddings by normalizing an concatenating.

        Args:
        embeddings: list of Tensors, where each Tensor is the embeddings
            of a particular backbone.
        selected_features: list of Tensors, where each Tensor indicates the
            indices to be selected.
        normalization: str, 'unit_vector', 'per_feature_std'.
            'unit_vector' SUR style normalization
            'per_feature' similar to Batch-Normalization

        Returns:
        flattened and possibly scaled embeddings.
        """
        # shape= (n_image, n_features)
        assert normalization in ('unit_vector', 'per_feature', '')
        if selected_features:
            pass
        # Following removes the backbones altogether if no feature is selected.
        # embeddings = [
        #     tf.gather(embedding, indices, axis=1) for embedding, indices
        #     in zip(embeddings, selected_features)
        #     if np.prod(indices.shape) > 0
        # ]
        if normalization == 'unit_vector':
            embeddings = [self._zero_aware_normalize(e, axis=1) for e in embeddings]
        
        embeddings = torch.concat(embeddings, -1)
        if normalization == 'per_feature':
        # Normalize each feature to have unit variance and zero mean.
            mean = torch.mean(embeddings, dim=0)
            var = torch.var(embeddings, dim=0)

            bn_args = {'eps': 1e-5}

            embeddings = torch.nn.functional.batch_norm(embeddings, running_mean=mean, running_var=var,  **bn_args)
        return embeddings

    def _zero_aware_normalize(self, embedding, axis=1):
        """If the norm is zero leaves the row unnormalized."""
        norms = torch.linalg.norm(embedding, axis=axis).reshape(-1, 1)
        norms_equal_to_zero = (norms == 0)
        # Following will have nans when the norm of vector(the divider) is zero.
        normalized = embedding / norms
        return torch.where(norms_equal_to_zero, torch.zeros_like(embedding), normalized)

    def get_feature_importance(self):
        fc_weights = self.classification_layer.fc.weight
        feature_importance = torch.norm(fc_weights, dim=0, p=2).detach().cpu()
        return feature_importance
    
    def train_step(self, epoch, data_loader):
        # if self.finetune_backbone:
        #     # self.backbone.train()
        self.classification_layer.train()
        epoch_loss = 0
        all_trues = []
        all_preds = []
        with tqdm(total=len(data_loader), desc=f"Training classifier {epoch} / {self.epochs}") as t:
            for batch_id, batch in enumerate(data_loader, 1):
                x, l = batch
                if self.cuda:
                    x = x.cuda()
                logits = self.classification_layer(x)
                loss = self.loss_fn(input=logits, target=l) + self.gl_coeff * self.__group_lasso_reg() 

                epoch_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_trues.extend(l)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                postfixes = {f'train_loss': f'{epoch_loss / (batch_id) :.4f}'}

                t.set_postfix(**postfixes)
                t.update()
                
        train_acc = accuracy_score(y_pred=all_preds, y_true=all_trues)
        train_loss = epoch_loss / len(data_loader)

        return train_acc, train_loss
                
    def eval_step(self, epoch, data_loader):
        # self.backbone.eval()
        self.classification_layer.eval()
        epoch_loss = 0
        all_preds = []
        all_trues = []
        with tqdm(total=len(data_loader), desc=f"Validating epoch {epoch}: ") as t:
            for batch_id, batch in enumerate(data_loader, 1):
                x, l = batch
                if self.cuda:
                    x = x.cuda()
                logits = self.classification_layer(x)
                loss = self.loss_fn(input=logits, target=l) + self.gl_coeff * self.__group_lasso_reg() 

                epoch_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_trues.extend(l)

                
                
                postfixes = {f'val_loss': f'{epoch_loss / (batch_id) :.4f}'}

                t.set_postfix(**postfixes)
                t.update()

        eval_loss = epoch_loss / len(data_loader)
        eval_acc = accuracy_score(y_pred=all_preds, y_true=all_trues)

        return eval_acc, eval_loss

    def train_classifier(self, train_data_loader, val_data_loader):
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            train_acc, train_loss = self.train_step(epoch=0, data_loader=train_data_loader)
            val_acc, val_loss = self.eval_step(epoch=0, data_loader=val_datal_oader)
            print('val_acc: ', val_acc)


            
            

    def _optimize_finetune(self, train_loader, val_loader, selected_feature_indices=None):
        train_embeddings, train_labels = self._get_embedding(train_loader)
        train_embeddings = self._process_embeddings(embeddings=train_embeddings,
                                                    selected_features=selected_feature_indices,
                                                    normalization=self.emb_normalization)   
        train_emb_dataset = list(zip(train_embeddings, train_labels))
        train_embedding_dl = DataLoader(train_emb_dataset, shuffle=True, batch_size=self.train_batch_size)
        if val_loader is not None:
            val_embedding_dl = DataLoader(train_embeddings, shuffle=True, batch_size=self.train_batch_size)

        self.train_classifier(train_embedding_dl)


        
        
