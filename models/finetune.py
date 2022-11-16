import torch.nn as nn
import torch

from backbones import resnet
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import utils, ssl_utils
import os, pickle

from sklearn.model_selection import train_test_split



class FineTune(nn.Module):
    def __init__(self, args, backbone):
        super(FineTune, self).__init__()
        self.use_cuda = args.cuda

        self.img_size = args.img_size
        self.nb_classes = args.nb_classes
        self.using_wandb = args.wandb

        self.backbone_name = backbone
        self.backbone_mode = args.backbone_mode
        self.dataset_name = args.dataset

        self.log_path = args.log_path
        utils.make_dirs(self.log_path)

        self.tol_count = 0
        self.es_tolerence = args.es_tol

        self.use_cache = args.data_use_cache
        self.epochs = args.epochs
        self.gl_p = args.loss_gl_p
        self.gl_r = args.loss_gl_r
        self.gl_coeff = args.loss_gl_coeff
        self.emb_normalization = args.emb_normalization
        self.train_to_val_ratio_split = args.train_to_val_ratio_split
        self.target_size = args.target_size
        self.layers_to_use = args.layers_to_use

        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        if backbone == 'resnet50':
            if self.backbone_mode == 'supervised':
                self.backbone = resnet.resnet50(ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = resnet.resnet50()
                self.backbone = ssl_utils.load_ssl_weight_to_model(model=self.backbone,
                                                                method_name=self.backbone_mode,
                                                                arch_name='resnet50',
                                                                ssl_path=args.ssl_backbone_path)
                
        else:
            raise Exception('Backbone not supported')

        if not args.finetune_backbone:
            self.backbone.eval()
        
        self.loss_fn = CrossEntropyLoss()

        self.classification_layer = None
        self.feature_importance = []
        self.output_size = -1 # will be set in _prepare_fc()
        self.total_output_size = -1
        self.classification_layer = None

        self.optimizer_name = args.optimizer
        self.args = args
        self.optimizer = None

    
    def _set_optimizer(self, args):
        if self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.classification_layer.parameters(), momentum=0.9, lr=args.lr)
        elif self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params=self.classification_layer.parameters(), lr=args.lr)
        else:
            raise Exception('Optimizer not set, or not supported')
    
    def _prepare_fc(self, selected_feature_indices=None):
        if selected_feature_indices is None:
            x = torch.rand((1, 3, self.img_size, self.img_size))
            out = self.backbone(x)
            out = self._choose_layers(out)
            out = utils.flatten_and_concat(out, target_size=self.target_size)
            self.output_size = 0
            self.embedding_sizes = []
            for o in out:
                self.output_size += o.shape[-1]
                self.embedding_sizes.append(o.shape[-1])

            self.total_output_size = self.output_size
        else:
            self.output_size = 0
            self.embedding_sizes = []
            for o in selected_feature_indices:
                self.output_size += len(o)
                self.embedding_sizes.append(len(o))

        print('Output feature size is:', self.output_size)

        self.classification_layer = nn.Sequential(OrderedDict([('fc', nn.Linear(self.output_size, self.nb_classes))]))

    
    def __group_lasso_reg(self):
        # self.feature_importance = torch.norm(fc_weights, dim=0, p=2).detach().cpu()
        fc_weights = self.classification_layer.fc.weight
        return torch.norm(torch.norm(fc_weights, dim=0, p=self.gl_r), p=self.gl_r)

    def _choose_layers(self, output):
        if self.layers_to_use == "all" or set(output.keys()) == set(self.layers_to_use):
            return output
        else:
            assert len(set(self.layers_to_use) - set(output.keys())) == 0
            final_output = {}
            self.layers_to_use = sorted(self.layers_to_use)
            final_output = {l_n: output[l_n] for l_n in self.layers_to_use}
            return final_output
                

    def _get_embedding(self, dataloader):
        self.backbone.eval()
        batch_embedding_lists = []
        labels = []
        with tqdm(total=len(dataloader), desc="Getting head2toe embeddings...") as t:
            for idx, batch in enumerate(dataloader):
                x, l = batch
                if self.use_cuda:
                    x = x.cuda()
                out = self.backbone(x)
                out = self._choose_layers(out)
                batch_embedding_lists.append(utils.flatten_and_concat(output_dict=out, 
                                                target_size=self.target_size)) # should I detach?
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
            embeddings = [
                embedding[:, indices] for embedding, indices
                in zip(embeddings, selected_features)
                if np.prod(indices.shape) > 0
            ]
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
    
    def update_feature_importance(self):
        fc_weights = self.classification_layer.fc.weight
        self.feature_importance = torch.norm(fc_weights, dim=0, p=2).detach().cpu().numpy()
        return

    def get_feature_importance(self):
        return self.feature_importance
    
    def train_step(self, epoch, data_loader):
        # if self.finetune_backbone:
        #     # self.backbone.train()
        self.classification_layer.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_gl_reg = 0
        all_trues = []
        all_preds = []
        with tqdm(total=len(data_loader), desc=f"Training classifier {epoch} / {self.epochs}") as t:
            for batch_id, batch in enumerate(data_loader, 1):
                x, l = batch
                if self.use_cuda:
                    x = x.cuda()
                    l = l.cuda()
                logits = self.classification_layer(x)
                loss = self.loss_fn(input=logits, target=l) + self.gl_coeff * self.__group_lasso_reg() 

                ce_loss = self.loss_fn(input=logits, target=l)
                group_lasso_reg = self.__group_lasso_reg()

                loss = ce_loss + self.gl_coeff * group_lasso_reg
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_gl_reg += self.gl_coeff * group_lasso_reg.item()

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_trues.extend(l.cpu().numpy())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                postfixes = {f'train_loss': f'{epoch_loss / (batch_id) :.4f}',
                            f'train_ce_loss': f'{epoch_ce_loss / (batch_id) :.4f}',
                            f'train_gl_reg': f'{epoch_gl_reg / (batch_id) :.4f}'}

                t.set_postfix(**postfixes)
                t.update()
        

        train_acc = accuracy_score(y_pred=all_preds, y_true=all_trues)
        train_loss = epoch_loss / len(data_loader)
        train_ce_loss = epoch_ce_loss / len(data_loader)
        train_gl_reg = epoch_gl_reg / len(data_loader)

        utils.wandb_update_value({'train/acc': train_acc,
                                    'train/loss': train_loss,
                                    'train/ce_loss': train_ce_loss,
                                    'train/group_lasso': train_gl_reg})

        return train_acc, train_loss
                
    def eval_step(self, epoch, data_loader):
        # self.backbone.eval()
        self.classification_layer.eval()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_gl_reg = 0
        all_preds = []
        all_trues = []
        with tqdm(total=len(data_loader), desc=f"Validating epoch {epoch}: ") as t:
            for batch_id, batch in enumerate(data_loader, 1):
                x, l = batch
                if self.use_cuda:
                    x = x.cuda()
                    l = l.cuda()
                logits = self.classification_layer(x)
                ce_loss = self.loss_fn(input=logits, target=l)
                group_lasso_reg = self.__group_lasso_reg()

                loss = ce_loss + self.gl_coeff * group_lasso_reg
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_gl_reg += self.gl_coeff * group_lasso_reg.item()

                preds = logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_trues.extend(l.cpu().numpy())
                
                postfixes = {f'val_loss': f'{epoch_loss / (batch_id) :.4f}',
                             f'val_ce_loss': f'{epoch_ce_loss / (batch_id) :.4f}',
                              f'val_gl_reg': f'{epoch_gl_reg / (batch_id) :.4f}'}

                t.set_postfix(**postfixes)
                t.update()

        eval_loss = epoch_loss / len(data_loader)
        eval_acc = accuracy_score(y_pred=all_preds, y_true=all_trues)
        eval_ce_loss = epoch_ce_loss / len(data_loader)
        eval_gl_reg = epoch_gl_reg / len(data_loader)

        utils.wandb_update_value({'val/acc': eval_acc,
                                    'val/loss': eval_loss,
                                    'val/ce_loss': eval_ce_loss,
                                    'val/group_lasso': eval_gl_reg})

        return eval_acc, eval_loss

    def _train_classifier(self, train_data_loader, val_data_loader):
        self._set_optimizer(args=self.args)
        best_val_acc = 0
        for epoch in range(1, self.epochs + 1):
            train_acc, train_loss = self.train_step(epoch=epoch, data_loader=train_data_loader)
            print('train_acc: ', train_acc)
            val_acc, val_loss = self.eval_step(epoch=epoch, data_loader=val_data_loader)
            if val_acc >= best_val_acc:
                if self.gl_coeff > 0:
                    self.update_feature_importance()
                best_val_acc = val_acc
                self.tol_count  = 0
            else:
                self.tol_count += 1
            print('val_acc: ', val_acc)
            if self.using_wandb:
                utils.wandb_log()
            if self.es_tolerence > 0 and self.tol_count > self.es_tolerence:
                print(f'Early stopping, val_acc did not improve over {best_val_acc} for {self.es_tolerence} epochs!')
                break

        return best_val_acc

    def optimize_finetune(self, train_loader, val_loader, selected_feature_indices=None):
        self._prepare_fc(selected_feature_indices)
        assert self.total_output_size != -1
        emb_path = os.path.join(self.log_path, f'{self.dataset_name}_{self.backbone_name}_{self.backbone_mode}_ts{self.target_size}_imgsize{self.img_size}_outputsize{self.total_output_size}')
        utils.make_dirs(emb_path)
        train_emb_path = os.path.join(emb_path, 'train_emb.pkl')
        train_lbls_path = os.path.join(emb_path, 'train_lbls.npy')
        val_emb_path = os.path.join(emb_path, 'val_emb.pkl')
        val_lbls_path = os.path.join(emb_path, 'val_lbls.npy')

        if self.use_cuda:
            self.cuda()

        if self.use_cache and os.path.exists(train_emb_path):
            print(f'Using cache.... {train_emb_path}')
            print(f'Using cache.... {train_lbls_path}')
            train_embeddings = self._load_dataset(train_emb_path)
            train_labels = self._load_dataset_npy(train_lbls_path)
            train_embeddings = [torch.tensor(t) for t in train_embeddings]
            train_labels = torch.tensor(train_labels)
        else:
            train_embeddings, train_labels = self._get_embedding(train_loader)
            if self.use_cache:
                to_save = [t.numpy() for t in train_embeddings]
                self._save_dataset(to_save, train_emb_path)
                self._save_dataset_npy(train_labels.numpy(), train_lbls_path)

        train_embeddings = self._process_embeddings(embeddings=train_embeddings,
                                                    selected_features=selected_feature_indices,
                                                    normalization=self.emb_normalization) 
        train_emb_dataset = list(zip(train_embeddings.numpy(), train_labels.numpy()))


        
        train_embedding_dl = DataLoader(train_emb_dataset, shuffle=True, batch_size=self.train_batch_size)

        if val_loader is not None: # this is for testing, i.e. test split
            if self.use_cache and os.path.exists(val_emb_path):
                print(f'Using cache.... {val_emb_path}')
                print(f'Using cache.... {val_lbls_path}')
                val_embeddings = self._load_dataset(val_emb_path)
                val_labels = self._load_dataset_npy(val_lbls_path)
                val_embeddings = [torch.tensor(t) for t in val_embeddings]
                val_labels = torch.tensor(val_labels) 
            else:
                val_embeddings, val_labels = self._get_embedding(val_loader)

                if self.use_cache:
                    to_save = [t.numpy() for t in val_embeddings]
                    self._save_dataset(to_save, val_emb_path)
                    self._save_dataset_npy(val_labels.numpy(), val_lbls_path)

            val_embeddings = self._process_embeddings(embeddings=val_embeddings,
                                                        selected_features=selected_feature_indices,
                                                        normalization=self.emb_normalization)   
            val_emb_dataset = list(zip(val_embeddings.numpy(), val_labels.numpy()))
    
            val_embedding_dl = DataLoader(val_emb_dataset, shuffle=True, batch_size=self.val_batch_size)

        else: # i.e. val split
            assert self.train_to_val_ratio_split != 0
            tr_x = np.array(list(list(zip(*train_emb_dataset))[0]))
            tr_y = np.array(list(list(zip(*train_emb_dataset))[1]))
            train_embeddings, val_embeddings, train_labels, val_labels = train_test_split(tr_x, tr_y, train_size=self.train_to_val_ratio_split, stratify=tr_y)
            train_emb_dataset = list(zip(train_embeddings, train_labels))
            val_emb_dataset = list(zip(val_embeddings, val_labels))
            print(f'Splitting train into train-val like {self.train_to_val_ratio_split}-{1 - self.train_to_val_ratio_split}')
            print('Train set new size:', len(train_emb_dataset))
            print('Validation set new size:', len(val_emb_dataset))
            train_embedding_dl = DataLoader(train_emb_dataset, shuffle=True, batch_size=self.train_batch_size)
            val_embedding_dl = DataLoader(val_emb_dataset, shuffle=True, batch_size=self.val_batch_size)

        return self._train_classifier(train_embedding_dl, val_embedding_dl)

    def evaluate(self, train_loader, val_loader):
        final_val_acc = self.optimize_finetune(train_loader=train_loader, 
                                val_loader=val_loader,
                                selected_feature_indices=None)

        print('Final validation acc:', final_val_acc)

    def _load_dataset(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def _save_dataset(self, data, data_path):
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    def _save_dataset_npy(self, data, data_path):
        np.save(data_path, data)
    
    def _load_dataset_npy(self, data_path):
        data = np.load(data_path)
        return data