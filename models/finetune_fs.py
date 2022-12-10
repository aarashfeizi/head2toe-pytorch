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
from models import finetune

from sklearn.model_selection import train_test_split



class FineTune_FS(finetune.FineTune):

    def __init__(self, args, backbone, nb_classes=None):
        super(FineTune_FS, self).__init__(args, backbone, nb_classes)
        self.keep_fraction = args.fraction
        self.keep_fraction_offset = args.fraction_offset
        self.mean_interpolation_coef = args.mean_interpolation_coef
        self.average_over_k = args.average_over_k

    def _select_fraction(self, scores,
                       mean_interpolation_coef=0):
        """Given a scoring function returns the indices of high scores features."""
        n_kept = self.keep_fraction_offset + int(scores.shape[0] * self.keep_fraction)
        if mean_interpolation_coef > 0:
        # We need to interpolate the scores towards it's mean.
            scores, _ = self._interpolate_scores_towards_mean(
                scores, mean_interpolation_coef)
        temp, sorted_indices = torch.topk(input=torch.tensor(scores), k=n_kept)
        selected_indices = sorted_indices[self.keep_fraction_offset:]
        return selected_indices


    def _interpolate_scores_towards_mean(self, scores, coef):
        new_scores = []
        mean_scores = []
        for c_scores in torch.split(scores, split_size_or_sections=self.embedding_sizes):
            c_score_mean = torch.mean(c_scores)
            mean_scores.append(c_score_mean)
            c_scores = c_scores * (1 - coef) + c_score_mean * coef
            new_scores.append(c_scores)
        return torch.concat(new_scores, dim=0), mean_scores

    def _calculate_scores(self, train_loader, val_loader):
        # Pre-generate the embeddings
        _ = self.optimize_finetune(train_loader, val_loader, None,
                                split_names={'train': f'train_{self.fold_idx}', 'val': f'val_{self.fold_idx}'})
        
        all_scores = self.get_feature_importance()
        return all_scores

    def _broadcast_indices(self, kept_indices_all):
        """Splits and removes the offset for indices."""
        start_index = 0
        selected_feature_indices = []
        for embedding_size in self.embedding_sizes:
            end_index = start_index + embedding_size
            kept_indices = torch.masked_select(
                kept_indices_all,
                torch.logical_and(
                    kept_indices_all >= start_index,
                    kept_indices_all < end_index))
            # Remove the offset.
            kept_indices -= start_index

            start_index = end_index
            selected_feature_indices.append(kept_indices)
        return selected_feature_indices
    
    def _select_features(self, train_loader, val_loader=None):
        # if config_fs.type == 'none':
        #     return None, None
        if self.average_over_k > 1:
            all_scores = []
            for _ in range(self.average_over_k):
                all_scores.append(self._calculate_scores(train_loader, val_loader))
            all_scores = torch.mean(torch.stack(all_scores), 0)
        else:
            all_scores = self._calculate_scores(train_loader, val_loader)
        all_scores = torch.tensor(all_scores)
        kept_indices_all = self._select_fraction(
            all_scores, mean_interpolation_coef=self.mean_interpolation_coef)
        _, mean_scores = self._interpolate_scores_towards_mean(all_scores, 1.)
        selected_feature_indices = self._broadcast_indices(kept_indices_all)
        return selected_feature_indices, mean_scores


    def evaluate(self, train_loader, val_loader, test_loader, trainval_loader):

        selected_features, mean_scores = self._select_features(train_loader=train_loader, val_loader=val_loader) # choose with cross validating

        self.gl_coeff = 0 # for final finetuning, no regularizer

        feature_importance = self.get_feature_importance()

        if test_loader is not None:
            final_val_acc = self.optimize_finetune(train_loader=trainval_loader, 
                                    val_loader=test_loader,
                                    selected_feature_indices=selected_features,
                                    split_names={'train': 'trainval', 'val': 'test'})
        else:
            final_val_acc = self.optimize_finetune(train_loader=train_loader, 
                        val_loader=val_loader,
                        selected_feature_indices=selected_features,
                        split_names={'train': f'train_{self.fold_idx}', 'val': f'val_{self.fold_idx}'})

        return feature_importance, final_val_acc