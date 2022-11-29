# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint

import numpy as np
import torch
from classy_vision.generic.distributed_util import get_cuda_device_index, get_rank
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.utils.distributed_utils import gather_from_all


@register_loss("false_loss")
class FALSELoss(ClassyLoss):
    """
    Config params:
        temperature (float): the temperature to be applied on the logits
        confidence_weight (float) : the confidence weight to judge false negative sample
        positive_threshold (float) : positive sample pair similarity threshold T (0<T<1 (cos. similarity),close to 1)
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(FALSELoss, self).__init__()

        self.loss_config = loss_config
        # loss constants
        self.temperature = self.loss_config.temperature
        self.buffer_params = self.loss_config.buffer_params
        self.confidence_weight=self.loss_config.confidence_weight
        self.positive_threshold=self.loss_config.positive_threshold
        self.info_criterion = FALSECriterion(
            self.buffer_params, self.temperature, self.confidence_weight, self.positive_threshold
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates FALSELoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            FALSELoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        normalized_output = nn.functional.normalize(output, dim=1, p=2)
        loss = self.info_criterion(normalized_output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "info_average": self.info_criterion}
        return pprint.pformat(repr_dict, indent=2)


class FALSECriterion(nn.Module):
    """
    Args:
        temperature (float): the temperature to be applied on the logits
        confidence_weight (float) : the confidence weight to judge false negative sample. 0<alpha<1
        positive_threshold (float) : positive sample pair similarity threshold T (0<T<1 (cos. similarity),close to 1)
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, buffer_params, temperature: float, confidence_weight: float, positive_threshold: float):
        super(FALSECriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1
        self.temperature = temperature
        self.confidence_weight=confidence_weight
        self.positive_threshold=positive_threshold 
        self.num_pos = 2 
        self.buffer_params = buffer_params
        self.criterion = nn.CrossEntropyLoss()
        self.dist_rank = get_rank()
        self.pos_mask = None
        self.neg_mask = None
        self.precompute_pos_neg_mask()
        logging.info(f"Creating Info-NCE loss on Rank: {self.dist_rank}")

    def precompute_pos_neg_mask(self):
        """
        We precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the begining of training
        total_images = self.buffer_params.effective_batch_size
        world_size = self.buffer_params.world_size
        batch_size = total_images // world_size
        orig_images = batch_size // self.num_pos
        rank = self.dist_rank

        pos_mask = torch.zeros(batch_size, total_images)
        neg_mask = torch.zeros(batch_size, total_images)

        all_indices = np.arange(total_images)
        pos_members = orig_images * np.arange(self.num_pos)
        orig_members = torch.arange(orig_images)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(orig_images):
                delete_inds = batch_size * rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * orig_images + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = batch_size * rank + pos * orig_images + orig_members
                pos_mask[
                    torch.arange(
                        anchor * orig_images, (anchor + 1) * orig_images
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask.cuda(non_blocking=True) if self.use_gpu else pos_mask
        self.neg_mask = neg_mask.cuda(non_blocking=True) if self.use_gpu else neg_mask

    def forward(self, embedding: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.buffer_params.embedding_dim)

        batch_size = embedding.shape[0]
        T = self.temperature
        num_pos = self.num_pos
        assert batch_size % num_pos == 0, "Batch size should be divisible by num_pos"

        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        embeddings_buffer = self.gather_embeddings(embedding)

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096 and
        # divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / T)

        all_pos=similarity * self.pos_mask
        all_neg=similarity * self.neg_mask

        pos_sim=(torch.zeros((batch_size,1))-2).cuda() 
        diff_sim=(torch.zeros((batch_size,batch_size))+3).cuda()
        for pos_idx in range(batch_size):
            if pos_idx>=int(batch_size/2):
                pos_sim[pos_idx]=similarity[pos_idx,pos_idx-int(batch_size/2)]
            else:
                pos_sim[pos_idx]=similarity[pos_idx,pos_idx+int(batch_size/2)] 
            diff_sim[pos_idx,pos_idx:]=torch.abs(similarity[pos_idx,pos_idx:]-pos_sim[pos_idx])
            diff_sim[pos_idx,pos_idx]=3 
            if pos_idx<int(batch_size/2):
                diff_sim[pos_idx,pos_idx+int(batch_size/2)]=3 
        select_pair=torch.where(diff_sim==torch.min(diff_sim)) 
        for pair in range(len(select_pair[0])):
            anchor_n=select_pair[0][pair]
            m_neg_n=select_pair[1][pair]
            cos_pos_sim=torch.log(pos_sim[anchor_n])*T
            if cos_pos_sim>=self.positive_threshold:
                if all_pos[anchor_n,m_neg_n]==0:
                    all_pos[anchor_n,m_neg_n]=self.confidence_weight*all_neg[anchor_n,m_neg_n] #  S_EP(enhanced positive signal)
                if all_neg[anchor_n,m_neg_n]!=0:
                    all_neg[anchor_n,m_neg_n]=(1-self.confidence_weight)*all_neg[anchor_n,m_neg_n] # S_WN(weaken negative signal)
    
        pos=torch.sum(all_pos,1)
        neg=torch.sum(all_neg,1)

        loss = -torch.mean(torch.log((pos / (pos + neg))))

        return loss

    def __repr__(self):
        num_negatives = self.buffer_params.effective_batch_size - 2
        T = self.temperature
        num_pos = self.num_pos
        repr_dict = {
            "name": self._get_name(),
            "temperature": T,
            "num_negatives": num_negatives,
            "num_pos": num_pos,
            "dist_rank": self.dist_rank,
        }
        return pprint.pformat(repr_dict, indent=2)

    @staticmethod
    def gather_embeddings(embedding: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            embedding_gathered = gather_from_all(embedding)
        else:
            embedding_gathered = embedding
        return embedding_gathered
