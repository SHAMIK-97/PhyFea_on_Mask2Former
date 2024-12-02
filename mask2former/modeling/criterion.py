# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from .PhyFea import PhysicsFormer
from .invalid_list_ADE20k import invalid_list_ADE20k
import random

invalid_pairs_cityscape = [(6, 11), (15, 6), (6, 0), (6, 12), (1, 14), (0, 10), (16, 1), (10, 4), (10, 18), (7, 9), (15, 5), (1, 10), (15, 11), (15, 0), (3, 8), (5, 17), (1, 15), (15, 12), (12, 9), (10, 11), (10, 0), (14, 15), (9, 14), (11, 6), (10, 12), (3, 17)] #, (12, 7), (15, 1), (9, 10), (4, 17), (0, 7), (0, 3), (9, 15), (4, 16), (14, 13), (8, 16), (7, 15), (9, 13), (18, 6), (6, 17), (3, 1), (13, 16), (1, 4), (16, 9), (18, 17), (7, 13), (1, 18), (12, 15), (16, 7), (14, 4), (15, 17), (12, 10), (14, 18), (0, 15), (0, 12), (3, 14), (1, 12), (10, 17), (4, 14), (6, 1), (3, 10), (9, 18), (0, 14), (14, 5), (16, 14), (14, 11), (16, 3), (4, 10), (7, 4), (14, 12), (17, 14), (7, 18), (17, 3), (1, 13), (16, 10), (3, 16), (8, 15), (16, 15), (17, 10), (6, 14), (10, 2), (14, 2), (9, 12), (10, 1), (7, 11), (0, 8), (12, 6), (7, 12), (3, 9), (17, 16), (2, 14), (15, 14), (0, 4), (2, 3), (15, 3), (0, 18), (0, 6), (12, 5), (6, 16), (5, 14), (15, 10), (10, 14), (10, 3), (14, 8), (1, 6), (18, 16), (2, 15), (17, 9), (1, 17), (17, 7), (15, 16), (16, 4), (14, 17), (16, 18), (6, 7), (0, 2), (10, 16), (11, 10), (3, 15), (9, 6), (15, 9), (13, 4), (9, 17), (4, 15), (15, 7), (4, 12), (8, 12), (16, 0), (10, 9), (16, 12), (7, 17), (17, 15), (18, 14), (14, 1), (16, 13), (6, 15), (18, 10), (12, 17), (18, 15), (12, 16), (0, 17), (2, 0), (14, 3), (6, 13), (2, 12), (7, 1), (0, 16), (14, 10), (3, 6), (1, 16), (5, 12), (10, 15), (15, 13), (0, 9), (16, 6), (14, 16), (17, 4), (3, 11), (7, 3), (3, 12), (10, 13), (16, 17), (17, 6), (11, 12), (16, 5), (6, 4), (6, 18), (1, 7), (16, 11), (9, 16), (13, 6), (12, 14), (12, 3), (14, 9), (17, 11), (17, 12), (7, 16), (15, 4), (15, 18), (16, 2)]
invalid_pairs_acdc = [(16, 4), (14, 18), (4, 9), (6, 11), (9, 1), (13, 17), (17, 7), (2, 0), (4, 16), (12, 10), (3, 10), (14, 11), (11, 10), (11, 6), (6, 13), (15, 9), (8, 12), (13, 12), (4, 12), (10, 15), (0, 5), (6, 17), (9, 14), (9, 3), (16, 9), (15, 7), (9, 2), (14, 13), (17, 10), (10, 18), (16, 7), (17, 6), (7, 10), (13, 4), (14, 17), (9, 16), (1, 11), (12, 5), (12, 15), (3, 15), (10, 11), (1, 14), (11, 15), (6, 1), (15, 6), (14, 12), (12, 18), (3, 18), (9, 12), (0, 14), (0, 3), (1, 13), (11, 18), (18, 10), (18, 6), (0, 2), (5, 14), (13, 9), (14, 1), (1, 16), (10, 13), (17, 15), (7, 5), (14, 4), (13, 16), (7, 15), (10, 17), (12, 11), (13, 7), (0, 17), (0, 16), (6, 3), (6, 14), (17, 18), (5, 17), (7, 18), (5, 16), (12, 14), (1, 12), (3, 14), (6, 9), (0, 8), (10, 12), (3, 13), (0, 12), (6, 16), (12, 17), (3, 17), (5, 12), (16, 15), (17, 11), (1, 4), (7, 11), (14, 9), (15, 18), (13, 6), (4, 10), (18, 15), (10, 1), (4, 6), (14, 16), (10, 4), (14, 7), (0, 4), (6, 12), (7, 14), (9, 7), (10, 0), (12, 8), (3, 12), (11, 5), (7, 13), (15, 11), (7, 17), (1, 2), (6, 4), (10, 3), (12, 1), (3, 1), (6, 0), (18, 11), (12, 4), (12, 0), (17, 5), (14, 6), (10, 9), (8, 15), (18, 14), (9, 6), (7, 8), (4, 15), (9, 10), (7, 12), (18, 13), (8, 18), (12, 3), (17, 1), (18, 17), (7, 1), (11, 14), (7, 4), (15, 5), (12, 9), (3, 9), (6, 7), (16, 5), (1, 6), (8, 11), (12, 16), (3, 16), (12, 7), (4, 11), (13, 10), (11, 17), (15, 1), (0, 10), (17, 3), (17, 14), (0, 6), (16, 18), (9, 15), (7, 3), (17, 9), (9, 18), (2, 1), (4, 13), (7, 9), (11, 12), (8, 17), (17, 16), (4, 17), (7, 16), (16, 11), (15, 14), (15, 3), (12, 6), (3, 6), (1, 15), (16, 14), (16, 3), (9, 11), (13, 5), (15, 13), (13, 15), (18, 3), (17, 12), (2, 14), (2, 3), (15, 17), (0, 15), (16, 13), (15, 16), (1, 18), (13, 18), (16, 17), (18, 9), (2, 9), (0, 18), (4, 1), (9, 13), (18, 16), (2, 17), (15, 8), (18, 7), (2, 16), (9, 17), (8, 0), (6, 15), (15, 12), (5, 18), (4, 0), (1, 10), (13, 11), (16, 12), (6, 18), (14, 5), (14, 15), (2, 12), (11, 7), (8, 14), (15, 4), (4, 14), (5, 11), (15, 0)]

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule



def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,dataset_used=None):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.dataset_used = dataset_used
        self.phyfea = None
        self.sublist_size = len(invalid_pairs_cityscape)
        if self.dataset_used =='Cityscapes':
           self.phyfea = PhysicsFormer(invalid_pairs_cityscape)
        elif self.dataset_used == 'ACDC':
           self.phyfea = PhysicsFormer(invalid_pairs_acdc)
        else:
           self.phyfea = PhysicsFormer()

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def phyfea_loss(self, outputs, targets, indices, num_masks):
        """
        calculates phyfea loss
        :param outputs: product of class and mask
        :return: l_opening
        """
                
        pred_logits = outputs["pred_logits"][..., :-1]
        pred_masks =  outputs["pred_masks"]
        
        semseg = torch.einsum("bqc,bqhw->bchw",pred_logits ,pred_masks)

        if self.dataset_used =='ADE20k':
           with torch.no_grad():
                sublist = random.sample(invalid_list_ADE20k, self.sublist_size)
                self.phyfea.invalid_pair_list = sublist

        l1_norm = self.phyfea(semseg)
        
        losses = {'l1_norm': l1_norm}
        del semseg
         
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'l1_norm': self.phyfea_loss
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
