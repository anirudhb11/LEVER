import torch
import torch.nn.functional as F
import math
import torch.nn as nn


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean', pad_ind=None):
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss):
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'custom':
            return loss.sum(dim=1).mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss):
        """
        Mask the loss at padding index, i.e., make it zero
        """
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss, mask=None):
        """
        Mask the loss at padding index, i.e., make it zero
        * Mask should be a boolean array with 1 where loss needs
        to be considered.
        * it'll make it zero where value is 0
        """
        if mask is not None:
            loss = loss.masked_fill(~mask, 0.0)
        return loss



class TripletMarginLossOHNM(_Loss):
    r""" Triplet Margin Loss with Online Hard Negative Mining

    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', margin=0.8, k=3, apply_softmax=False, tau=0.1, num_violators=False, num_random_negs=-1):
        super(TripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax
        self.num_random_negs = num_random_negs

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.min(input, 1-target)
        k = self.k
        k = min(k, similarities.shape[1])
        if self.num_random_negs > 0:
            k = self.num_random_negs
        _, indices = torch.topk(similarities, largest=True, dim=1, k=k)
        if self.num_random_negs > 0:
            perm = torch.randperm(k)
            indices = indices[:, perm[:self.k]]
            assert indices.shape[1] == self.k
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = loss != 0 #torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        if self.apply_softmax:
            prob = torch.softmax(sim_n/self.tau * mask, dim=1)
            loss = loss * prob
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss

class LogisticTripletMarginLossOHNM(_Loss):
    r""" Logistic Triplet Margin Loss with Online Hard Negative Mining

    * Applies loss using the hardest negative in the mini-batch
    * Assumes diagonal entries are ground truth (for multi-class as of now)

    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    margin: float, optional (default=0.8)
        margin in triplet margin loss
    k: int, optional (default=2)
        compute loss only for top-k negatives in each row 
    apply_softmax: boolean, optional (default=2)
        promotes hard negatives using softmax
    """

    def __init__(self, reduction='mean', margin=0.8, k=3, apply_softmax=False, tau=0.1, num_violators=False, num_random_negs=-1, pairwise_multiplier=5):
        super(LogisticTripletMarginLossOHNM, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax
        self.num_random_negs = num_random_negs
        self.pairwise_multiplier = pairwise_multiplier

    def forward(self, input, target, mask=None):
        """
        Arguments:
        ---------
        input: torch.FloatTensor
            real number pred matrix of size: batch_size x output_size
            cosine similarity b/w label and document
        target:  torch.FloatTensor
            0/1 ground truth matrix of size: batch_size x output_size
        mask: torch.BoolTensor or None, optional (default=None)
            ignore entries [won't contribute to loss] where mask value is False

        Returns:
        -------
        loss: torch.FloatTensor
            dimension is defined based on reduction
        """
        sim_p = torch.diagonal(input).view(-1, 1)
        similarities = torch.min(input, 1-target)
        k = self.k
        k = min(k, similarities.shape[1])
        if self.num_random_negs > 0:
            k = self.num_random_negs
        _, indices = torch.topk(similarities, largest=True, dim=1, k=k)
        if self.num_random_negs > 0:
            perm = torch.randperm(k)
            indices = indices[:, perm[:self.k]]
            assert indices.shape[1] == self.k
        sim_n = input.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = loss != 0 #torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        if self.apply_softmax:
            prob = torch.softmax(sim_n/self.tau * mask, dim=1)
            loss = loss * prob
        loss = torch.log(1 + torch.exp(self.pairwise_multiplier * loss))
        reduced_loss = self._reduce(loss)
        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss

def prepare_loss(args):
    """
    Set-up the loss function
    * num_violators can be printed, if required
    * apply_softmax is more agressive (focus more on top violators)
    """
    if args.loss_type == 'ohnm':
        criterion = TripletMarginLossOHNM(
            margin=args.margin,
            k=args.num_negatives,
            num_violators=args.num_violators,
            apply_softmax=args.agressive_loss)
    elif args.loss_type == 'logistic-ohnm':
        criterion = LogisticTripletMarginLossOHNM(
            margin=args.margin,
            k=args.num_negatives,
            num_violators=args.num_violators,
            apply_softmax=args.agressive_loss)
    else:
        raise NotImplementedError("")
    return criterion

if __name__ == '__main__':
    loss = LogisticTripletMarginLossOHNM(reduction='mean', margin=0.3, k=5)
    print(loss(torch.randn(20, 20), torch.ones(20, 20)))