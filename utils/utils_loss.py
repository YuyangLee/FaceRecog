import numpy as np
import torch
import torch.nn.functional as F

def triplet_thres(anchor, positive, negative):
    threshold = (torch.norm(positive - anchor, dim=-1).max() + torch.norm(negative - anchor, dim=-1).min()) / 2
    return threshold

def triplet_acc(anchor, positive, negative, threshold):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    tn = (dist_ap > threshold).float().mean()
    fp = (dist_an < threshold).float().mean()
    return tn, fp

def pair_thres(dst_mat, adj_mat):
    min_neg_dist = (dst_mat * (1 - adj_mat) + 1e9 * adj_mat).min()
    max_pos_dist = (dst_mat * adj_mat).max()
    threshold = (min_neg_dist + max_pos_dist) / 2
    return threshold

def pair_acc(dst_mat, adj_mat, threshold):
    tn = (dst_mat * adj_mat       > threshold).float().sum() /        adj_mat.sum()
    fp = (dst_mat * (1 - adj_mat) < threshold).float().mean() / (1 - adj_mat).sum()
    return tn, fp
    
def triplet_l2(anchor, positive, negative, margin):
    dist_ap = torch.norm(anchor - positive, dim=-1)
    dist_an = torch.norm(anchor - negative, dim=-1)
    loss = torch.relu(dist_ap - dist_an + margin)
    
    threshold = triplet_thres(anchor, positive, negative)
    tn, fp = triplet_acc(anchor, positive, negative, threshold)
    
    return loss, threshold, tn, fp

def triplet_cos(anchor, positive, negative, margin):
    dist_ap = torch.cosine_similarity(anchor, positive, dim=-1)
    dist_an = torch.cosine_similarity(anchor, negative, dim=-1)
    loss = torch.relu(dist_ap - dist_an + margin)
    
    threshold = triplet_thres(anchor, positive, negative)
    tn, fp = triplet_acc(anchor, positive, negative, threshold)
    
    return loss, threshold, tn, fp

def pairwise_l2(fts, labels, margin):
    """
    Pairwise Loss
    
    Args:
        fts: B x L
        labels: B
        margin: scalar
    """
    B, L = fts.shape
    adj_mat = (labels.unsqueeze(2).expand([B, B]) == labels.unsqueeze(1).expand([B, B])).float()
    dst_mat = torch.norm(fts.unsqueeze(2).expand([B, B, L]) - fts.unsqueeze(1).expand([B, B, L]), p=2, dim=-1)
    pos_loss = dst_mat * adj_mat
    neg_loss = torch.relu(margin - dst_mat * (1 - adj_mat))
    loss = (pos_loss + neg_loss).mean() / 2
    
    threshold  = pair_thres(dst_mat, adj_mat)
    tn, fp = pair_acc(dst_mat, adj_mat, pair_thres(dst_mat, adj_mat))
    
    return loss, threshold, tn, fp

def liftstr_l2(fts, labels, margin):
    """
    Smoothed Lifted Structured Loss, see https://arxiv.org/abs/1511.06452
    Args:
        fts: B x L
        labels: B
        margin: scalar
    """
    B, L = fts.shape
    adj_mat = (labels.unsqueeze(1).expand([B, B]) == labels.unsqueeze(0).expand([B, B])).float()
    dst_mat = torch.norm(fts.unsqueeze(1).expand([B, B, L]) - fts.unsqueeze(0).expand([B, B, L]), p=2, dim=-1)
    
    pos_loss = dst_mat * adj_mat
    neg_loss = torch.log(torch.exp(margin - dst_mat * (1 - adj_mat)))
    loss = (pos_loss + neg_loss).mean() / 2
    
    threshold  = pair_thres(dst_mat, adj_mat)
    tn, fp = pair_acc(dst_mat, adj_mat, pair_thres(dst_mat, adj_mat))
    
    return loss, threshold, tn, fp
