import numpy as np
import torch
import torch.nn.functional as F

def l2_dist(ft0, ft1):
    return torch.norm(ft0 - ft1, dim=-1)

def cos_dist(ft0, ft1):
    return 1 - torch.cosine_similarity(ft0, ft1, dim=-1)

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
    pos_adj = torch.triu(    adj_mat, diagonal=1)
    neg_adj = torch.triu(1 - adj_mat, diagonal=1)
    tn = ((dst_mat > threshold) * pos_adj).float().sum() / (pos_adj.sum() + 1e-12)
    fp = ((dst_mat < threshold) * neg_adj).float().sum() / (neg_adj.sum() + 1e-12)
    return tn, fp
    
def get_loss(loss='triplet', metric='l2'):
    if metric == 'l2':
        dist_fn = l2_dist
    elif metric == 'cos':
        dist_fn = cos_dist
    else:
        raise NotImplementedError("Metric not implemented.")
    
    if loss == 'triplet':
        def loss_fn(anchor, positive, negative, margin):
            dist_ap = dist_fn(anchor, positive)
            dist_an = dist_fn(anchor, negative)
            loss = torch.relu(dist_ap - dist_an + margin)
            
            threshold = (dist_fn(anchor, positive).max() + dist_fn(anchor, negative).min()) / 2
            tn, fp = triplet_acc(anchor, positive, negative, threshold)
            
            return loss, threshold, tn, fp
    
    elif loss == 'pairwise':
        def loss_fn(fts, labels, margin):
            """
            Pairwise Loss
            
            Args:
                fts: B x L
                labels: B
                margin: scalar
            """
            B, L = fts.shape
            adj_mat = (labels.unsqueeze(2).expand([B, B]) == labels.unsqueeze(1).expand([B, B])).float()
            pos_adj = torch.triu(    adj_mat, diagonal=1)
            neg_adj = torch.triu(1 - adj_mat, diagonal=1)
            
            dst_mat = dist_fn(fts.unsqueeze(2).expand([B, B, L]), fts.unsqueeze(1).expand([B, B, L]))
            
            pos_loss = dst_mat * pos_adj
            neg_loss = torch.relu(margin - dst_mat * neg_adj)
            loss = (pos_loss + neg_loss).mean() # No need to /2 for we have processed the symm adjaceny matrix
            
            threshold  = pair_thres(dst_mat, adj_mat)
            tn = ((dst_mat > threshold) * pos_adj).float().sum() / (pos_adj.sum() + 1e-12)
            fp = ((dst_mat < threshold) * neg_adj).float().sum() / (neg_adj.sum() + 1e-12)
            
            return loss, threshold, tn, fp
    
    elif loss == 'liftstr':
        def loss_fn(fts, labels, margin):
            """
            Smoothed Lifted Structured Loss, see https://arxiv.org/abs/1511.06452
            Args:
                fts: B x L
                labels: B
                margin: scalar
            """
            B, L = fts.shape
            adj_mat = (labels.unsqueeze(1).expand([B, B]) == labels.unsqueeze(0).expand([B, B])).float()
            pos_adj = torch.triu(    adj_mat, diagonal=1)
            neg_adj = torch.triu(1 - adj_mat, diagonal=1)
            
            dst_mat = dist_fn(fts.unsqueeze(1).expand([B, B, L]), fts.unsqueeze(0).expand([B, B, L]))
            
            pos_loss = dst_mat * adj_mat
            neg_loss = torch.log(torch.exp(margin - dst_mat * (1 - adj_mat)))
            loss = torch.relu(pos_loss + neg_loss).mean() / 2
            
            threshold  = pair_thres(dst_mat, adj_mat)
            tn = ((dst_mat > threshold) * pos_adj).float().sum() / (pos_adj.sum() + 1e-12)
            fp = ((dst_mat < threshold) * neg_adj).float().sum() / (neg_adj.sum() + 1e-12)
            
            return loss, threshold, tn, fp
        
    else:
        raise NotImplementedError("Loss not implemented.")
        
    return loss_fn
