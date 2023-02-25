import numpy as np
import torch
import torch.nn.functional as F

def l2_dist(ft0, ft1):
    return torch.norm(ft0 - ft1, dim=-1)

def cos_dist(ft0, ft1):
    return 1 - torch.cosine_similarity(ft0, ft1, dim=-1, eps=1e-12)

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
    
def get_loss(loss='triplet', metric='l2'):
    if metric == 'l2':
        dist_fn = l2_dist
    elif metric == 'cos':
        dist_fn = cos_dist
    else:
        raise NotImplementedError("Metric not implemented.")
    
    if loss == 'triplet_weak':
        def loss_fn(anchor, positive, negative, margin):
            dist_ap = dist_fn(anchor, positive)
            dist_an = dist_fn(anchor, negative)
            loss = torch.relu(dist_ap - dist_an + margin)
            
            return loss
    
    elif loss == 'triplet':
        def loss_fn(anchor, positive, labels_anc, labels_pos, margin):
            B, L = anchor.shape
            pos_dist = dist_fn(anchor, positive)
            
            _lables = torch.cat([labels_anc, labels_pos], dim=0)
            _fts = torch.cat([anchor, positive], dim=0)
            neg_adj_mat = (labels_anc.unsqueeze(1).expand([B, 2*B]) != _lables.unsqueeze(0).expand([B, 2*B])).float()
            neg_dist = dist_fn(anchor.unsqueeze(1).expand([B, 2*B, L]), _fts.unsqueeze(0).expand([B, 2*B, L])) * neg_adj_mat + 1e12 * (1 - neg_adj_mat)
            # neg_dist_min = neg_dist.min(dim=1)[0]
            neg_dist_min = (F.softmin(neg_dist, dim=1) * neg_dist).sum(dim=1)
            loss = torch.relu((pos_dist - neg_dist_min) + margin) # + 0.1 * pos_dist
            
            return loss * 20
    
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
            adj_mat = (labels.unsqueeze(1).expand([B, B]) == labels.unsqueeze(0).expand([B, B])).float()
            pos_adj = torch.triu(    adj_mat, diagonal=1)
            neg_adj = torch.triu(1 - adj_mat, diagonal=1)
            
            dst_mat = dist_fn(fts.unsqueeze(2).expand([B, B, L]), fts.unsqueeze(1).expand([B, B, L]))
            
            pos_loss = dst_mat * pos_adj
            neg_loss = torch.relu(margin - dst_mat * neg_adj)
            loss = (pos_loss + neg_loss).mean() # No need to /2 for we have processed the symm adjaceny matrix
            
            return loss
    
    elif loss == 'liftstr':
        def loss_fn(anc, pos, labels_0, labels_1, margin):
            """
            Smoothed Lifted Structured Loss, see https://arxiv.org/abs/1511.06452
            """
            B, L = anc.shape
            cp_fts, cp_labels = torch.cat([anc, pos], dim=0), torch.cat([labels_0, labels_1], dim=0)
            an_adj = (labels_0.unsqueeze(1).expand([B, 2*B]) != cp_labels.unsqueeze(0).expand([B, 2*B])).float()
            pn_adj = (labels_1.unsqueeze(1).expand([B, 2*B]) != cp_labels.unsqueeze(0).expand([B, 2*B])).float()
            
            an_dist = dist_fn(anc.unsqueeze(1).expand([B, 2*B, L]), cp_fts.unsqueeze(0).expand([B, 2*B, L])) * an_adj
            pn_dist = dist_fn(pos.unsqueeze(1).expand([B, 2*B, L]), cp_fts.unsqueeze(0).expand([B, 2*B, L])) * pn_adj
            ap_dist = dist_fn(anc, pos)
            
            neg_dist_loss = (torch.exp(-an_dist + margin) + torch.exp(-pn_dist + margin)).sum(dim=-1)
            loss = torch.relu(torch.log(neg_dist_loss) + ap_dist).mean()
            
            return loss**2 * 0.1
        
    else:
        raise NotImplementedError("Loss not implemented.")
        
    return loss_fn
