import torch
import torch.nn.functional as F

def build_knn_candidates_and_labels(gt_nodes_xy: torch.Tensor,
                                    gt_edges_idx: torch.Tensor,
                                    k: int = 4):
    """
    Args:
      gt_nodes_xy:  (N,2) pixel coords (x,y) y-down
      gt_edges_idx: (E_gt,2) int64 undirected pairs from JSON
    Returns:
      edge_index:   (2, E) int64 directed (both directions, no self loops)
      edge_label:   (E,)   float (0/1) for each directed edge
    """
    device = gt_nodes_xy.device
    gt_nodes_xy = gt_nodes_xy.to(device=device)
    gt_edges_idx = gt_edges_idx.to(device=device)

    N = gt_nodes_xy.size(0)
    if N <= 1:
        return torch.empty(2,0, dtype=torch.long, device=device), torch.empty(0)

    # pairwise distances
    D = torch.cdist(gt_nodes_xy, gt_nodes_xy)  # (N,N)
    D[torch.arange(N, device=device), torch.arange(N, device=device)] = float('inf')

    # kNN indices (N, k)
    knn_idx = torch.topk(-D, k=min(k, N-1), dim=1).indices  # negative for smallest

    # make directed edges (src i -> dst j)
    src = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    dst = knn_idx.reshape(-1)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst], dim=0)  # (2,E)

    # label positives using undirected GT set
    gt_set = set(tuple(sorted(map(int, e.tolist()))) for e in gt_edges_idx)
    und = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=1)
    lbl = torch.tensor([1.0 if (int(a), int(b)) in gt_set else 0.0 for a,b in und.tolist()],
                       dtype=torch.float32, device=device)

    return edge_index, lbl

def detect_nodes(junc_logits, offsets_cell, HI, WI, threshold=0.3):
    """
    Args:
      junc_logits:  (1,1,Hc,Wc)
      offsets_cell: (1,2,Hc,Wc) offsets in cell units
    Returns:
      nodes_xy
      scores
      cells_ij
      
    """  
    p = junc_logits[0, 0]                   # (Hc, Wc)
    # mask = p > threshold

    p_max = F.max_pool2d(
        p.unsqueeze(0).unsqueeze(0), 
        kernel_size=3, stride=1, padding=1
    )[0, 0]

    mask = p > threshold
    mask = mask & (p >= p_max * 0.95)
    if mask.sum() == 0:
        device = p.device
        return (torch.zeros((0,2), device=device),
                torch.zeros((0,),  device=device),
                torch.zeros((0,2),  dtype=torch.long, device=device))

    Hc, Wc = p.shape
    ys, xs = mask.nonzero(as_tuple=True)     # (N,)
    scores = p[ys, xs]                       # (N,)

    # offsets in cell units at those positions
    u_cell = offsets_cell[0, 0, ys, xs]      # (N,)
    v_cell = offsets_cell[0, 1, ys, xs]      # (N,)

    xs_f = xs.to(torch.float32)
    ys_f = ys.to(torch.float32)

    stride = HI // Hc

    x = (u_cell + xs_f + 0.5) * stride
    y = (v_cell + ys_f + 0.5) * stride

    nodes_xy = torch.stack([x, y], dim=1)    # (N,2)
    cells_ij = torch.stack([ys, xs], dim=1)  # (N,2)

    return nodes_xy, scores, cells_ij

def assign_pred_to_gt(pred_xy, gt_xy, max_dist=20.0):
    """
    Args:
      pred_xy:  (Np, 2)
      gt_xy:    (Ng, 2)
    """
    if pred_xy.numel() == 0:
      return torch.zeros((0,), dtype=torch.long, device=pred_xy.device)

    if gt_xy.numel() == 0:
      return -torch.ones((pred_xy.size(0),), dtype=torch.long, device=pred_xy.device)
    
    Np = pred_xy.size(0)
    Ng = gt_xy.size(0)

    # Compute pairwise distance matrix (Np, Ng)
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    dist = diff.norm(dim=2) # (Np, Ng)

    # For every predicted node choose the nearest GT
    min_dist, min_idx = dist.min(dim=1) # (Np, )

    # If too far -> no match
    min_idx [min_dist > max_dist] = -1

    return min_idx
        
def build_knn_from_pred(pred_xy, k=8):
    """"
    Args:
        pred_xy: (Np, 2) predicted node coords
    Returns:
        edge_index: (2, E) directed edges
    """
    device = pred_xy.device
    N = pred_xy.size(0)

    if N <= 1:
        return torch.empty(2,0, dtype=torch.long, device=device)
    
    # pairwise distances
    D = torch.cdist(pred_xy.unsqueeze(0), pred_xy.unsqueeze(0)).squeeze(0)  # (N,N)
    D[torch.arange(N, device=device), torch.arange(N, device=device)] = float('inf')

    # kNN indices (N, k)
    knn_idx = torch.topk(-D, k=min(k, N-1), dim=1).indices  # negative for smallest

    # make directed edges (src i -> dst j)
    src = torch.arange(N, device=device).unsqueeze(1).expand_as(knn_idx).reshape(-1)
    dst = knn_idx.reshape(-1)
    
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = torch.stack([src, dst], dim=0)  # (2,E)

    return edge_index

def build_edge_labels(edge_index, min_idx, gt_edges_local):
    """
    Args:
        edge_index: (2, E) edges between predicted node indices
        min_idx:   (N_pred,) mapping pred_idx -> gt_idx or -1
        gt_edges_local: (M, 2) ground-truth edges in local GT index space
    Returns:
        labels: (E,) with 0/1
    """
    device = edge_index.device

    if isinstance(gt_edges_local, torch.Tensor):
        gt = gt_edges_local.clone().to(torch.long)
    else:
        gt = torch.tensor(gt_edges_local, dtype=torch.long, device=device)

    a = torch.minimum(gt[:, 0], gt[:, 1])
    b = torch.maximum(gt[:, 0], gt[:, 1])
    gt_pairs = torch.stack([a, b], dim=1)

    gt_set = set((int(u), int(v)) for u, v in gt_pairs)

    E = edge_index.size(1)
    labels = torch.zeros(E, dtype=torch.long, device=device)

    u_pred = edge_index[0]
    v_pred = edge_index[1]

    for e in range(E):
        pu = int(u_pred[e])
        pv = int(v_pred[e])

        gu = int(min_idx[pu])  # GT index of predicted node u
        gv = int(min_idx[pv])  # GT index of predicted node v

        # unmatched → cannot be positive edge
        if gu < 0 or gv < 0:
            continue

        if gu > gv:
            gu, gv = gv, gu

        if (gu, gv) in gt_set:
            labels[e] = 1

    return labels

def build_knn_feature_space(node_feats, k=8):
    """
    Args:
        node_feats:
    Returns:
        edge_index: (2, E)
    """
    N, D = node_feats.shape
    if N < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=node_feats.device)
    
    k = min(k, N - 1)

    dist = torch.cdist(node_feats, node_feats) # (N, N)

    dist.fill_diagonal_(1e9)

    knn_idx= torch.topk(dist, k, dim=1, largest=False).indices # (N, k)

    # build edge list
    src = torch.arange(N, device=node_feats.device).unsqueeze(1).expand_as(knn_idx)
    dst = knn_idx

    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
    return edge_index


def convert_edges_to_local(node_ids_global, edges_global):
    """
    node_ids_global: (Ng,) each GT node's global id
    edges_global: (E, 2) each pair is (global_src, global_dst)
    """
    id2local = {gid.item(): i for i, gid in enumerate(node_ids_global)}

    edges_local = []
    for src, dst in edges_global:
        src = int(src.item())
        dst = int(dst.item())
        if src in id2local and dst in id2local:
            a = id2local[src]
            b = id2local[dst]
            edges_local.append((min(a, b), max(a, b)))

    if edges_local:
        return torch.tensor(edges_local, dtype=torch.long, device=node_ids_global.device)
    else:
        return torch.zeros((0, 2), dtype=torch.long, device=node_ids_global.device)
