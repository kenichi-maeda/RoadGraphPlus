import torch

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