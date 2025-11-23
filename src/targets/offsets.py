import torch

@torch.no_grad()
def build_offset_targets(batch, stride: int = 32):
    """
    Inputs:
        batch["image]: (B, 3, H, W)
        batch["nodes"]: list of length B, each (N_i, 2) in pixel coords (x,y), y-down.
    Returns:
        U: (B, 2, Hc, Wc): Offsets (u, v) in cell units [-0.5, 0.5]
        M: (B, 1, Hc, Wc): masks (1 where a junction exists, else 0)
    """

    imgs = batch["image"]
    B, _, H, W = imgs.shape
    h, w = H // stride, W // stride
    device = imgs.device

    U = torch.zeros((B, 2, h, w), dtype=torch.float32, device=device)
    M = torch.zeros((B, 1, h, w), dtype=torch.float32, device=device)

    for b in range(B):
        nodes = batch["nodes"][b] # (N, 2)

        # rescale node coords (512 -> 16)
        xc = nodes[:, 0] / W * w  # (N, )
        yc = nodes[:, 1] / H * h  # (N, )

        # cell indices
        Xc = torch.clamp(torch.floor(xc), 0, w - 1).to(torch.int64)
        Yc = torch.clamp(torch.floor(yc), 0, h - 1).to(torch.int64)

        # fractional offsets from cell center 
        u = xc - (Xc.float() + 0.5) 
        v = yc - (Yc.float() + 0.5)

        # Each coarse cell covers 32 x 32 pixels of the original image.
        # Someties two or more GT nodes (junctions) can fall into the same coarse cell.
        # But in the coarse GT tensor, each cell can only hold one value for the offset (u, v)
        # So,, we choose one representative junction for that cell (the first node encountered)
        acc = {} # (yi, xi) -> [sum_u, sum_v, count]
        for xi, yi, ui, vi in zip(Xc.tolist(), Yc.tolist(), u.tolist(), v.tolist()):
            key = (yi, xi)
            if key not in acc:
                acc[key] = [0.0, 0.0, 0]

            acc[key][0] += ui
            acc[key][1] += vi
            acc[key][2] += 1

        for (yi, xi), (sum_u, sum_v, count) in acc.items():
            avg_u = sum_u / count
            avg_v = sum_v / count

            M[b, 0, yi, xi] = 1.0
            U[b, 0, yi, xi] = avg_u
            U[b, 1, yi, xi] = avg_v
        
    return U, M