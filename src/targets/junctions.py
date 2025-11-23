import torch
import math

@torch.no_grad()
def build_junction_heatmaps(batch, stride: int = 32):
    """
    Inputs:
        batch["image"]: (B, 3, H, W)
        batch["nodes"]: list of length B, each (N_i, 2) in pixel coords (x,y), y-down.
    Returns:
        heat: (B, 1, H//strides, W//stride) in [0, 1]
    """
    # imgs = batch["image"]
    # B, _, H, W = imgs.shape
    # h, w = H // stride, W // stride
    # device = imgs.device

    # J = torch.zeros((B, 1, h, w), dtype=torch.float32, device=device)

    # def image_to_cell(x, y, HI, WI, Hc, Wc):
    #     Xj = int((x / WI) * Wc)
    #     Yj = int((y / HI) * Hc)
    #     Xj = min(max(Xj, 0), Wc - 1)
    #     Yj = min(max(Yj, 0), Hc - 1)
    #     return Xj, Yj

    # for b in range(B):
    #     nodes = batch["nodes"][b] # (N, 2)
    #     for x, y in nodes:
    #         Xj, Yj = image_to_cell(x, y, H, W, h, w)
    #         J[b, 0, Yj, Xj] = 1.0

    # return J
    
    imgs = batch["image"]
    B, _, H, W = imgs.shape
    h, w = H // stride, W // stride
    device = imgs.device

    J = torch.zeros((B, 1, h, w), dtype=torch.float32, device=device)

    def image_to_cell(x, y, HI, WI, Hc, Wc):
        nx = x / (WI - 1)
        ny = y / (HI - 1)
        Xj = int(round(nx.item() * (Wc - 1)))
        Yj = int(round(ny.item() * (Hc - 1)))
        return Xj, Yj

    for b in range(B):
        nodes = batch["nodes"][b] # (N, 2)
        for x, y in nodes:
            cx = x / stride
            cy = y / stride

            ix = int(cx)
            iy = int(cy)

            if not (0 <= ix < w and 0 <=iy < h):
                continue

            rad = 1
            sigma = 1.0
            for yy in range(max(0, iy - rad), min(h, iy + rad + 1)):
                for xx in range(max(0, ix - rad), min(w, ix + rad + 1)):
                    dx = cx - (xx + 0.5)
                    dy = cy - (yy + 0.5)
                    dist = (dx)**2 + (dy)**2
                    J[b, 0, yy, xx] = max(J[b, 0, yy, xx], math.exp(-dist / (2 * sigma * sigma)))

    return J