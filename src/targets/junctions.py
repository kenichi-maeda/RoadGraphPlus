import torch

@torch.no_grad()
def build_junction_heatmaps(batch, stride: int = 32):
    """
    Inputs:
        batch["image"]: (B, 3, H, W)
        batch["nodes"]: list of length B, each (N_i, 2) in pixel coords (x,y), y-down.
    Returns:
        heat: (B, 1, H//strides, W//stride) in [0, 1]
    """
    imgs = batch["image"]
    B, _, H, W = imgs.shape
    h, w = H // stride, W // stride
    device = imgs.device

    J = torch.zeros((B, 1, h, w), dtype=torch.float32, device=device)

    def image_to_cell(x, y, HI, WI, Hc, Wc):
        Xj = int(torch.clamp(torch.floor(x / WI * Wc), 0, Wc - 1))
        Yj = int(torch.clamp(torch.floor(y / HI * Hc), 0, Hc - 1))
        return Xj, Yj

    for b in range(B):
        nodes = batch["nodes"][b] # (N, 2)
        for x, y in nodes:
            Xj, Yj = image_to_cell(x, y, H, W, h, w)
            J[b, 0, Yj, Xj] = 1.0

    return J