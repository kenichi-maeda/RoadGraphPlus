import torch

def offset_loss(pred_offset: torch.Tensor, target_offset: torch.Tensor, mask: torch.Tensor):
    """
    Args:
        pred_offset, target_offset: (B, 2, h, w)
        mask: (B, 1, h, w) 1 if a junction exists, else 0
    Returns:
        Masked MSE loss 
    """
    # broadcase mask to 2 channels
    m = mask.expand_as(pred_offset) # (B, 2, h, w)
    num_pos = m.sum().clamp(min=1.0) # avoid zero div error
    return ((pred_offset - target_offset) ** 2 * m).sum() / num_pos