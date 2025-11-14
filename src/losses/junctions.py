import torch
import torch.nn.functional as F

def junction_bce_loss(prob: torch.Tensor, target: torch.Tensor):
    """
    Args:
        prob: (B, 1, h, w) raw outputs
        target: (B, 1, h, w) binary map in {0, 1}
    Returns BCE loss.
    """
    assert prob.shape == target.shape, f"shape mismatch: {prob.shape} vs {target.shape}"

    return F.binary_cross_entropy(prob, target)