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

    pos = target.float()
    neg = 1 - pos
    num_pos = pos.sum()
    num_neg = neg.sum()
    pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=5.0)
    return F.binary_cross_entropy_with_logits(prob, target, pos_weight)