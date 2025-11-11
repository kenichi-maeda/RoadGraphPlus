from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch_geometric.nn import EdgeConv


class BackBoneEncoder(nn.Module):
    def __init__(self):
        """
        Encodes aerial images using a pretrained ResNet-50 (initialized with ImageNet1K).
        Produces 2048-channel feature maps at 1/32 resolution.
        """

        super().__init__()

        resnet = models.resnet50(weights="IMAGENTE1K_V2")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # remove avgpool & fc

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_sz, 3, 512, 512)
        Returns:
            torch.Tensor: Feature map (batch_sz, N_in, H, W). N_in is 2048.
        """
        return self.backbone(x)

class CNNFeaturEnoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        """
        3 Conv-BN-ReLU layers.
        Converts feature maps (2048) to compact features (256)
        """

        super().__init__()
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchNorm_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchNorm_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU(inplace=True)

        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchNorm_3 = nn.BatchNorm2d(out_channels)
        self.relu_3 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Encoded images (batch_sz, N_in, H, W). N_in is 2048.
        Returns:
            torch.Tensor: Extracted Features (batch_sz, N_feat, H, W). N_feat is 256.
        """
        x = self.relu_1(self.batchNorm_1(self.conv_1(x)))
        x = self.relu_2(self.batchNorm_2(self.conv_2(x)))
        x = self.relu_3(self.batchNorm_3(self.conv_3(x)))
        return x
    
class OffsetPredictor(nn.Module):
    def __init__(self, in_channels=256, out_channels=2):
        """
        Precits 2D offset vectors.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Features Maps (batch_sz, N_feat, H, W). N_feat is 256.
        Returns:
            torch.Tensor: Offset predictions (batch_sz, 2, H, W)
        """
        return self.conv(x)

    
class JunctionPredictor(nn.Module):
    def __init__(self, in_channels=256, out_channels=1):
        """
        Predicts junctions.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Feature maps (batch_sz, N_feat, H, W)
        Returns:
            torch.Tensor: Junction predictions (batch_sz, 1, H, W)
        """
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
    
class NodePredictor(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        """
        Predicts nodes.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Image features (batch_sz, feat, H, W)
        Returns:
            torch.Tensor: Node predictions (batch_sz, 256, H, W)
        """
        return self.conv(x)
    

def _mlp_edge(in_dim, out_dim):
    # EdgeConv expects a nn
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(out_dim)
    )

class EdgeScorer(nn.Module):
    # returns logits
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        xi, xj = x[dst], x[src]
        feat = torch.cat([xi, xj, (xi - xj).abs(), xi * xj], dim=-1)
        return self.mlp(feat).squeeze(-1)
    
class RoadGraphGNN(nn.Module):
    def __init__(
            self,
            in_dim: int = 256,            
            edgeconv_dims=(256, 256, 256), 
        ):
        """
        Predicts edges.
        """

        super().__init__()
        
        d0 = in_dim + 2
        d1, d2, d3 = edgeconv_dims

        self.ec1 = EdgeConv(_mlp_edge(2 * d0, d1), aggr='max')
        self.ec2 = EdgeConv(_mlp_edge(2 * d1, d2), aggr='max')
        self.ec3 = EdgeConv(_mlp_edge(2 * d2, d3), aggr='max')

        self.scorer = EdgeScorer(d3, hidden=256)

    
    def forward(self, node_feats: torch.Tensor, xy: torch.Tensor):
        """
        Args:
            node_feats: (N, 256)    node features from NodePredictor
            xy:         (N, 2)      absolute pixel coords (x, y) in input image
        Returns:
            node_emb:   (N, F)      final node embeddings after EdgeConvs (F = edgeconv_dims[-1])
            edge_index: (2, E)      complete directed edges (src, dst)
            edge_logits:(E,)        logits
        """

        N = node_feats.size(0)

        # 1) concat (x, y) to features
        x = torch.cat([node_feats, xy.to(node_feats.dtype)], dim=-1)

        # 2) build complete directed graph
        # Suppose N = 3
        # src: [0, 1, 2, 0, 1, 2, 0, 1, 2]
        # dst: [0, 0, 0, 1, 1, 1, 2, 2, 2]
        # After removing loops
        # src: [1, 2, 0, 2, 0, 1]
        # dst: [0, 0, 1, 1, 2, 2]
        # edge_index:
        # [[1, 2, 0, 2, 0, 1],
        #  [0, 0, 1, 1, 2, 2]]

        idx = torch.arange(N, device=node_feats.device)
        src = idx.repeat(N)                # [0,1,..,N-1, 0,1,..,N-1, ...]
        dst = idx.repeat_interleave(N)     # [0,0,..,0, 1,1,..,1, ...]
        mask = src != dst       # removing self-loops!
        edge_index = torch.stack([src[mask], dst[mask]], dim=0) # (2, E), E = N*(N-1)

        # 3) 3x EdgeConv
        x = self.ec1(x, edge_index)         # (N, d1)
        x = self.ec2(x, edge_index)         # (N, d2)
        node_emb = self.ec3(x, edge_index)  # (N, d3)

        # 4) edge scoring to get probabilities
        edge_logits = self.scorer(node_emb, edge_index) # (E, ) in [0, 1]

        return node_emb, edge_index, edge_logits


class BaselineModel(pl.LightningModule):
    def __init__(self, **args):
        """
        Initializes PyTorch Lightning module
        """
        super().__init__()

        self.args = Namespace(**args)

        self.jun_loss_fn = ...
        self.off_loss_fn = ...
        self.edge_loss_fn = ...

        self.backboneEncoder = BackBoneEncoder(...)

        self.cnnFeatureEncoder_1 = CNNFeaturEnoder(...)
        self.cnnFeatureEncoder_2 = CNNFeaturEnoder(...)
        self.cnnFeatureEncoder_3 = CNNFeaturEnoder(...)

        self.offsetPredictor = OffsetPredictor(...)
        self.junctionPredictor = JunctionPredictor(...)
        self.nodePredictor = NodePredictor(...)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: ...
        """

        return ...
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def training_step(self, batch, batch_idx):
        loss = ...
        return loss
    
    def validation_step(self, *args, **kwargs):
        loss = ...
        return loss
    
    def on_validation_epoch_end(self):
        acc = ...

    def on_test_epoch_end(self):
        acc = ...