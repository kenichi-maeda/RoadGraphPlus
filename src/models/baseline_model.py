import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torch_geometric.nn import EdgeConv

from src.targets.junctions import build_junction_heatmaps
from src.targets.offsets import build_offset_targets
from src.data.graph_utils import build_knn_candidates_and_labels, detect_nodes, build_edge_labels, build_knn_from_pred, assign_pred_to_gt
from src.losses.junctions import junction_bce_loss
from src.losses.offsets import offset_loss


class BackBoneEncoder(nn.Module):
    def __init__(self):
        """
        Encodes aerial images using a pretrained ResNet-50 (initialized with ImageNet1K).
        Produces 2048-channel feature maps at 1/32 resolution.
        """
        super().__init__()

        resnet = models.resnet50(weights="IMAGENET1K_V2")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) # remove avgpool & fc. We do not need classification

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_sz, 3, 512, 512)
        Returns:
            torch.Tensor: Feature map (batch_sz, N_in, H, W). N_in is 2048.
        """
        return self.backbone(x)


class OffsetPredictor(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=256, out_channels=2):
        """
        Precits 2D offset vectors.

        NOTE: This version internally applies a 3-layer Conv-BN-ReLU stack
        directly to the backbone feature maps (2048 -> 256 channels) before
        producing the 2-channel offset output.
        """
        super().__init__()

        # 3x Conv-BN-ReLU feature encoder on top of backbone features
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, encoded_images):
        """
        Args:
            encoded_images (torch.Tensor): Backbone feature maps
                (batch_sz, 2048, H, W)
        Returns:
            torch.Tensor: Offset predictions (batch_sz, 2, H, W)
        """
        x = self.encoder(encoded_images)
        return self.conv(x)


class JunctionPredictor(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=256, out_channels=1):
        """
        Predicts junctions.

        NOTE: This version internally applies a 3-layer Conv-BN-ReLU stack
        directly to the backbone feature maps (2048 -> 256 channels) before
        producing the 1-channel junction heatmap.
        """
        super().__init__()

        # 3x Conv-BN-ReLU feature encoder on top of backbone features
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_images):
        """
        Args:
            encoded_images (torch.Tensor): Backbone feature maps
                (batch_sz, 2048, H, W)
        Returns:
            torch.Tensor: Junction predictions (batch_sz, 1, H, W)
        """
        x = self.encoder(encoded_images)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class NodePredictor(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=256, out_channels=256):
        """
        Predicts nodes.

        NOTE: This version internally applies a 3-layer Conv-BN-ReLU stack
        directly to the backbone feature maps (2048 -> 256 channels) before
        producing the 256-channel node feature map.
        """
        super().__init__()

        # 3x Conv-BN-ReLU feature encoder on top of backbone features
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, encoded_images):
        """
        Args:
            encoded_images (torch.Tensor): Backbone feature maps
                (batch_sz, 2048, H, W)
        Returns:
            torch.Tensor: Node predictions (batch_sz, 256, H, W)
        """
        x = self.encoder(encoded_images)
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

    
    def forward(self, node_feats: torch.Tensor, xy: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            node_feats: (N, 256)    node features from NodePredictor
            xy:         (N, 2)      absolute pixel coords (x, y) in input image
            edge_index: (2, E)      graph conectivity matrix
        Returns:
            node_emb:   (N, F)      final node embeddings after EdgeConvs (F = edgeconv_dims[-1])
            edge_index: (2, E)      complete directed edges (src, dst)
            edge_logits:(E,)        logits
        """

        N = node_feats.size(0)

        # 1) concat (x, y) to features
        x = torch.cat([node_feats, xy.to(node_feats.dtype)], dim=-1)

        # 2) 3x EdgeConv
        x = self.ec1(x, edge_index)         # (N, d1)
        x = self.ec2(x, edge_index)         # (N, d2)
        node_emb = self.ec3(x, edge_index)  # (N, d3)

        # 3) edge scoring to get probabilities
        edge_logits = self.scorer(node_emb, edge_index) # (E, ) in [0, 1]

        return node_emb, edge_index, edge_logits


class BaselineModel(pl.LightningModule):
    def __init__(self, **args):
        """
        Initializes PyTorch Lightning module
        """
        super().__init__()

        self.save_hyperparameters()

        self.lambda_j = 1.0
        self.lambda_o = 1.0
        self.lambda_e = 1.0

        # ResNet-50 backbone encoder    
        self.backboneEncoder = BackBoneEncoder()

        # CNN Predictors for different features of the road graph
        self.offsetPredictor = OffsetPredictor()       # takes backbone features (2048)
        self.junctionPredictor = JunctionPredictor()   # takes backbone features (2048)
        self.nodePredictor = NodePredictor()           # takes backbone features (2048)

        self.roadGraphGNN = RoadGraphGNN()         # takes node features (256) + (x,y)


    def forward(self, images):
        """
        Forward pass

        Args:
            images: (B, C, HI, WI)
        Retuens:
            o_pred: (B, 2, Hc, Wc) offset predictions
            j_pred: (B, 1, Hc, Wc) junction predections
            n_map:  (B, Nfeature, Hc, Wc) node feature maps
        """
        encoded_images = self.backboneEncoder(images)

        # Each predictor now internally does its own 3x Conv-BN-ReLU on the backbone features
        o_pred = self.offsetPredictor(encoded_images)
        j_pred = self.junctionPredictor(encoded_images)
        n_map = self.nodePredictor(encoded_images)

        return o_pred, j_pred, n_map

    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]      # (B,3,512,512)
        nodes_list = batch["nodes"]  # list of length B
        edges_list = batch["edges"]  # list of length B
        batch_size = images.size(0)

        o_pred, j_pred, n_map = self(images)

        # Junction & Offset targets (batch)
        J_gt = build_junction_heatmaps(batch_size, stride=32)       # (B,1,16,16)
        offset_gt, offset_mask = build_offset_targets(batch, stride=32)

        HI,WI = images.shape[-2:]
        batch_loss = 0.0

        for b in range(batch_size):
            nodes_xy_gt = nodes_list[b]     # (Nb,2)
            gt_edges    = edges_list[b] 

            # build candidate edges
            # edge_index, edge_label = build_knn_candidates_and_labels(
            #     nodes_xy_gt, gt_edges, k=8
            # )
            nodes_xy_pred, scores, cells_ij = detect_nodes(
                j_pred[b:b+1], o_pred[b:b+1], HI, WI, threshold=0.1
            )

            if nodes_xy_pred.size(0) == 0:
                continue

            edge_index = build_knn_from_pred(nodes_xy_pred)

            min_idx = assign_pred_to_gt(nodes_xy_pred, nodes_xy_gt)

            edge_label = build_edge_labels(edge_index, min_idx, gt_edges)

            Hc, Wc = n_map[b].shape[1:]
            node_map_b = n_map[b] # (256, 16, 16)

            px = nodes_xy_pred[:,0]
            py = nodes_xy_pred[:,1]

            Xc = torch.clamp((px / WI * Wc).floor().long(), 0, Wc-1)
            Yc = torch.clamp((py / HI * Hc).floor().long(), 0, Hc-1)

            node_feats = node_map_b[:, Yc, Xc].permute(1,0)

            node_emb, eidx, edge_logits = self.roadGraphGNN(
                node_feats,
                nodes_xy_pred,
                edge_index
            )

            # losses for tile b 
            loss_j = junction_bce_loss(j_pred[b:b+1], J_gt[b:b+1])
            loss_off = offset_loss(o_pred[b:b+1], offset_gt[b:b+1], offset_mask[b:b+1])
            loss_e   = F.binary_cross_entropy_with_logits(edge_logits, edge_label.float())

            loss_total = self.lambda_j*loss_j + self.lambda_o*loss_off + self.lambda_e*loss_e
            batch_loss += loss_total

            self.log("train/loss_j",   loss_j,   prog_bar=False)
            self.log("train/loss_off", loss_off, prog_bar=False)
            self.log("train/loss_e",   loss_e,   prog_bar=False)
            self.log("train/loss_total", loss_total, prog_bar=True)

        batch_loss /= batch_size
        return batch_loss
    
    def validation_step(self, *args, **kwargs):
        loss = ...
        return loss
    
    def on_validation_epoch_end(self):
        acc = ...

    def on_test_epoch_end(self):
        acc = ...