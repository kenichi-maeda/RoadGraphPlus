import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torch_geometric.nn import EdgeConv

from src.data.graph_utils import build_knn_candidates_and_labels, detect_nodes, build_edge_labels, build_knn_from_pred, assign_pred_to_gt, build_knn_feature_space, convert_edges_to_local, detect_nodes_unet
from src.models.unet import RoadMapUNet

class BackBoneEncoder(nn.Module):
    def __init__(self):
        """
        Encodes aerial images using a pretrained ResNet-50 (initialized with ImageNet1K).
        Produces 2048-channel feature maps at 1/32 resolution.
        """

        super().__init__()

        resnet = models.resnet50(weights="IMAGENET1K_V2")
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


class SegBasedModel(pl.LightningModule):
    def __init__(self, 
                 unet_path="",
                 warmup_epochs=10, 
                 anneal_epochs=20, 
                 min_gt_prob=0.2,
                 **args):
        """
        Initializes PyTorch Lightning module
        """
        super().__init__()

        self.save_hyperparameters()

        self.unet = RoadMapUNet.load_from_checkpoint(checkpoint_path=unet_path)
        self.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad = False

        self.backboneEncoder = BackBoneEncoder()
        self.cnnFeatureEncoder = CNNFeaturEnoder(2048, 256)
        self.nodePredictor = NodePredictor(256, 256)
        self.roadGraphGNN = RoadGraphGNN()

    def forward(self, images):
        """
        Forward pass

        Args:
            images: (B, C, HI, WI)
        Retuens:
            n_map:  (B, Nfeature, Hc, Wc) node feature maps
        """
        encoded_images = self.backboneEncoder(images)
        z = self.cnnFeatureEncoder(encoded_images)
        n_map = self.nodePredictor(z)
        
        return n_map
    
    def _gt_prob(self):
        e = self.current_epoch

        if e < self.hparams.warmup_epochs:
            return 1.0
        
        t = min(1.0, (e - self.hparams.warmup_epochs) / float(self.hparams.anneal_epochs))
        return 1.0 * (1.0 - t) + self.hparams.min_gt_prob  * t
    
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    def training_step(self, batch, batch_idx):
        images = batch["image"]      # (B,3,512,512)
        nodes_list = batch["nodes"]  # list of length B
        edges_list = batch["edges"]  # list of length B
        node_ids_list = batch["node_ids"]
        B = images.size(0)

        n_map = self(images)

        HI,WI = images.shape[-2:]
        batch_loss = images.new_tensor(0.0) 

        tiles_used = 0

        for b in range(B):
            image = batch["image"][b].unsqueeze(0)
            nodes_xy_gt = nodes_list[b]     # (Nb,2)
            gt_edges_global    = edges_list[b] 
            node_ids_global = node_ids_list[b]

            gt_edges_local = convert_edges_to_local(node_ids_global, gt_edges_global)

            Hc, Wc = n_map[b].shape[1:]
            node_map_b = n_map[b] # (256, 16, 16)

            bs = 1

            p_gt   = self._gt_prob()
            use_gt = (torch.rand(1, device=images.device) < p_gt).item()

            if use_gt:
                nodes_xy = nodes_xy_gt

                px = nodes_xy[:,0]
                py = nodes_xy[:,1]

                Xc = torch.clamp((px / WI * Wc).floor().long(), 0, Wc-1)
                Yc = torch.clamp((py / HI * Hc).floor().long(), 0, Hc-1)

                node_feats = node_map_b[:, Yc, Xc].permute(1,0)

                edge_index, edge_label = build_knn_candidates_and_labels(nodes_xy_gt, gt_edges_local)
                
            else:
                with torch.no_grad():
                    pred_mask = self.unet(image)
                nodes_xy_pred = detect_nodes_unet(pred_mask=pred_mask, threshold=0.4)
                nodes_xy = nodes_xy_pred

                if nodes_xy_pred.size(0) < 2:
                    loss_e = images.new_tensor(0.0)
                    f1     = images.new_tensor(0.0)
                    batch_loss = batch_loss + loss_e
                    tiles_used += 1
                    continue 

                px = nodes_xy[:,0]
                py = nodes_xy[:,1]

                Xc = torch.clamp((px / WI * Wc).floor().long(), 0, Wc-1)
                Yc = torch.clamp((py / HI * Hc).floor().long(), 0, Hc-1)

                node_feats = node_map_b[:, Yc, Xc].permute(1,0)

                # This was wrong!!!
                # edge_index = build_knn_from_pred(nodes_xy) 

                edge_index = build_knn_feature_space(node_feats, k=6)
                min_idx = assign_pred_to_gt(nodes_xy, nodes_xy_gt, max_dist=32)
                valid_edge_mask = (min_idx[edge_index[0]] >= 0) & (min_idx[edge_index[1]] >= 0)
                edge_index = edge_index[:, valid_edge_mask]
                edge_label = build_edge_labels(edge_index, min_idx, gt_edges_local)

            if edge_index.numel() == 0 or edge_label.numel() == 0:
                loss_e = images.new_tensor(0.0)
                f1 = images.new_tensor(0.0)
            else:
                node_emb, eidx, edge_logits = self.roadGraphGNN(
                    node_feats,
                    nodes_xy,
                    edge_index
                )

                pos = edge_label.float()
                neg = 1 - pos
                num_pos = pos.sum()
                num_neg = neg.sum()

                pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=10.0)

                loss_e   = F.binary_cross_entropy_with_logits(edge_logits, edge_label.float(), pos_weight=pos_weight)

                pred = (edge_logits.sigmoid() > 0.5).float()

                tp = (pred * edge_label).sum()
                fp = (pred * (1 - edge_label)).sum()
                fn = ((1 - pred) * edge_label).sum()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

            batch_loss += loss_e
            tiles_used += 1

            self.log("train/loss_e", loss_e, prog_bar=True, sync_dist=True, batch_size=bs)
            self.log("train/edge_f1", f1, prog_bar=True, sync_dist=True, batch_size=bs)

        
        if tiles_used == 0:
            return None

        batch_loss /= tiles_used
        return batch_loss
    

    def validation_step(self, batch, batch_idx):
        metrics_pred = self._eval_on_batch_predicted(batch)
        metrics_gt  = self._eval_on_batch_gt_graph(batch)

        bs = batch["image"].size(0)

        # Predicted-node metrics 
        self.log("val/loss_e", metrics_pred["loss_e"],
                prog_bar=False, sync_dist=True, batch_size=bs)
        self.log("val/edge_f1", metrics_pred["edge_f1"],
                prog_bar=True, sync_dist=True, batch_size=bs)

        # GT-node metrics
        self.log("val/loss_e_gt", metrics_gt["loss_e_gt"],
                prog_bar=False, sync_dist=True, batch_size=bs)
        self.log("val/edge_f1_gt", metrics_gt["edge_f1_gt"],
                prog_bar=True, sync_dist=True, batch_size=bs)
    
    def test_step(self, batch, batch_idx):
        metrics = self._eval_on_batch_predicted(batch)
        bs = batch["image"].size(0)

        self.log("test/loss_e", metrics["loss_e"], prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("test/edge_f1", metrics["edge_f1"], prog_bar=True, sync_dist=True, batch_size=bs)
        
        return metrics
    
    def on_train_end(self):
        return ...
    
    def on_train_epoch_end(self):
        return ...

    def on_validation_epoch_end(self):
        return ...

    def on_test_epoch_end(self):
        return ...
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        return ...
    
    def _eval_on_batch_predicted(self, batch):
        images = batch["image"]      # (B,3,512,512)
        nodes_list = batch["nodes"]  # list of length B
        edges_list = batch["edges"]  # list of length B
        node_ids_list = batch["node_ids"]
        B = images.size(0)

        with torch.no_grad():
            n_map = self(images)


        HI,WI = images.shape[-2:]        

        total_loss_e = images.new_tensor(0.0)
        total_f1 = images.new_tensor(0.0)

        tiles_used = 0

        for b in range(B):
            image = batch["image"][b].unsqueeze(0)
            nodes_xy_gt = nodes_list[b]     # (Nb,2)
            gt_edges_global = edges_list[b] 
            node_ids_global = node_ids_list[b]

            gt_edges_local = convert_edges_to_local(node_ids_global, gt_edges_global)

            Hc, Wc = n_map[b].shape[1:]
            node_map_b = n_map[b] # (256, 16, 16)

            with torch.no_grad():
                pred_mask = self.unet(image)
            nodes_xy_pred = detect_nodes_unet(pred_mask=pred_mask, threshold=0.7)

            loss_e = images.new_tensor(0.0)
            f1     = images.new_tensor(0.0)

            if nodes_xy_pred.size(0) < 2:
                total_loss_e += 0.0
                total_f1 += 0.0
                tiles_used += 1
                continue

            nodes_xy = nodes_xy_pred

            px = nodes_xy[:,0]
            py = nodes_xy[:,1]

            Xc = torch.clamp((px / WI * Wc).floor().long(), 0, Wc-1)
            Yc = torch.clamp((py / HI * Hc).floor().long(), 0, Hc-1)

            node_feats = node_map_b[:, Yc, Xc].permute(1,0)

            # This was wrong!!!
            #edge_index = build_knn_from_pred(nodes_xy) 
            edge_index = build_knn_feature_space(node_feats, k=6)
            min_idx = assign_pred_to_gt(nodes_xy, nodes_xy_gt, max_dist=32)

            valid_edge_mask = (min_idx[edge_index[0]] >= 0) & (min_idx[edge_index[1]] >= 0)
            edge_index = edge_index[:, valid_edge_mask]
            edge_label = build_edge_labels(edge_index, min_idx, gt_edges_local)

            if edge_index.numel() > 0 and edge_label.numel() > 0:
                node_emb, eidx, edge_logits = self.roadGraphGNN(
                    node_feats,
                    nodes_xy,
                    edge_index
                )

                pos = edge_label.float()
                neg = 1 - pos
                num_pos = pos.sum()
                num_neg = neg.sum()

                pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=10.0)

                loss_e   = F.binary_cross_entropy_with_logits(edge_logits, edge_label.float(), pos_weight=pos_weight)

                pred = (edge_logits.sigmoid() > 0.5).float()

                tp = (pred * edge_label).sum()
                fp = (pred * (1 - edge_label)).sum()
                fn = ((1 - pred) * edge_label).sum()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

            total_loss_e += loss_e.detach()
            total_f1 += f1.detach()
            tiles_used += 1


        if tiles_used == 0:
             return {
                "loss_e": images.new_tensor(0.0),
                "edge_f1": images.new_tensor(0.0),
            }
        else:
            return {
                "loss_e": total_loss_e / tiles_used,
                "edge_f1": total_f1 / tiles_used,
            }
        
    def _eval_on_batch_gt_graph(self, batch):
        """
        Evaluate GNN using GT nodes only.

        Returns:
            dict with:
                - "edge_f1_gt": mean F1 over tiles in this batch
                - "loss_e_gt": mean BCE loss over tiles in this batch
        """
        images         = batch["image"]      # (B, 3, HI, WI)
        nodes_list     = batch["nodes"]      # list of length B
        edges_list     = batch["edges"]      # list of length B
        node_ids_list  = batch["node_ids"]   # list of length B

        B        = images.size(0)
        HI, WI   = images.shape[-2:]

        with torch.no_grad():
            n_map = self(images)

        total_f1     = images.new_tensor(0.0)
        total_loss_e = images.new_tensor(0.0)
        tiles        = 0

        for b in range(B):
            nodes_xy_gt     = nodes_list[b]      # (N_gt, 2)
            gt_edges_global = edges_list[b]
            node_ids_global = node_ids_list[b]

            # Skip empty tiles
            if nodes_xy_gt.numel() == 0 or len(gt_edges_global) == 0:
                continue

            gt_edges_local = convert_edges_to_local(node_ids_global, gt_edges_global)
            if gt_edges_local.numel() == 0:
                continue

            Hc, Wc      = n_map[b].shape[1:]
            node_map_b  = n_map[b]

            px = nodes_xy_gt[:, 0]
            py = nodes_xy_gt[:, 1]

            Xc = torch.clamp((px / WI * Wc).floor().long(), 0, Wc - 1)
            Yc = torch.clamp((py / HI * Hc).floor().long(), 0, Hc - 1)

            node_feats = node_map_b[:, Yc, Xc].permute(1, 0)   # (N_gt, C)

            # Build KNN edges between GT nodes
            edge_index, edge_label = build_knn_candidates_and_labels(nodes_xy_gt, gt_edges_local)
            if edge_index.numel() == 0 or edge_label.numel() == 0:
                continue

            node_emb, eidx, edge_logits = self.roadGraphGNN(
                node_feats,
                nodes_xy_gt,
                edge_index
            )

            loss_e = F.binary_cross_entropy_with_logits(edge_logits, edge_label.float())

            # F1 on GT graph
            pred = (edge_logits.sigmoid() > 0.5).float()
            tp   = (pred * edge_label).sum()
            fp   = (pred * (1 - edge_label)).sum()
            fn   = ((1 - pred) * edge_label).sum()

            precision = tp / (tp + fp + 1e-6)
            recall    = tp / (tp + fn + 1e-6)
            f1        = 2 * precision * recall / (precision + recall + 1e-6)

            total_f1     += f1
            total_loss_e += loss_e
            tiles        += 1

        if tiles == 0:
            return {
                "edge_f1_gt": images.new_tensor(0.0),
                "loss_e_gt": images.new_tensor(0.0),
            }

        return {
            "edge_f1_gt": total_f1 / tiles,
            "loss_e_gt": total_loss_e / tiles,
        }