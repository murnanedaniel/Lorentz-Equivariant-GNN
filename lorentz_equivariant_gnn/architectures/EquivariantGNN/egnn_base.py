import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear
import torch
import numpy as np

from .utils import *

class EGNNBase(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        """
        Initialise the Lightning Module that can scan over different Equivariant GNN training regimes
        """
        # Assign hyperparameters
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        self.trainset, self.valset = load_datasets(self.hparams["input_dir"], self.hparams["data_split"])

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        p, y = torch.squeeze(batch["p"]), batch["y"]

        n_nodes = p.size()[0]

        edges = get_edges(n_nodes)

        output, x = self(p, edges)

        # output, _ = L_GCL.compute_radials(edges, x)
        # output = torch.sigmoid(torch.mean(output).unsqueeze(0))

        output = torch.mean(output)
        output = torch.sigmoid(output)
        output = output.unsqueeze(0)

        loss = F.binary_cross_entropy(output, y.float())

        prediction = output.round()
        
        tp = (prediction == y).sum().item()
        t = y.sum().item()
        acc = tp / max(t, 1)
        
        self.log_dict({"train_loss": loss, "train_acc": acc})

        return loss

    def validation_step(self, batch, batch_idx):

        p, y = torch.squeeze(batch["p"]), batch["y"]
        n_nodes = p.size()[0]

        edges = get_edges(n_nodes)
        row, column = edges

        h, _ = compute_radials(edges, p)  # torch.zeros(n_nodes, 1)

        output, x = self(h, p, edges)

        # output, _ = L_GCL.compute_radials(edges, x)
        # output = torch.sigmoid(torch.mean(output).unsqueeze(0))

        output = torch.mean(output)
        output = torch.sigmoid(output)
        output = output.unsqueeze(0)

        prediction = output.round()

        loss = F.binary_cross_entropy(output, y.float())

        tp = (prediction == y).sum().item()
        t = y.sum().item()
        acc = tp / max(t, 1)

        current_lr = self.optimizers().param_groups[0]["lr"]
        
        self.log_dict({"val_loss": loss, "acc": acc, "current_lr": current_lr})


        return {
            "loss": loss
        }
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        
    
def load_datasets(input_dir, data_split):
    
    train_file = os.path.join(input_dir, 'test.h5')
    with pd.HDFStore(train_file, mode = 'r') as store:
        train_df = store['table']

    val_file = os.path.join(input_dir, 'val.h5')
    with pd.HDFStore(train_file, mode = 'r') as store:
        val_df = store['table']

    all_p, all_y = build_dataset(train_df, data_split[0])
    train_dataset = JetDataset(all_p, all_y)
#     train_loader = DataLoader(train_dataset)#, batch_size = 100, shuffle = True)

    val_all_p, val_all_y = build_dataset(val_df, data_split[1])
    val_dataset = JetDataset(val_all_p, val_all_y)
#     val_loader = DataLoader(val_dataset)

    return train_dataset, val_dataset

def compute_radials(edge_index, x):
    """
    Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j

    :param edge_index: Array containing the connection between nodes
    :param x: The coordinates (node embeddings)
    :return: Minkowski distances (squared) and coordinate differences x_i - x_j
    """

    row, col = edge_index
    coordinate_differences = x[row] - x[col]
    minkowski_distance_squared = coordinate_differences ** 2
    minkowski_distance_squared[:, 0] = -minkowski_distance_squared[:, 0]  # Place minus sign on time coordinate as \eta = diag(-1, 1, 1, 1)
    radial = torch.sum(minkowski_distance_squared, 1).unsqueeze(1)
    return radial, coordinate_differences
        
