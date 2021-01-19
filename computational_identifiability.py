import argparse


import torch
import torch.nn
import pytorch_lightning as pl

import datasets


class linear(pl.Module):

    def __init__(self, num_features):

        self.net = nn.Linear(num_features, 1)
    
    def forward(self, x, y):

        input = torch.cat([x, y.unsqueeze(1)], dim=1).float()

        out = self.net(input)

        return out
    
    def training_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        loss = self.loss_fct(pred, s) # CHECK THIS

        self.log('training/loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        loss = self.loss_fct(pred, s) # CHECK THIS

        self.log('validation/loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        x, y, s = batch

        pred = self.forward(x, y)

        loss = self.loss_fct(pred, s) # CHECK THIS

        self.log('test/loss', loss)

        return loss
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adagrad(self.net.parameters(), lr=self.lr)

        return optimizer


def main(args):

    







if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=['Adult', 'LSAC', 'COMPAS'], required=True)

    args = parser.parse_args()

    main(args)