import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import utils as U


class Test(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.network = nn.Linear(10, 29)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return optim.Adam(self.network.parameters())

    def validation_step(self, batch, batch_idx):
        return self(batch).mean()

    def training_step(self, batch, batch_idx):
        return self(batch).mean()


class TestCB(pl.Callback):

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        print("validations")

    def on_epoch_end(self, trainer, pl_module):
        print("validations")


solver = Test()
dataset = U.data.dconst(torch.randn(10), 300)
trainer = pl.Trainer(
    max_epochs=1,
    callbacks=[TestCB()]
)
trainer.fit(solver, train_dataloader=U.data.DataLoader(dataset, batch_size=3))
