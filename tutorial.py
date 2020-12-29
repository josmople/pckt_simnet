from pytorch_lightning import callbacks
import torch
import torch.nn as nn
import pytorch_lightning as pl


class Solver(pl.LightningModule):

    def __init__(self, lr):
        self.lr = lr
        self.generator = nn.Conv2d(3, 1)
        self.discriminator = nn.Conv2d(3, 1)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.generator.parameters(), lr=self.lr), torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        x, y = batch
        out = self(x)
        return (out - y) ** 2


import pytorch_lightning.callbacks as cb
x = pl.Trainer(
    callbacks=[
        cb.EarlyStopping(),
        cb.ModelCheckpoint()
    ]
)


class CustomCallbak(pl.Callback):

    def on_batch_start(self, trainer, pl_module):
        return super().on_batch_start(trainer, pl_module)
