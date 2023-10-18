import itertools

import torch
import torch.nn.functional as F
from lightning.fabric import Fabric

from .models import (
    Generator,
    HyperParams,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)


def train(fabric: Fabric, hparams: HyperParams, epochs):
    fabric.seed_everything(hparams.seed)

    generator = Generator(hparams)
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        hparams.learning_rate,
        betas=[hparams.adam_b1, hparams.adam_b2],
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        hparams.learning_rate,
        betas=[hparams.adam_b1, hparams.adam_b2],
    )

    last_epoch = -1
    torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hparams.lr_decay, last_epoch=last_epoch
    )
    torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hparams.lr_decay, last_epoch=last_epoch
    )

    fabric.setup(generator, optim_g)

    fabric.setup_module(mpd)
    fabric.setup_module(msd)
    fabric.setup_optimizers(optim_d)

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), epochs):
        for i, batch in enumerate(train_loader):
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(fabric.device, non_blocking=True))
            y = torch.autograd.Variable(y.to(fabric.device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(fabric.device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1),
                hparams.n_fft,
                hparams.num_mels,
                hparams.sampling_rate,
                hparams.hop_size,
                hparams.win_size,
                hparams.fmin,
                hparams.fmax_for_loss,
            )

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()
