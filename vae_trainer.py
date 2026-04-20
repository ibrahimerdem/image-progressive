import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

import config as cfg
from utils.dataset import create_dataloaders
from utils.training import AverageMeter, MetricsLogger, denormalize_image
from utils.vae_loader import load_vae


# ── SSIM loss (Gaussian-windowed, no external deps) ──────────────────────────
def _gaussian_kernel(window_size: int, sigma: float, channels: int):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).view(1, 1, window_size, window_size)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(x, y, window_size: int = 11, sigma: float = 1.5):
    """
    1 - mean SSIM.  Both tensors in [-1, 1].
    C1/C2 scaled for dynamic range 2 (i.e. L=2).
    """
    ch   = x.shape[1]
    C1   = (0.01 * 2) ** 2   # 4e-4
    C2   = (0.03 * 2) ** 2   # 3.6e-3
    pad  = window_size // 2
    kern = _gaussian_kernel(window_size, sigma, ch).to(x.device)

    mu_x  = F.conv2d(x,     kern, padding=pad, groups=ch)
    mu_y  = F.conv2d(y,     kern, padding=pad, groups=ch)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sg_x2 = F.conv2d(x * x, kern, padding=pad, groups=ch) - mu_x2
    sg_y2 = F.conv2d(y * y, kern, padding=pad, groups=ch) - mu_y2
    sg_xy = F.conv2d(x * y, kern, padding=pad, groups=ch) - mu_xy

    num  = (2 * mu_xy + C1) * (2 * sg_xy + C2)
    den  = (mu_x2 + mu_y2 + C1) * (sg_x2 + sg_y2 + C2)
    return 1.0 - (num / (den + 1e-8)).mean()


class VAEWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _run_encoder(self, x):
        for module in self.encoder:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        return mean, log_var

    def forward(self, x):
        mean, log_var = self._run_encoder(x)
        noise = torch.randn_like(mean)
        z = (mean + log_var.exp().sqrt() * noise) * cfg.VAE_SCALE
        recon = self.decoder(z)
        return recon, mean, log_var

    def reconstruct(self, x):
        mean, _ = self._run_encoder(x)
        z = mean * cfg.VAE_SCALE
        return self.decoder(z)


def _setup_ddp(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(cfg.DEVICE_IDS[rank])
    return torch.device(f"cuda:{cfg.DEVICE_IDS[rank]}")


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def _save_checkpoint(vae, optimizer, epoch, save_dir):
    m = vae.module if isinstance(vae, DDP) else vae
    path = os.path.join(save_dir, f"vae_{cfg.VERSION_NAME}_epoch_{epoch:04d}.pth")
    torch.save({
        "epoch": epoch,
        "encoder": m.encoder.state_dict(),
        "decoder": m.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def _save_val_samples(inputs, recons, save_path, num=4):
    n = min(num, inputs.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        inp  = np.clip(denormalize_image(inputs[i]).cpu().permute(1, 2, 0).float().numpy(), 0, 1)
        rec  = np.clip(denormalize_image(recons[i]).cpu().permute(1, 2, 0).float().numpy(), 0, 1)
        diff = np.abs(inp - rec).mean(axis=2)
        axes[i, 0].imshow(inp);  axes[i, 0].set_title("Defected Input");      axes[i, 0].axis("off")
        axes[i, 1].imshow(rec);  axes[i, 1].set_title("VAE Reconstruction");  axes[i, 1].axis("off")
        im = axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        axes[i, 2].set_title("Residual"); axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _run_val(vae, val_loader, sample_dir, epoch, device, logger):
    """Run validation on rank-0, save samples, log. Returns mean rec loss."""
    vae.eval()
    val_rec_m    = AverageMeter()
    sample_saved = False
    with torch.no_grad():
        for val_imgs in val_loader:
            val_imgs = val_imgs.to(device)
            with autocast(device_type="cuda"):
                recon_val = vae.module.reconstruct(val_imgs)
            val_rec_m.update(F.l1_loss(recon_val, val_imgs).item(), val_imgs.size(0))
            if not sample_saved:
                sp = os.path.join(sample_dir, f"val_epoch_{epoch:04d}.png")
                _save_val_samples(val_imgs.cpu(), recon_val.cpu(), sp)
                sample_saved = True
    print(f"[Val  {epoch:04d}]  rec={val_rec_m.avg:.5f}")
    logger.log({"epoch": epoch, "val_rec": val_rec_m.avg})
    return val_rec_m.avg


def _ddp_worker(rank, world_size, epochs, resume_path):
    device = _setup_ddp(rank, world_size)

    train_loader, val_loader = create_dataloaders(
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    save_dir   = cfg.VAE_RETRAIN_DIR
    log_dir    = os.path.join(save_dir, "logs")
    sample_dir = os.path.join(save_dir, "samples")
    if rank == 0:
        os.makedirs(save_dir,   exist_ok=True)
        os.makedirs(log_dir,    exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    encoder, decoder = load_vae(cfg.VAE_CKPT, device, freeze=False)

    vae = VAEWrapper(encoder, decoder).to(device)
    vae = DDP(vae, device_ids=[cfg.DEVICE_IDS[rank]])

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.LR, betas=(0.9, 0.999))
    scaler    = GradScaler()

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        m = vae.module
        m.encoder.load_state_dict(ckpt["encoder"])
        m.decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        if rank == 0:
            print(f"[VAE] Resumed from {resume_path} at epoch {start_epoch}")

    if rank == 0:
        logger = MetricsLogger(log_dir, "vae_train.csv")

    # ── epoch-0 baseline: val before any weight update (only when starting fresh)
    if start_epoch == 0:
        if rank == 0:
            print("[Val  0000]  (baseline — no training yet)")
            _run_val(vae, val_loader, sample_dir, 0, device, logger)
        dist.barrier()

    prev_loss = None
    for epoch in range(start_epoch + 1, epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        vae.train()

        loss_m  = AverageMeter()
        rec_m   = AverageMeter()
        ssim_m  = AverageMeter()
        kl_m    = AverageMeter()
        t0 = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:04d}/{epochs}",
            disable=(rank != 0),
            leave=False,
            ncols=100,
        )
        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)

            with autocast(device_type="cuda"):
                recon, mean, log_var = vae(imgs)
                rec_loss  = F.l1_loss(recon, imgs)
                ssim_l    = ssim_loss(recon, imgs)
                kl_loss   = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).mean()
                loss      = rec_loss + cfg.SSIM_WEIGHT * ssim_l + cfg.KL_WEIGHT * kl_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            loss_m.update(loss.item(), bs)
            rec_m.update(rec_loss.item(), bs)
            ssim_m.update(ssim_l.item(), bs)
            kl_m.update(kl_loss.item(), bs)

            if rank == 0:
                pbar.set_postfix(loss=f"{loss_m.avg:.5f}", rec=f"{rec_m.avg:.5f}", ssim=f"{1-ssim_m.avg:.4f}")

        pbar.close()

        t_loss = torch.tensor(loss_m.avg,  device=device)
        t_rec  = torch.tensor(rec_m.avg,   device=device)
        t_ssim = torch.tensor(ssim_m.avg,  device=device)
        t_kl   = torch.tensor(kl_m.avg,    device=device)
        dist.all_reduce(t_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(t_rec,  op=dist.ReduceOp.AVG)
        dist.all_reduce(t_ssim, op=dist.ReduceOp.AVG)
        dist.all_reduce(t_kl,   op=dist.ReduceOp.AVG)

        if rank == 0:
            delta = f"  Δ{t_loss.item() - prev_loss:+.5f}" if prev_loss is not None else ""
            print(f"[Epoch {epoch:04d}/{epochs}]  loss={t_loss.item():.5f}{delta}  rec={t_rec.item():.5f}  ssim={1-t_ssim.item():.4f}  kl={t_kl.item():.5f}  {time.time()-t0:.1f}s")
            prev_loss = t_loss.item()
            logger.log({"epoch": epoch, "loss": t_loss.item(), "rec_loss": t_rec.item(), "ssim": 1-t_ssim.item(), "kl_loss": t_kl.item()})

        dist.barrier()

        if epoch % cfg.VAL_EPOCH == 0 or epoch == epochs:
            if rank == 0:
                _run_val(vae, val_loader, sample_dir, epoch, device, logger)
            dist.barrier()

        if rank == 0 and epoch % cfg.SAVE_EVERY == 0:
            _save_checkpoint(vae, optimizer, epoch, save_dir)

    if rank == 0:
        _save_checkpoint(vae, optimizer, epochs, save_dir)
        print("[VAE] Training complete.")

    _cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume",  type=str, default="")
    args = parser.parse_args()

    mp.spawn(
        _ddp_worker,
        args=(cfg.WORLD_SIZE, args.epochs, args.resume or None),
        nprocs=cfg.WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
