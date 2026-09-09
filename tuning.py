"""
Cultural Algorithm (CA) — fine-tunes training hyperparameters of the
diffusion model in `models/latent_diffusion_attn.py` around a pretrained
checkpoint. No classifier-free-guidance (CFG) settings are involved here:
the model this script fine-tunes has no CFG support (no context dropout,
no null-token conditioning, no guidance_scale at sampling time), so the
search space below is plain training/fine-tuning hyperparameters only.

Why CA operates on *hyperparameters*, not raw network weights:
CA (and evolutionary algorithms generally) are population-based and
gradient-free -- searching directly over millions of network weights this
way is computationally infeasible. What IS tractable, and is what actual
"CA-assisted fine-tuning" papers in the GAN/diffusion literature do, is
using CA to search a small hyperparameter space, where each individual's
fitness = "fine-tune the pretrained model for a few epochs with these
hyperparameters, then evaluate on the validation set."

Search space (3 genes):
  lr_scale        multiplies the base learning rate
  rgb_loss_weight weight of the auxiliary pixel-space L1 loss in p_loss
  timesteps_used  fraction of the 1000 diffusion timesteps actually
                   sampled from during this fine-tune stage (a cheap proxy
                   for "does spending more capacity on a timestep subrange
                   help fine-tuning" -- kept simple as a [0.5, 1.0] scalar
                   that subsets `torch.randint` range)

Belief space (classic CA components, per Reynolds 1994):
  situational : best individual found so far
  normative   : per-gene [low, high] range, narrowed around accepted individuals
  (domain/history knowledge omitted -- not meaningfully applicable to a
  low-dimensional continuous hyperparameter space)

Influence function: each new individual is sampled from a Gaussian
centered on the normative range's midpoint (with situational knowledge
biasing the mean further for the elite fraction), clipped to the range.
Acceptance function: top `accept_ratio` of the population (by fitness)
update the belief space each generation.
"""

import copy
import json
import os
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from models.latent_diffusion_attn import LatentDiffusionConditioned, GaussianDiffusion
from utils.dataset import CustomDataset
from utils.vae_loader import load_vae

# ------------------------------- defaults -----------------------------------
# These used to be pulled in via `from config import *`. They're now plain
# defaults, overridable from the CLI, so this script has no dependency on
# config.py at all.
DEFAULT_LR = 0.00005
DEFAULT_EMB_DIM = 768
DEFAULT_BASE_CHANNELS = 256
DEFAULT_USE_INITIAL_IMAGE = True
# The diffusion model operates on 4-channel VAE latents, not raw RGB pixels --
# a frozen pretrained VAE encodes/decodes between the two.
DEFAULT_VAE_CKPT = "checkpoints/diffusion/diffusion_pytorch_model.bin"
DEFAULT_LATENT_CHANNELS = 4

# ----------------------------- search space ------------------------------

GENE_RANGES = {
    "lr_scale":        (0.1, 2.0),
    "rgb_loss_weight":  (0.0, 0.1),
    "timesteps_used":   (0.75, 1.0),
}
GENE_NAMES = list(GENE_RANGES.keys())


@dataclass
class Individual:
    genes: dict
    fitness: float = None


# ----------------------------- belief space -------------------------------

@dataclass
class BeliefSpace:
    normative: dict = field(default_factory=lambda: dict(GENE_RANGES))
    situational: Individual = None

    def update(self, accepted: list):
        """Narrow the normative range toward the accepted individuals'
        min/max per gene, and update situational knowledge with the best."""
        for gene in GENE_NAMES:
            values = [ind.genes[gene] for ind in accepted]
            lo, hi = min(values), max(values)
            # shrink toward the accepted range but never collapse to a point
            pad = max((hi - lo) * 0.1, 1e-6)
            self.normative[gene] = (lo - pad, hi + pad)

        best = min(accepted, key=lambda ind: ind.fitness)  # lower fitness = better (see fitness())
        if self.situational is None or best.fitness < self.situational.fitness:
            self.situational = copy.deepcopy(best)


def random_individual() -> Individual:
    return Individual(genes={g: random.uniform(*GENE_RANGES[g]) for g in GENE_NAMES})


def influence(belief: BeliefSpace, elite_bias: float = 0.3) -> Individual:
    """Sample a new individual guided by the belief space."""
    genes = {}
    for gene in GENE_NAMES:
        lo, hi = belief.normative[gene]
        lo, hi = max(lo, GENE_RANGES[gene][0]), min(hi, GENE_RANGES[gene][1])
        mid = (lo + hi) / 2
        std = max((hi - lo) / 4, 1e-6)

        if belief.situational is not None and random.random() < elite_bias:
            mid = belief.situational.genes[gene]

        val = random.gauss(mid, std)
        val = min(max(val, GENE_RANGES[gene][0]), GENE_RANGES[gene][1])
        genes[gene] = val
    return Individual(genes=genes)


# ------------------------------ fitness ------------------------------------

def fine_tune_and_evaluate(
    genes: dict,
    base_checkpoint: str,
    device: torch.device,
    ft_steps: int = 200,
    val_batches: int = 8,
    batch_size: int = 4,
    sample_steps: int = 50,
    lr: float = DEFAULT_LR,
    emb_dim: int = DEFAULT_EMB_DIM,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    use_initial_image: bool = DEFAULT_USE_INITIAL_IMAGE,
    vae_checkpoint: str = DEFAULT_VAE_CKPT,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
) -> float:
    """Fine-tunes a fresh copy of the pretrained model for `ft_steps`
    optimizer steps with the given hyperparameters, then evaluates L1 +
    PSNR on a handful of validation batches (kept small -- this function is
    called once per individual per generation, so its cost dominates the
    whole CA run).

    Returns a *fitness to minimize* (lower is better): L1 minus a PSNR
    bonus, so the CA searches for hyperparameters that improve both.
    """
    train_dataset = CustomDataset(split="train")
    val_dataset = CustomDataset(split="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4, drop_last=False)

    # Frozen VAE: encodes RGB targets -> latents for training, decodes
    # sampled latents -> RGB for validation metrics. Never trained here.
    vae_encoder, vae_decoder = load_vae(vae_checkpoint, device)
    vae_encoder.eval()
    vae_decoder.eval()
    for p in vae_encoder.parameters():
        p.requires_grad = False
    for p in vae_decoder.parameters():
        p.requires_grad = False

    model = LatentDiffusionConditioned(
        latent_channels=latent_channels,
        emb_dim=emb_dim,
        base_channels=base_channels,
        use_initial_image=use_initial_image,
    ).to(device)

    ckpt = torch.load(base_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt["model_state_dict"])

    schedule = GaussianDiffusion(timesteps=1000).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr * genes["lr_scale"], betas=(0.5, 0.999))

    max_t = int(schedule.timesteps * genes["timesteps_used"])

    model.train()
    train_iter = iter(train_loader)
    for step in range(ft_steps):
        try:
            input_image, input_feat, target_image, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_image, input_feat, target_image, _ = next(train_iter)

        target_image = target_image.to(device)
        input_feat = input_feat.to(device)
        initial_images = input_image.to(device) if use_initial_image else None

        # encode the RGB target into a latent (frozen VAE, no grad)
        with torch.no_grad():
            latent_h, latent_w = target_image.size(2) // 8, target_image.size(3) // 8
            vae_noise = torch.randn(target_image.size(0), latent_channels, latent_h, latent_w, device=device)
            target_latent = vae_encoder(target_image, vae_noise)

        # timesteps_used gene: restrict which t's this fine-tune stage sees
        batch_size_actual = target_latent.size(0)
        t = torch.randint(0, max_t, (batch_size_actual,), device=device)
        noise = torch.randn_like(target_latent)
        x_t = schedule.q_sample(target_latent, t, noise)
        pred_noise = model(x_t, t, input_feat, initial_images)
        noise_loss = F.mse_loss(pred_noise, noise)

        loss = noise_loss
        if genes["rgb_loss_weight"] > 0:
            x0_pred = schedule.predict_start(x_t, t, pred_noise)
            rgb_loss = F.l1_loss(x0_pred, target_latent)
            loss = loss + genes["rgb_loss_weight"] * rgb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- validation: sample + compare on a handful of batches ---
    model.eval()
    total_l1, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (v_input_image, v_input_feat, v_target_image, _) in enumerate(val_loader):
            if i >= val_batches:
                break
            v_target_image = v_target_image.to(device)
            v_input_feat = v_input_feat.to(device)
            v_initial = v_input_image.to(device) if use_initial_image else None

            latent_h, latent_w = v_target_image.size(2) // 8, v_target_image.size(3) // 8
            latent_shape = (v_target_image.size(0), latent_channels, latent_h, latent_w)

            fake_latent = schedule.sample(
                model, v_input_feat, steps=sample_steps,
                latent_shape=latent_shape,
                initial_images=v_initial,
            )
            fake = torch.clamp(vae_decoder(fake_latent), -1.0, 1.0)

            l1 = F.l1_loss(fake, v_target_image).item()
            mse = F.mse_loss(fake, v_target_image).item()
            psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-8))).item()  # data range [-1,1] -> 4.0

            bs = v_target_image.size(0)
            total_l1 += l1 * bs
            total_psnr += psnr * bs
            n += bs

    avg_l1 = total_l1 / max(n, 1)
    avg_psnr = total_psnr / max(n, 1)

    # fitness to MINIMIZE: lower L1 is better, higher PSNR is better
    # (PSNR normalized by /50 so both terms are roughly comparable in scale)
    fitness = avg_l1 - (avg_psnr / 50.0)
    return fitness, {"val_l1": avg_l1, "val_psnr": avg_psnr}


# ------------------------------- CA loop ------------------------------------

def run_cultural_algorithm(
    base_checkpoint: str,
    device: torch.device,
    population_size: int = 10,
    generations: int = 5,
    accept_ratio: float = 0.34,
    ft_steps: int = 200,
    log_path: str = "./ca_finetune_log.json",
    lr: float = DEFAULT_LR,
    emb_dim: int = DEFAULT_EMB_DIM,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    use_initial_image: bool = DEFAULT_USE_INITIAL_IMAGE,
    vae_checkpoint: str = DEFAULT_VAE_CKPT,
    latent_channels: int = DEFAULT_LATENT_CHANNELS,
):
    accept_n = max(1, int(population_size * accept_ratio))
    belief = BeliefSpace()
    log = []

    population = [random_individual() for _ in range(population_size)]

    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} ===")
        for idx, ind in enumerate(population):
            fitness, metrics = fine_tune_and_evaluate(
                ind.genes, base_checkpoint, device, ft_steps=ft_steps,
                lr=lr, emb_dim=emb_dim, base_channels=base_channels,
                use_initial_image=use_initial_image,
                vae_checkpoint=vae_checkpoint, latent_channels=latent_channels,
            )
            ind.fitness = fitness
            print(f"  individual {idx}: genes={ {k: round(v, 4) for k, v in ind.genes.items()} } "
                  f"-> fitness={fitness:.4f} ({metrics})")
            log.append({"generation": gen, "individual": idx, "genes": ind.genes,
                        "fitness": fitness, "metrics": metrics})

        population.sort(key=lambda ind: ind.fitness)
        accepted = population[:accept_n]
        belief.update(accepted)

        print(f"  best so far: fitness={belief.situational.fitness:.4f} genes="
              f"{ {k: round(v, 4) for k, v in belief.situational.genes.items()} }")

        # next generation: keep the accepted elite, refill the rest via influence()
        population = accepted + [influence(belief) for _ in range(population_size - accept_n)]

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nCA fine-tuning done. Log written to {log_path}")
    print(f"Best genes found: {belief.situational.genes}")
    print(f"Best fitness: {belief.situational.fitness:.4f}")

    return belief.situational


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cultural Algorithm hyperparameter fine-tuning")
    parser.add_argument("--checkpoint", type=str, 
                        default="./checkpoints/diffusion/diffusion_attentional.pth")
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--ft-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="base learning rate before the lr_scale gene is applied")
    parser.add_argument("--emb-dim", type=int, default=DEFAULT_EMB_DIM)
    parser.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    parser.add_argument("--no-initial-image", action="store_true",
                        help="disable conditioning on the initial image (default: enabled)")
    parser.add_argument("--vae-checkpoint", type=str, default=DEFAULT_VAE_CKPT,
                        help="frozen VAE checkpoint used to encode/decode between RGB and latents")
    parser.add_argument("--latent-channels", type=int, default=DEFAULT_LATENT_CHANNELS)
    parser.add_argument("--log-path", type=str, default="./ca_finetune_log.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = run_cultural_algorithm(
        base_checkpoint=args.checkpoint,
        device=device,
        population_size=args.population_size,
        generations=args.generations,
        ft_steps=args.ft_steps,
        log_path=args.log_path,
        lr=args.lr,
        emb_dim=args.emb_dim,
        base_channels=args.base_channels,
        use_initial_image=not args.no_initial_image,
        vae_checkpoint=args.vae_checkpoint,
        latent_channels=args.latent_channels,
    )