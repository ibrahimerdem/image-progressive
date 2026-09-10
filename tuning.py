"""
Cultural Algorithm (CA) — fine-tunes the conditional GAN
(`Generator`/`Discriminator` in `models/multimodal_process.py`), optionally
warm-started from a pretrained checkpoint. The fine-tuning step mirrors the
adversarial training loop in `basic_trainer.py`.

Search space (6 genes):
  g_lr_scale  multiplies the base generator learning rate (cfg.FIXED_G_LR)
  d_lr_scale  multiplies the base discriminator learning rate (cfg.FIXED_D_LR)
  l1          weight of the L1 reconstruction loss in the generator loss
  l2          weight of the discriminator feature-matching L2 loss
  noise_dim   size of the generator's input noise vector
  embed_dim   per-feature embedding dimension shared by generator/discriminator

By default (no --checkpoint given) every individual trains from a fresh,
randomly initialized model, since noise_dim/embed_dim resize model layers
and rarely match any fixed pretrained checkpoint's shapes. Pass --checkpoint
to warm-start instead; individuals whose sampled values don't match the
checkpoint's shapes still fall back to random initialization.
"""

import copy
import json
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from models.multimodal_process import Generator, Discriminator
from utils.dataset import CustomDataset
from utils.training import calculate_psnr, load_checkpoint
import config as cfg

DEFAULT_G_LR = cfg.FIXED_G_LR
DEFAULT_D_LR = cfg.FIXED_D_LR
DEFAULT_CHANNELS = cfg.CHANNELS
DEFAULT_USE_INITIAL_IMAGE = cfg.INITIAL_IMAGE

# ----------------------------- search space ------------------------------

GENE_RANGES = {
    "g_lr_scale": (0.1, 2.0),
    "d_lr_scale": (0.1, 2.0),
    "l1":         (10.0, 100.0),
    "l2":         (50.0, 300.0),
    "noise_dim":  (75, 150),
    "embed_dim":  (256, 512),
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
    base_checkpoint: str | None,
    device: torch.device,
    ft_steps: int = 200,
    val_batches: int = 8,
    batch_size: int = 4,
    g_lr: float = DEFAULT_G_LR,
    d_lr: float = DEFAULT_D_LR,
    channels: int = DEFAULT_CHANNELS,
    use_initial_image: bool = DEFAULT_USE_INITIAL_IMAGE,
) -> float:

    train_dataset = CustomDataset(split="train")
    val_dataset = CustomDataset(split="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4, drop_last=False)

    feature_dim = train_dataset.input_data.shape[1]

    # noise_dim / embed_dim are searched genes -- round to int since they size
    # nn.Linear/ConvTranspose2d layers.
    noise_dim = int(round(genes["noise_dim"]))
    embed_dim = int(round(genes["embed_dim"]))

    generator = Generator(
        channels=channels,
        noise_dim=noise_dim,
        embed_dim=embed_dim,
        num_features=feature_dim,
        initial_image=use_initial_image,
    ).to(device)
    discriminator = Discriminator(
        channels=channels,
        embed_dim=embed_dim,
        num_features=feature_dim,
    ).to(device)

    # noise_dim/embed_dim changes resize layers, so the pretrained checkpoint
    # won't always load cleanly -- fall back to random init for individuals
    # whose sampled architecture genes don't match the checkpoint's shapes.
    # base_checkpoint=None skips loading entirely and always trains from a
    # fresh (randomly initialized) model.
    if base_checkpoint is not None:
        try:
            load_checkpoint(base_checkpoint, generator, discriminator)
        except RuntimeError as e:
            print(f"  (noise_dim={noise_dim}, embed_dim={embed_dim} don't match checkpoint shapes, "
                  f"training from random init: {e})")

    criterion = nn.BCELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    l2_loss = nn.MSELoss().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr * genes["g_lr_scale"], betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr * genes["d_lr_scale"], betas=(0.5, 0.999))

    generator.train()
    discriminator.train()
    train_iter = iter(train_loader)
    for step in range(ft_steps):
        try:
            input_image, input_feat, target_image, wrong_image = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_image, input_feat, target_image, wrong_image = next(train_iter)

        images = target_image.to(device)
        wrong_images = wrong_image.to(device)
        embeddings = input_feat.to(device)
        cond_image = input_image.to(device) if (use_initial_image and generator.image_embedding is not None) else None
        batch_size_actual = images.size(0)

        # --- Train Discriminator ---
        d_optimizer.zero_grad()

        noise = torch.randn(batch_size_actual, noise_dim, 1, 1, device=device)
        fake_images = generator(noise, embeddings, cond_image)

        real_out, _ = discriminator(images, embeddings)
        d_loss_real = criterion(real_out, torch.full_like(real_out, 1.0, device=device))

        wrong_out, _ = discriminator(wrong_images, embeddings)
        d_loss_wrong = criterion(wrong_out, torch.full_like(wrong_out, 0.0, device=device))

        fake_out, _ = discriminator(fake_images.detach(), embeddings)
        d_loss_fake = criterion(fake_out, torch.full_like(fake_out, 0.0, device=device))

        d_loss = d_loss_real + d_loss_wrong + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # --- Train Generator ---
        g_optimizer.zero_grad()

        noise = torch.randn(batch_size_actual, noise_dim, 1, 1, device=device)
        fake_images = generator(noise, embeddings, cond_image)

        out_fake, act_fake = discriminator(fake_images, embeddings)
        out_real, act_real = discriminator(images, embeddings)

        g_bce = criterion(out_fake, torch.full_like(out_fake, 1.0, device=device))
        g_l1 = genes["l1"] * l1_loss(fake_images, images)
        g_l2 = genes["l2"] * l2_loss(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())

        g_loss = g_bce + g_l1 + g_l2
        g_loss.backward()
        g_optimizer.step()

    # --- validation: generate + compare on a handful of batches ---
    generator.eval()
    discriminator.eval()
    total_l1, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (v_input_image, v_input_feat, v_target_image, _) in enumerate(val_loader):
            if i >= val_batches:
                break
            v_images = v_target_image.to(device)
            v_embeddings = v_input_feat.to(device)
            v_cond = v_input_image.to(device) if (use_initial_image and generator.image_embedding is not None) else None
            bs = v_images.size(0)

            noise = torch.randn(bs, noise_dim, 1, 1, device=device)
            v_fake = generator(noise, v_embeddings, v_cond)

            l1 = l1_loss(v_fake, v_images).item()
            batch_psnr = calculate_psnr(v_fake, v_images)

            total_l1 += l1 * bs
            total_psnr += batch_psnr * bs
            n += bs

    avg_l1 = total_l1 / max(n, 1)
    avg_psnr = total_psnr / max(n, 1)

    fitness = avg_l1 - (avg_psnr / 50.0)
    return fitness, {"val_l1": avg_l1, "val_psnr": avg_psnr}


# ------------------------------- CA loop ------------------------------------

def run_cultural_algorithm(
    base_checkpoint: str | None,
    device: torch.device,
    population_size: int = 10,
    generations: int = 5,
    accept_ratio: float = 0.34,
    ft_steps: int = 200,
    log_path: str = "./ca_finetune_log.json",
    g_lr: float = DEFAULT_G_LR,
    d_lr: float = DEFAULT_D_LR,
    channels: int = DEFAULT_CHANNELS,
    use_initial_image: bool = DEFAULT_USE_INITIAL_IMAGE,
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
                g_lr=g_lr, d_lr=d_lr, channels=channels,
                use_initial_image=use_initial_image,
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

    parser = argparse.ArgumentParser(description="Cultural Algorithm hyperparameter fine-tuning for the GAN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="optional base checkpoint to warm-start from; omit to always "
                             "train from a fresh (randomly initialized) model")
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--ft-steps", type=int, default=200)
    parser.add_argument("--g-lr", type=float, default=DEFAULT_G_LR,
                        help="base generator learning rate before the g_lr_scale gene is applied")
    parser.add_argument("--d-lr", type=float, default=DEFAULT_D_LR,
                        help="base discriminator learning rate before the d_lr_scale gene is applied")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--no-initial-image", action="store_true",
                        help="disable conditioning on the initial image (default: enabled)")
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
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        channels=args.channels,
        use_initial_image=not args.no_initial_image,
    )