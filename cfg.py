"""
Separate ablation: takes a (CA fine-tuned or not) checkpoint and compares
sampling quality WITH vs WITHOUT classifier-free guidance, across a small
sweep of guidance_scale values. This is deliberately independent of
`cultural_algorithm_finetune.py` -- the CA search optimizes *training*
hyperparameters, while guidance_scale is a *sampling-time* setting that
costs nothing to sweep on an already-trained model, so no fine-tuning is
needed here at all.

guidance_scale=1.0 is the "CFG off" baseline (single conditional forward
pass per step -- see the docstring in diffusion_model_cfg.py's
GaussianDiffusion.sample for why this is mathematically equivalent to
having no guidance term).
"""

import argparse
import json

import torch
import torch.nn.functional as F

from config import * 
from models.latent_diffusion_cfg import LatentDiffusionConditioned, GaussianDiffusion
from utils.dataset import CustomDataset

def load_model(checkpoint_path, device):
    model = LatentDiffusionConditioned(
        latent_channels=4,
        emb_dim=EMB_DIM,
        base_channels=BASE_CHANNELS,
        use_initial_image=INITIAL_IMAGE,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("generator_state", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_guidance_scale(model, schedule, val_loader, device, guidance_scale,
                             sample_steps=50, max_batches=None):
    total_l1, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (v_input_image, v_input_feat, v_target_image, _) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            v_target_image = v_target_image.to(device)
            v_input_feat = v_input_feat.to(device)
            v_initial = v_input_image.to(device) if INITIAL_IMAGE else None

            fake = schedule.sample(
                model, v_input_feat, steps=sample_steps,
                latent_shape=v_target_image.shape,
                initial_images=v_initial,
                guidance_scale=guidance_scale,
            )
            fake = torch.clamp(fake, -1.0, 1.0)

            l1 = F.l1_loss(fake, v_target_image).item()
            mse = F.mse_loss(fake, v_target_image).item()
            psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-8))).item()

            bs = v_target_image.size(0)
            total_l1 += l1 * bs
            total_psnr += psnr * bs
            n += bs

    return {"guidance_scale": guidance_scale, "val_l1": total_l1 / max(n, 1),
            "val_psnr": total_psnr / max(n, 1), "n_samples": n}


def main():
    parser = argparse.ArgumentParser(description="Compare CFG on vs off (and a scale sweep) on a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 7.5],
                         help="1.0 == CFG off; the rest sweep CFG-on strength")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=None,
                         help="limit validation batches for a quick check; default = full val set")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default="./cfg_on_off_comparison.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)
    schedule = GaussianDiffusion(timesteps=1000).to(device)

    val_dataset = CustomDataset(split="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=4, drop_last=False)

    results = []
    for scale in args.guidance_scales:
        label = "CFG OFF" if scale == 1.0 else f"CFG ON (scale={scale})"
        print(f"Evaluating {label} ...")
        res = evaluate_guidance_scale(model, schedule, val_loader, device, scale,
                                       sample_steps=args.sample_steps, max_batches=args.max_batches)
        res["label"] = label
        results.append(res)
        print(f"  -> val_l1={res['val_l1']:.4f}  val_psnr={res['val_psnr']:.2f}  (n={res['n_samples']})")

    off = next(r for r in results if r["guidance_scale"] == 1.0)
    best_on = min((r for r in results if r["guidance_scale"] != 1.0), key=lambda r: r["val_l1"], default=None)

    print("\n=== Summary ===")
    print(f"CFG OFF (scale=1.0):        L1={off['val_l1']:.4f}  PSNR={off['val_psnr']:.2f}")
    if best_on is not None:
        print(f"Best CFG ON (scale={best_on['guidance_scale']}): "
              f"L1={best_on['val_l1']:.4f}  PSNR={best_on['val_psnr']:.2f}")
        delta_l1 = off["val_l1"] - best_on["val_l1"]
        delta_psnr = best_on["val_psnr"] - off["val_psnr"]
        print(f"Delta (CFG ON - CFG OFF): L1 {'-' if delta_l1 >= 0 else '+'}{abs(delta_l1):.4f}  "
              f"PSNR {'+' if delta_psnr >= 0 else '-'}{abs(delta_psnr):.2f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()