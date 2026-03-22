import torch
from models.encoder import VAE_Encoder
from models.decoder import VAE_Decoder
import os


def remap_hf_encoder_keys(hf_sd: dict) -> dict:
    """
    Maps HuggingFace Diffusers VAE encoder keys → your sequential VAE_Encoder keys.
    
    Your VAE_Encoder Sequential layout:
      0  = Conv2d(3, 128)           ← conv_in
      1  = ResidualBlock(128,128)   ← down_blocks.0.resnets.0
      2  = ResidualBlock(128,128)   ← down_blocks.0.resnets.1
      3  = Conv2d(128,128,stride=2) ← down_blocks.0.downsamplers.0.conv
      4  = ResidualBlock(128,256)   ← down_blocks.1.resnets.0  (has shortcut)
      5  = ResidualBlock(256,256)   ← down_blocks.1.resnets.1
      6  = Conv2d(256,256,stride=2) ← down_blocks.1.downsamplers.0.conv
      7  = ResidualBlock(256,512)   ← down_blocks.2.resnets.0  (has shortcut)
      8  = ResidualBlock(512,512)   ← down_blocks.2.resnets.1
      9  = Conv2d(512,512,stride=2) ← down_blocks.2.downsamplers.0.conv
      10 = ResidualBlock(512,512)   ← down_blocks.3.resnets.0
      11 = ResidualBlock(512,512)   ← down_blocks.3.resnets.1
      12 = ResidualBlock(512,512)   ← mid_block.resnets.0
      13 = AttentionBlock(512)      ← mid_block.attentions.0
      14 = ResidualBlock(512,512)   ← mid_block.resnets.1
      15 = GroupNorm(32,512)        ← conv_norm_out
      16 = SiLU()                   ← (no weights)
      17 = Conv2d(512,8)            ← conv_out
      18 = Conv2d(8,8,k=1)          ← quant_conv
    """
    mapping = {}

    def res(src_prefix, dst_idx, has_shortcut=False):
        """Map a ResidualBlock from HF naming to your sequential index."""
        mapping[f"{src_prefix}.norm1.weight"]   = f"{dst_idx}.groupnorm_1.weight"
        mapping[f"{src_prefix}.norm1.bias"]     = f"{dst_idx}.groupnorm_1.bias"
        mapping[f"{src_prefix}.conv1.weight"]   = f"{dst_idx}.conv_1.weight"
        mapping[f"{src_prefix}.conv1.bias"]     = f"{dst_idx}.conv_1.bias"
        mapping[f"{src_prefix}.norm2.weight"]   = f"{dst_idx}.groupnorm_2.weight"
        mapping[f"{src_prefix}.norm2.bias"]     = f"{dst_idx}.groupnorm_2.bias"
        mapping[f"{src_prefix}.conv2.weight"]   = f"{dst_idx}.conv_2.weight"
        mapping[f"{src_prefix}.conv2.bias"]     = f"{dst_idx}.conv_2.bias"
        if has_shortcut:
            mapping[f"{src_prefix}.conv_shortcut.weight"] = f"{dst_idx}.residual_layer.weight"
            mapping[f"{src_prefix}.conv_shortcut.bias"]   = f"{dst_idx}.residual_layer.bias"

    def attn(src_prefix, dst_idx):
        """Map an AttentionBlock. HF uses query/key/value separately; yours uses in_proj (combined)."""
        mapping[f"{src_prefix}.group_norm.weight"] = f"{dst_idx}.groupnorm.weight"
        mapping[f"{src_prefix}.group_norm.bias"]   = f"{dst_idx}.groupnorm.bias"
        # HF stores Q, K, V separately — must be concatenated into in_proj
        # Handled manually below; mark with special sentinel
        mapping[f"__attn_qkv__{src_prefix}"] = f"{dst_idx}.attention.in_proj"
        mapping[f"{src_prefix}.proj_attn.weight"] = f"{dst_idx}.attention.out_proj.weight"
        mapping[f"{src_prefix}.proj_attn.bias"]   = f"{dst_idx}.attention.out_proj.bias"

    # ── conv_in ──────────────────────────────────────────────────────────
    mapping["conv_in.weight"] = "0.weight"
    mapping["conv_in.bias"]   = "0.bias"

    # ── down_blocks ──────────────────────────────────────────────────────
    res("down_blocks.0.resnets.0", 1)
    res("down_blocks.0.resnets.1", 2)
    mapping["down_blocks.0.downsamplers.0.conv.weight"] = "3.weight"
    mapping["down_blocks.0.downsamplers.0.conv.bias"]   = "3.bias"

    res("down_blocks.1.resnets.0", 4, has_shortcut=True)
    res("down_blocks.1.resnets.1", 5)
    mapping["down_blocks.1.downsamplers.0.conv.weight"] = "6.weight"
    mapping["down_blocks.1.downsamplers.0.conv.bias"]   = "6.bias"

    res("down_blocks.2.resnets.0", 7, has_shortcut=True)
    res("down_blocks.2.resnets.1", 8)
    mapping["down_blocks.2.downsamplers.0.conv.weight"] = "9.weight"
    mapping["down_blocks.2.downsamplers.0.conv.bias"]   = "9.bias"

    res("down_blocks.3.resnets.0", 10)
    res("down_blocks.3.resnets.1", 11)

    # ── mid_block ────────────────────────────────────────────────────────
    res("mid_block.resnets.0", 12)
    attn("mid_block.attentions.0", 13)
    res("mid_block.resnets.1", 14)

    # ── output ───────────────────────────────────────────────────────────
    mapping["conv_norm_out.weight"] = "15.weight"
    mapping["conv_norm_out.bias"]   = "15.bias"
    mapping["conv_out.weight"]      = "17.weight"
    mapping["conv_out.bias"]        = "17.bias"
    mapping["quant_conv.weight"]    = "18.weight"
    mapping["quant_conv.bias"]      = "18.bias"


    # ── Build remapped state dict ─────────────────────────────────────────
    new_sd = {}
    for hf_key, hf_val in hf_sd.items():
        if hf_key in mapping:
            new_sd[mapping[hf_key]] = hf_val
        # Skip sentinel keys — handled below

    # ── Merge Q/K/V → in_proj ────────────────────────────────────────────
    # Your SelfAttention.in_proj = Linear(d, 3*d) storing [W_q; W_k; W_v]
    for sentinel, dst_prefix in mapping.items():
        if not sentinel.startswith("__attn_qkv__"):
            continue
        src_prefix = sentinel[len("__attn_qkv__"):]
        for suffix in ("weight", "bias"):
            q = hf_sd.get(f"{src_prefix}.query.{suffix}")
            k = hf_sd.get(f"{src_prefix}.key.{suffix}")
            v = hf_sd.get(f"{src_prefix}.value.{suffix}")
            if q is not None and k is not None and v is not None:
                new_sd[f"{dst_prefix}.{suffix}"] = torch.cat([q, k, v], dim=0)

    return new_sd


def remap_hf_decoder_keys(hf_sd: dict) -> dict:
    """
    Maps HuggingFace Diffusers VAE decoder keys → your sequential VAE_Decoder keys.

    Your VAE_Decoder Sequential layout:
      0  = Conv2d(4,4,k=1)          ← post_quant_conv
      1  = Conv2d(4,512)            ← conv_in
      2  = ResidualBlock(512,512)   ← mid_block.resnets.0
      3  = AttentionBlock(512)      ← mid_block.attentions.0
      4  = ResidualBlock(512,512)   ← mid_block.resnets.1
      5  = ResidualBlock(512,512)   ← up_blocks.0.resnets.0
      6  = ResidualBlock(512,512)   ← up_blocks.0.resnets.1
      7  = ResidualBlock(512,512)   ← up_blocks.0.resnets.2
      8  = Upsample                 ← (no weights)
      9  = Conv2d(512,512)          ← up_blocks.0.upsamplers.0.conv
      10 = ResidualBlock(512,512)   ← up_blocks.1.resnets.0
      11 = ResidualBlock(512,512)   ← up_blocks.1.resnets.1
      12 = ResidualBlock(512,512)   ← up_blocks.1.resnets.2
      13 = Upsample                 ← (no weights)
      14 = Conv2d(512,512)          ← up_blocks.1.upsamplers.0.conv
      15 = ResidualBlock(512,256)   ← up_blocks.2.resnets.0  (has shortcut)
      16 = ResidualBlock(256,256)   ← up_blocks.2.resnets.1
      17 = ResidualBlock(256,256)   ← up_blocks.2.resnets.2
      18 = Upsample                 ← (no weights)
      19 = Conv2d(256,256)          ← up_blocks.2.upsamplers.0.conv
      20 = ResidualBlock(256,128)   ← up_blocks.3.resnets.0  (has shortcut)
      21 = ResidualBlock(128,128)   ← up_blocks.3.resnets.1
      22 = ResidualBlock(128,128)   ← up_blocks.3.resnets.2
      23 = GroupNorm(32,128)        ← conv_norm_out
      24 = SiLU()                   ← (no weights)
      25 = Conv2d(128,3)            ← conv_out
    """
    mapping = {}

    def res(src_prefix, dst_idx, has_shortcut=False):
        mapping[f"{src_prefix}.norm1.weight"]   = f"{dst_idx}.groupnorm_1.weight"
        mapping[f"{src_prefix}.norm1.bias"]     = f"{dst_idx}.groupnorm_1.bias"
        mapping[f"{src_prefix}.conv1.weight"]   = f"{dst_idx}.conv_1.weight"
        mapping[f"{src_prefix}.conv1.bias"]     = f"{dst_idx}.conv_1.bias"
        mapping[f"{src_prefix}.norm2.weight"]   = f"{dst_idx}.groupnorm_2.weight"
        mapping[f"{src_prefix}.norm2.bias"]     = f"{dst_idx}.groupnorm_2.bias"
        mapping[f"{src_prefix}.conv2.weight"]   = f"{dst_idx}.conv_2.weight"
        mapping[f"{src_prefix}.conv2.bias"]     = f"{dst_idx}.conv_2.bias"
        if has_shortcut:
            mapping[f"{src_prefix}.conv_shortcut.weight"] = f"{dst_idx}.residual_layer.weight"
            mapping[f"{src_prefix}.conv_shortcut.bias"]   = f"{dst_idx}.residual_layer.bias"

    def attn(src_prefix, dst_idx):
        mapping[f"{src_prefix}.group_norm.weight"] = f"{dst_idx}.groupnorm.weight"
        mapping[f"{src_prefix}.group_norm.bias"]   = f"{dst_idx}.groupnorm.bias"
        mapping[f"__attn_qkv__{src_prefix}"]       = f"{dst_idx}.attention.in_proj"
        mapping[f"{src_prefix}.proj_attn.weight"]  = f"{dst_idx}.attention.out_proj.weight"
        mapping[f"{src_prefix}.proj_attn.bias"]    = f"{dst_idx}.attention.out_proj.bias"

    # ── post_quant_conv + conv_in ─────────────────────────────────────────
    mapping["post_quant_conv.weight"] = "0.weight"
    mapping["post_quant_conv.bias"]   = "0.bias"
    mapping["conv_in.weight"]         = "1.weight"
    mapping["conv_in.bias"]           = "1.bias"

    # ── mid_block ────────────────────────────────────────────────────────
    res("mid_block.resnets.0", 2)
    attn("mid_block.attentions.0", 3)
    res("mid_block.resnets.1", 4)

    # ── up_blocks ────────────────────────────────────────────────────────
    res("up_blocks.0.resnets.0", 5)
    res("up_blocks.0.resnets.1", 6)
    res("up_blocks.0.resnets.2", 7)
    mapping["up_blocks.0.upsamplers.0.conv.weight"] = "9.weight"
    mapping["up_blocks.0.upsamplers.0.conv.bias"]   = "9.bias"

    res("up_blocks.1.resnets.0", 10)
    res("up_blocks.1.resnets.1", 11)
    res("up_blocks.1.resnets.2", 12)
    mapping["up_blocks.1.upsamplers.0.conv.weight"] = "14.weight"
    mapping["up_blocks.1.upsamplers.0.conv.bias"]   = "14.bias"

    res("up_blocks.2.resnets.0", 15, has_shortcut=True)
    res("up_blocks.2.resnets.1", 16)
    res("up_blocks.2.resnets.2", 17)
    mapping["up_blocks.2.upsamplers.0.conv.weight"] = "19.weight"
    mapping["up_blocks.2.upsamplers.0.conv.bias"]   = "19.bias"

    res("up_blocks.3.resnets.0", 20, has_shortcut=True)
    res("up_blocks.3.resnets.1", 21)
    res("up_blocks.3.resnets.2", 22)

    # ── output ───────────────────────────────────────────────────────────
    mapping["conv_norm_out.weight"] = "23.weight"
    mapping["conv_norm_out.bias"]   = "23.bias"
    mapping["conv_out.weight"]      = "25.weight"
    mapping["conv_out.bias"]        = "25.bias"

    # ── Build remapped state dict ─────────────────────────────────────────
    new_sd = {}
    for hf_key, hf_val in hf_sd.items():
        if hf_key in mapping:
            new_sd[mapping[hf_key]] = hf_val

    # ── Merge Q/K/V → in_proj ────────────────────────────────────────────
    for sentinel, dst_prefix in mapping.items():
        if not sentinel.startswith("__attn_qkv__"):
            continue
        src_prefix = sentinel[len("__attn_qkv__"):]
        for suffix in ("weight", "bias"):
            q = hf_sd.get(f"{src_prefix}.query.{suffix}")
            k = hf_sd.get(f"{src_prefix}.key.{suffix}")
            v = hf_sd.get(f"{src_prefix}.value.{suffix}")
            if q is not None and k is not None and v is not None:
                new_sd[f"{dst_prefix}.{suffix}"] = torch.cat([q, k, v], dim=0)

    return new_sd


def load_vae(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap common wrappers
    for wrapper in ("state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and wrapper in ckpt \
                and "encoder" not in ckpt and "decoder" not in ckpt:
            ckpt = ckpt[wrapper]
            break

    keys = set(ckpt.keys())
    vae_encoder = VAE_Encoder().to(device)
    vae_decoder = VAE_Decoder().to(device)

    # ── Format 1: custom {"encoder": sd, "decoder": sd} ──────────────────
    if "encoder" in ckpt and "decoder" in ckpt \
            and isinstance(ckpt["encoder"], dict):
        vae_encoder.load_state_dict(ckpt["encoder"], strict=True)
        vae_decoder.load_state_dict(ckpt["decoder"], strict=True)
        print("[VAE] Loaded: custom {encoder, decoder} format")

    # ── Format 2: "encoder.down_blocks.*" — HF keys with prefix ──────────
    elif any(k.startswith("encoder.down_blocks.") for k in keys):
        enc_hf = {k[len("encoder."):]: v for k, v in ckpt.items()
                if k.startswith("encoder.")}
        dec_hf = {k[len("decoder."):]: v for k, v in ckpt.items()
                if k.startswith("decoder.")}

        # quant_conv / post_quant_conv live at top level without prefix
        if "quant_conv.weight" in ckpt:
            enc_hf["quant_conv.weight"] = ckpt["quant_conv.weight"]
            enc_hf["quant_conv.bias"]   = ckpt["quant_conv.bias"]
        if "post_quant_conv.weight" in ckpt:
            dec_hf["post_quant_conv.weight"] = ckpt["post_quant_conv.weight"]
            dec_hf["post_quant_conv.bias"]   = ckpt["post_quant_conv.bias"]

        enc_sd = remap_hf_encoder_keys(enc_hf)
        dec_sd = remap_hf_decoder_keys(dec_hf)
        miss_e, _ = vae_encoder.load_state_dict(enc_sd, strict=False)
        miss_d, _ = vae_decoder.load_state_dict(dec_sd, strict=False)
        if miss_e: print(f"[VAE] Encoder missing: {miss_e}")
        if miss_d: print(f"[VAE] Decoder missing: {miss_d}")
        print("[VAE] Loaded: HF-prefixed format (encoder.down_blocks.*)")

    # ── Format 3: bare HF keys "down_blocks.*" ───────────────────────────
    elif any(k.startswith("down_blocks.") for k in keys):
        enc_sd = remap_hf_encoder_keys(ckpt)
        dec_sd = remap_hf_decoder_keys(ckpt)
        miss_e, _ = vae_encoder.load_state_dict(enc_sd, strict=False)
        miss_d, _ = vae_decoder.load_state_dict(dec_sd, strict=False)
        if miss_e: print(f"[VAE] Encoder missing: {miss_e}")
        if miss_d: print(f"[VAE] Decoder missing: {miss_d}")
        print("[VAE] Loaded: bare HF format (down_blocks.*)")

    # ── Format 4: full SD ckpt "first_stage_model.*" ─────────────────────
    elif any(k.startswith("first_stage_model.") for k in keys):
        enc_hf = {k[len("first_stage_model.encoder."):]: v for k, v in ckpt.items()
                  if k.startswith("first_stage_model.encoder.")}
        dec_hf = {k[len("first_stage_model.decoder."):]: v for k, v in ckpt.items()
                  if k.startswith("first_stage_model.decoder.")}
        enc_sd = remap_hf_encoder_keys(enc_hf)
        dec_sd = remap_hf_decoder_keys(dec_hf)
        vae_encoder.load_state_dict(enc_sd, strict=False)
        vae_decoder.load_state_dict(dec_sd, strict=False)
        print("[VAE] Loaded: full SD checkpoint (first_stage_model.*)")

    else:
        sample = list(ckpt.keys())[:8]
        raise RuntimeError(f"[VAE] Unrecognised format. Sample keys: {sample}")

    for model in (vae_encoder, vae_decoder):
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    print("[VAE] Frozen and ready.")
    return vae_encoder, vae_decoder