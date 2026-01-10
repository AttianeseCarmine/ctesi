# eval_patchwise.py
import math
import torch
import torch.nn.functional as F

def _get_eval_stage3_cfg(config: dict) -> dict:
    # config può essere un dict con chiavi upper-case come nel tuo yaml
    # fallback robusti
    eval_cfg = {}
    if isinstance(config, dict):
        eval_cfg = config.get("EVAL_STAGE3", {}) or {}
    return eval_cfg

@torch.no_grad()
def patchwise_count_from_config(
    joint_model,
    image,                 # [B,3,H,W] (consiglio B=1)
    config: dict,
    presence_reduce="max", # "max" (più sicuro) o "mean" (più aggressivo)
):
    """
    Legge:
      config["EVAL_STAGE3"]["ENABLED"]
      config["EVAL_STAGE3"]["PATCH_SIZE"]
      config["EVAL_STAGE3"]["STRIDE"]
      config["EVAL_STAGE3"]["THRESHOLD"]

    Ritorna:
      pred_count (float), debug dict
    """
    eval_cfg = _get_eval_stage3_cfg(config)

    enabled = bool(eval_cfg.get("ENABLED", True))
    patch_size = int(eval_cfg.get("PATCH_SIZE", 448))
    stride = int(eval_cfg.get("STRIDE", patch_size))
    threshold = float(eval_cfg.get("THRESHOLD", 0.35))

    if not enabled:
        # fallback: forward standard (soft gating)
        out = joint_model(image)
        pred = out["final_density"].sum().item()
        return pred, {
            "mode": "full",
            "enabled": False,
        }

    pred, dbg = patchwise_count(
        joint_model=joint_model,
        image=image,
        patch_size=patch_size,
        stride=stride,
        threshold=threshold,
        presence_reduce=presence_reduce,
    )
    dbg["mode"] = "patchwise"
    dbg["enabled"] = True
    return pred, dbg

@torch.no_grad()
def patchwise_count(
    joint_model,
    image,                 # [B,3,H,W]
    patch_size=448,
    stride=448,
    threshold=0.35,
    presence_reduce="max",
):
    device = image.device
    B, C, H, W = image.shape
    assert B == 1, "Patchwise eval: usa B=1 (è la cosa più stabile e semplice)."

    def extract_patch(img, top, left, ps):
        bottom = min(top + ps, H)
        right  = min(left + ps, W)

        patch = img[:, :, top:bottom, left:right]
        valid_mask = torch.ones((1, 1, bottom-top, right-left), device=device, dtype=img.dtype)

        pad_h = ps - (bottom - top)
        pad_w = ps - (right  - left)
        if pad_h > 0 or pad_w > 0:
            patch = F.pad(patch, (0, pad_w, 0, pad_h), value=0.0)
            valid_mask = F.pad(valid_mask, (0, pad_w, 0, pad_h), value=0.0)

        return patch, valid_mask

    # prima patch per derivare shape density
    first_patch, first_mask = extract_patch(image, 0, 0, patch_size)

    # CLIP per ottenere dh,dw
    clip_out = joint_model.clip_ebc_model(first_patch)
    density = clip_out["ebc_density"]  # [1,1,dh,dw]
    dh, dw = density.shape[-2], density.shape[-1]

    # scaling patch pixels -> density grid
    sh = dh / float(patch_size)
    sw = dw / float(patch_size)

    GH = int(math.ceil(H * sh))
    GW = int(math.ceil(W * sw))

    density_canvas = torch.zeros((1, 1, GH, GW), device=device, dtype=density.dtype)
    weight_canvas  = torch.zeros((1, 1, GH, GW), device=device, dtype=density.dtype)

    kept = 0
    total = 0

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            total += 1
            patch, valid_mask = extract_patch(image, top, left, patch_size)

            # ZIP decide
            zip_out = joint_model.zip_model(patch)
            pi_logits = zip_out["pi_logits"]
            pi_empty = torch.sigmoid(pi_logits)
            prob_presence = 1.0 - pi_empty

            if presence_reduce == "mean":
                score = prob_presence.mean().item()
            else:
                score = prob_presence.max().item()  # più conservativo

            if score < threshold:
                continue

            kept += 1

            # CLIP solo se utile
            clip_out = joint_model.clip_ebc_model(patch)
            raw_density = clip_out["ebc_density"]  # [1,1,dh,dw]

            # resize prob_presence e valid_mask a dh,dw
            if prob_presence.shape[-2:] != raw_density.shape[-2:]:
                prob_presence = F.interpolate(prob_presence, size=raw_density.shape[-2:], mode="bilinear", align_corners=False)

            valid_mask_rs = F.interpolate(valid_mask, size=raw_density.shape[-2:], mode="nearest")

            final_density_patch = raw_density * prob_presence * valid_mask_rs

            gy = int(round(top  * sh))
            gx = int(round(left * sw))

            gy2 = min(gy + dh, GH)
            gx2 = min(gx + dw, GW)
            ph2 = gy2 - gy
            pw2 = gx2 - gx

            density_canvas[:, :, gy:gy2, gx:gx2] += final_density_patch[:, :, :ph2, :pw2]
            weight_canvas[:, :, gy:gy2, gx:gx2]  += valid_mask_rs[:, :, :ph2, :pw2]

    density_canvas = density_canvas / torch.clamp(weight_canvas, min=1.0)
    pred_count = density_canvas.sum().item()

    debug = {
        "patch_total": total,
        "patch_kept": kept,
        "kept_ratio": kept / max(total, 1),
        "threshold": threshold,
        "presence_reduce": presence_reduce,
        "patch_size": patch_size,
        "stride": stride,
    }
    return pred_count, debug
