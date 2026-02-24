# geometry_augment.py
import numpy as np
import PIL.Image
from training.datasets.utils import cropping  # already handles intrinsics updates

def _rng(rng):
    return rng or np.random.default_rng()

def random_crop(image, depth, K, target_size=(256, 224), aspect_ratio_range=[1.0, 1.5], rng=None):
    rng = _rng(rng)
    W, H = image.size
    # randomly choose a aspect ratio in aspect_ratio_range, and randomly choose a Ws, Hs (abide aspect ratio), to make Wtarget < Ws <= W, Htarget < Hs <= H
    target_w, target_h = target_size
    if target_w > W or target_h > H:
        raise ValueError(f"target_size {target_size} exceeds image size {(W, H)}")

    min_ar = max(aspect_ratio_range[0], target_w / H)
    max_ar = min(aspect_ratio_range[1], W / target_h)
    if min_ar > max_ar:
        # Fallback to the closest feasible aspect ratio
        min_ar = max_ar
    ar = rng.uniform(min_ar, max_ar)

    ch_min = max(target_h, target_w / ar)
    ch_max = min(H, W / ar)
    if ch_min > ch_max:
        ch_min = ch_max
    ch = int(np.floor(rng.uniform(ch_min, ch_max)))
    ch = max(target_h, min(ch, H))
    cw = int(np.floor(ch * ar))
    cw = max(target_w, min(cw, W))

    left = int(rng.integers(0, W - cw + 1))
    top = int(rng.integers(0, H - ch + 1))

    image, depth, K = cropping.crop_image_depthmap(image, depth, K, crop_bbox=(left, top, left + cw, top + ch))
    return image, depth, K
