import numpy as np
from PIL import Image

def rotate_k90_aug(image, depth, K, c2w=None, mask=None, k=1, *, pixel_centered=True):
    """
    extended version of rotate (180 degree)
    Rotate RGB/depth/mask by k*90 degrees and update intrinsics K and c2w (camera-to-world).

    Conventions:
      - image/depth/mask are rotated in pixel space via np.rot90 (CCW for k>0).
      - K is assumed to be of the form [[fx, 0, cx],[0, fy, cy],[0,0,1]] (no skew).
      - c2w is camera-to-world; roll is applied in the camera local frame via right-multiply.

    Args:
        image: (H,W,3) ndarray
        depth: (H,W) ndarray or None
        K:     (3,3) ndarray
        c2w:   (4,4) ndarray or None
        mask:  optional ndarray with spatial dims matching image
        k:     integer rotation steps, positive=CCW, negative=CW
        pixel_centered:
            True  -> standard 0-based pixel-center convention (uses W-1 / H-1)
            False -> edge/corner convention (uses W / H)

    Returns:
        image_rot, depth_rot, K_new, c2w_new, mask_rot
    """
    k = int(k) % 4
    if k == 0:
        return image, depth, K, c2w, mask

    H, W = image.shape[:2]
    image_rot = np.rot90(image, k)
    depth_rot = np.rot90(depth, k) if depth is not None else None
    mask_rot  = np.rot90(mask,  k) if mask is not None else None

    # Mirror constants for principal point update
    Wx = (W - 1) if pixel_centered else W
    Hy = (H - 1) if pixel_centered else H

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Pose roll in camera local frame (c2w): right-multiply
    if k == 1:      # 90° CCW
        R = np.array([[0, -1, 0, 0],
                      [1,  0, 0, 0],
                      [0,  0, 1, 0],
                      [0,  0, 0, 1]], dtype=np.float32)
        fx_new, fy_new = fy, fx
        cx_new, cy_new = cy, Wx - cx

    elif k == 3:    # 90° CW
        R = np.array([[0,  1, 0, 0],
                      [-1, 0, 0, 0],
                      [0,  0, 1, 0],
                      [0,  0, 0, 1]], dtype=np.float32)
        fx_new, fy_new = fy, fx
        cx_new, cy_new = Hy - cy, cx

    elif k == 2:    # 180°
        R = np.array([[-1, 0, 0, 0],
                      [ 0,-1, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]], dtype=np.float32)
        fx_new, fy_new = fx, fy
        cx_new, cy_new = Wx - cx, Hy - cy

    c2w_new = (c2w @ R) if c2w is not None else None

    K_new = np.array([[fx_new, 0.0, cx_new],
                      [0.0, fy_new, cy_new],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

    return image_rot, depth_rot, K_new, c2w_new, mask_rot