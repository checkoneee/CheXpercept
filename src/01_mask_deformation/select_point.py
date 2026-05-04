import numpy as np


def _grid_points(mask_arr, positive, grid_step):
    pts = []
    h, w = mask_arr.shape
    for y in range(0, h, grid_step):
        for x in range(0, w, grid_step):
            if (mask_arr[y, x] == 255) == positive:
                pts.append((x, y))
    return pts


def select_points(mask, grid_step=64):
    if hasattr(mask, 'mode'):
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
    else:
        mask_arr = mask

    pos_pts = _grid_points(mask_arr, positive=True,  grid_step=grid_step)
    neg_pts = _grid_points(mask_arr, positive=False, grid_step=grid_step)
    return pos_pts, neg_pts
