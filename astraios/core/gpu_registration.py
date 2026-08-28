"""Rotation-, scale- and shift-invariant star matching on the GPU.

Proximity matching (pair each star with its nearest neighbour in pixel
coordinates) is fast but only correct when the frame-to-frame shift is small
compared with the spacing between stars; on a dithered set it pairs the wrong
stars and RANSAC then fits a confident, wrong transform. Triangle matching
does not care about the shift: for every star, triangles are formed with
pairs of its nearest neighbours, each triangle is reduced to two side ratios
(invariant under rotation, scaling and translation), the ratio vectors are
matched between the two frames, and every matched triangle votes for its
three star correspondences. The strongest votes are the true pairs.

Everything that scales with the number of stars or triangles (neighbour
search, ~1500 triangles per frame, the 1500 x 1500 feature distance matrix,
the vote table) runs as torch tensors on the device manager's device. The
final similarity fit over the ~50 voted pairs is a handful of microseconds
in OpenCV's RANSAC and stays there; it is not where the time goes.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from astraios.core.device_manager import get_device_manager

log = logging.getLogger(__name__)


@torch.no_grad()
def _local_triangles(pts: torch.Tensor, k_nn: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangles from each star and pairs of its k nearest neighbours.

    Returns (features (M, 2), vertices (M, 3)). Vertices are ordered by the
    length of the side opposite them, ascending, so that corresponding
    triangles in two frames list corresponding stars in the same order.
    """
    n = pts.shape[0]
    k = min(k_nn, n - 1)
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))
    nn = d.topk(k, dim=1, largest=False).indices  # (n, k)
    ii, jj = torch.triu_indices(k, k, 1, device=pts.device)  # neighbour pairs
    a = torch.arange(n, device=pts.device).unsqueeze(1).expand(n, ii.numel())
    tri = torch.stack([a, nn[:, ii], nn[:, jj]], dim=-1).reshape(-1, 3)
    tri = torch.unique(tri.sort(dim=1).values, dim=0)
    p = pts[tri]  # (M, 3, 2)
    opposite = torch.stack(
        [
            (p[:, 1] - p[:, 2]).norm(dim=1),
            (p[:, 2] - p[:, 0]).norm(dim=1),
            (p[:, 0] - p[:, 1]).norm(dim=1),
        ],
        dim=1,
    )
    order = opposite.argsort(dim=1)
    sides = torch.gather(opposite, 1, order)
    verts = torch.gather(tri, 1, order)
    valid = sides[:, 2] > 1e-6
    feat = torch.stack([sides[:, 0] / sides[:, 2], sides[:, 1] / sides[:, 2]], dim=1)
    return feat[valid], verts[valid]


@torch.no_grad()
def triangle_transform_gpu(
    ref_stars,
    tgt_stars,
    n_stars: int = 80,
    k_nn: int = 6,
    feature_tol: float = 0.02,
    tol_px: float = 3.0,
) -> np.ndarray | None:
    """Affine (similarity) transform target -> reference from star lists.

    ``ref_stars`` / ``tgt_stars`` are sequences of objects with ``x`` and ``y``
    (brightest first, as the detectors return them). Returns a 2x3 matrix or
    None when no consistent correspondence exists (a frame of a different
    sky, too few stars).
    """
    if len(ref_stars) < 4 or len(tgt_stars) < 4:
        return None
    dev = get_device_manager().device
    ref = torch.tensor([[s.x, s.y] for s in ref_stars[:n_stars]], device=dev, dtype=torch.float32)
    tgt = torch.tensor([[s.x, s.y] for s in tgt_stars[:n_stars]], device=dev, dtype=torch.float32)

    f_ref, v_ref = _local_triangles(ref, k_nn)
    f_tgt, v_tgt = _local_triangles(tgt, k_nn)
    if f_ref.shape[0] == 0 or f_tgt.shape[0] == 0:
        return None

    d = torch.cdist(f_tgt, f_ref)  # (M_t, M_r)
    dmin, jmin = d.min(dim=1)
    ok = dmin < feature_tol
    if int(ok.sum()) == 0:
        return None
    vt = v_tgt[ok]  # (K, 3)
    vr = v_ref[jmin[ok]]

    votes = torch.zeros((tgt.shape[0], ref.shape[0]), device=dev)
    votes.index_put_(
        (vt.reshape(-1), vr.reshape(-1)), torch.ones(vt.numel(), device=dev), accumulate=True
    )
    vals, flat = votes.flatten().sort(descending=True)
    n_keep = int((vals >= 2).sum())
    if n_keep < 4:
        return None
    flat = flat[:n_keep].cpu().numpy()
    n_ref = ref.shape[0]
    used_t: set[int] = set()
    used_r: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for f in flat:
        t, r = int(f // n_ref), int(f % n_ref)
        if t in used_t or r in used_r:
            continue
        pairs.append((t, r))
        used_t.add(t)
        used_r.add(r)
    if len(pairs) < 4:
        return None

    tgt_np = tgt.cpu().numpy()
    ref_np = ref.cpu().numpy()
    src = np.array([tgt_np[t] for t, _ in pairs], dtype=np.float32)
    dst = np.array([ref_np[r] for _, r in pairs], dtype=np.float32)
    transform, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=tol_px
    )
    if transform is None or inliers is None or int(inliers.sum()) < 4:
        return None

    # Refine on every star the first estimate brings within tolerance: more
    # points, a sub-pixel fit, and no dependence on which triangles voted.
    moved = tgt_np @ transform[:, :2].T + transform[:, 2]
    dist = torch.cdist(torch.from_numpy(moved.astype(np.float32)).to(dev), ref)
    dmin2, idx2 = dist.min(dim=1)
    m = (dmin2 < tol_px).cpu().numpy()
    if int(m.sum()) >= 4:
        refined, _ = cv2.estimateAffinePartial2D(
            tgt_np[m], ref_np[idx2.cpu().numpy()[m]],
            method=cv2.RANSAC, ransacReprojThreshold=tol_px,
        )
        if refined is not None:
            transform = refined
    return transform.astype(np.float32)
