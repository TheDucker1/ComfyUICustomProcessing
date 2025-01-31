import torch
import cv2
import numpy as np
from typing import Optional
from scipy.fftpack import fft2, ifft2

def L0Smooth(img: np.ndarray, _lambda: Optional[float] = 2e-2, _kappa: Optional[float] = 2.0, msk: Optional[np.ndarray] = None):
    _beta_max = 1e5
    isGray = len(img.shape) < 3 or img.shape[2] == 1
    S = img
    if S.ndim < 3:
        S = S[..., np.newaxis]
    N, M, D = S.shape
    beta = 2 * _lambda
    if msk is not None:
        assert(len(msk.shape) == 2)
        assert(msk.shape == S.shape[:2])

    otfx = np.zeros((N, M), dtype=np.float32)
    otfy = np.zeros((N, M), dtype=np.float32)
    otfx[0][0] = 1
    otfx[0][M-1] = -1
    otfy[0][0] = 1
    otfy[N-1][0] = -1
    otfx = fft2(otfx)
    otfy = fft2(otfy)

    Normin1 = fft2(np.squeeze(S), axes=(0,1))
    Denormin2 = np.square(np.abs(otfx)) + np.square(np.abs(otfy))
    if D > 1:
        Denormin2 = Denormin2[..., np.newaxis]
        Denormin2 = np.repeat(Denormin2, 3, axis=2)
    
    while beta < _beta_max:
        Denormin = 1 + beta * Denormin2

        h = np.diff(S, axis=1)
        last_col = S[:,0,:] - S[:,-1,:]
        last_col = last_col[:, np.newaxis, :]
        h = np.hstack([h, last_col])

        v = np.diff(S, axis=0)
        last_row = S[0,...] - S[-1,...]
        last_row = last_row[np.newaxis, ...]
        v = np.vstack([v, last_row])

        grad = np.square(h) + np.square(v)
        grad = np.sum(grad, axis=2)
        idx = grad < (_lambda / beta)
        if D > 1:
            idx = idx[..., np.newaxis]
            idx = np.repeat(idx, 3, axis=2)
        h[idx] = 0
        v[idx] = 0
        if msk is not None:
            h[msk>0.5] = 0
            v[msk>0.5] = 0

        h_diff = -np.diff(h, axis=1)
        first_col = h[:, -1, :] - h[:, 0, :]
        first_col = first_col[:, np.newaxis, :]
        h_diff = np.hstack([first_col, h_diff])

        v_diff = -np.diff(v, axis=0)
        first_row = v[-1, ...] - v[0, ...]
        first_row = first_row[np.newaxis, ...]
        v_diff = np.vstack([first_row, v_diff])

        Normin2 = h_diff + v_diff
        Normin2 = beta * fft2(Normin2, axes=(0, 1))

        FS = np.divide(np.squeeze(Normin1) + np.squeeze(Normin2),
                       Denormin)
        S = np.real(ifft2(FS, axes=(0, 1)))

        if S.ndim < 3:
            S = S[..., np.newaxis]

        beta *= _kappa
    return S.clip(0, 1)