# modified from https://github.com/guchengxi1994/RTV-in-Python/blob/master/rtv2.py

import numpy as np
import cv2
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import cg

def tsmooth(I, lambda_=0.01, sigma=3.0, sharpness=0.02, maxIter=4, msk=None):
    I = I.copy()
    if len(I.shape) < 3:
        I = I[..., np.newaxis]
    if msk is not None:
        oI = I.copy()
        if len(msk.shape) < 3:
            msk = msk[..., np.newaxis]
        H, W = I.shape[:2]
        def avgA(M):
            H, W = M.shape[:2]
            return (np.pad(M[:,1:W,:], ((0,0), (0,1), (0,0)), 'edge')  + 
                    np.pad(M[1:H,:,:], ((0,1), (0,0), (0,0)), 'edge')  +
                    np.pad(M[:,:W-1,:], ((0,0), (1,0), (0,0)), 'edge') +
                    np.pad(M[:H-1,:,:], ((1,0), (0,0), (0,0)), 'edge') )
        M = msk
        nM = 1 - M
        Mw = 1 - M + 1e-5
        piter = 32
        for i in range(piter):
            I = I * Mw
            I = avgA(I)
            Mw = avgA(Mw)
            I = I / Mw
            Mw = Mw * 0.25
            Mw[Mw < 0.25] = 0
            Mw[Mw > 0] = 1
            Mw += 1e-5
            I = I * M + oI * nM
    x = I.copy()
    sigma_iter = sigma
    lambda_ /= 2.0
    dec = 2.0

    for _ in range(maxIter):
        wx, wy = computeTextureWeights(x, sigma_iter, sharpness)
        x = solveLinearEquation(I, wx, wy, lambda_)
        sigma_iter = max(sigma_iter / dec, 0.5)

    return x


def computeTextureWeights(fin, sigma, sharpness):
    fx = np.diff(fin, axis=1)
    fx = np.pad(fx, ((0, 0), (0, 1), (0, 0)), mode="constant")
    fy = np.diff(fin, axis=0)
    fy = np.pad(fy, ((0, 1), (0, 0), (0, 0)), mode="constant")

    vareps_s = sharpness
    vareps = 0.001

    wto = (
        np.maximum(np.sum(np.sqrt(fx**2 + fy**2), axis=2) / fin.shape[2], vareps_s)
        ** -1
    )
    fbin = lpfilter(fin, sigma)
    gfx = np.diff(fbin, axis=1)
    gfx = np.pad(gfx, ((0, 0), (0, 1), (0, 0)), mode="constant")
    gfy = np.diff(fbin, axis=0)
    gfy = np.pad(gfy, ((0, 1), (0, 0), (0, 0)), mode="constant")

    wtbx = np.maximum(np.sum(np.abs(gfx), axis=2) / fin.shape[2], vareps) ** -1
    wtby = np.maximum(np.sum(np.abs(gfy), axis=2) / fin.shape[2], vareps) ** -1

    retx = wtbx * wto
    rety = wtby * wto

    retx[:, -1] = 0
    rety[-1, :] = 0

    return retx, rety


def conv2_sep(im, sigma):
    ksize = int(max(round(5 * sigma), 1))
    ksize |= 1
    
    g = cv2.getGaussianKernel(ksize, sigma)
    ret = cv2.filter2D(im, -1, g)
    ret = cv2.filter2D(ret, -1, g.T)
    return ret


def lpfilter(FImg, sigma):
    FBImg = np.zeros_like(FImg)
    for ic in range(FImg.shape[2]):
        FBImg[:, :, ic] = conv2_sep(FImg[:, :, ic], sigma)
    return FBImg


def solveLinearEquation(IN, wx, wy, lambda_):
    r, c, ch = IN.shape
    k = r * c

    dx = -lambda_ * wx.ravel(order="F")
    dy = -lambda_ * wy.ravel(order="F")

    B = np.vstack((dx, dy))
    d = [-r, -1]
    A = spdiags(B, d, k, k)

    e = dx
    w = np.pad(dx[:-r], (r, 0), "constant")
    s = dy
    n = np.pad(dy[:-1], (1, 0), "constant")
    D = 1 - (e + w + s + n)
    A = A + A.T + spdiags(D, 0, k, k)

    A = csr_matrix(A)

    OUT = np.zeros_like(IN)
    for i in range(ch):
        tin = IN[:, :, i].ravel(order="F")
        tout, _ = cg(A, tin)
        #tout = spsolve(A.astype(np.float64), tin.astype(np.float64))
        OUT[:, :, i] = tout.reshape((r, c), order="F")

    return OUT


if __name__ == "__main__":
    from eap import EAP
    SmoothFunc = lambda x, msk: tsmooth(x, msk=msk)
    img = cv2.imread("bg.png")
    S = img.astype(np.float32) / 255.
    img2 = EAP(S, SmoothFunc, 0.2)
    img2 = (img2*255).clip(0,255).astype(np.uint8)
    cv2.imshow("1", img)
    cv2.imshow("2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()