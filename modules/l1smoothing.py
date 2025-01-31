import numpy as np
import cv2
import scipy.linalg
import scipy.sparse.linalg
import numba
from numba.typed import List
from typing import Optional
from .graphbasedsegmentation import GraphBasedSegmentation
from numba import prange

@numba.njit
def DrawLine(x1: int, y1: int, x2: int, y2: int) -> List[tuple[int,int]]:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return [(x1,y1)]
    swap = abs(dx) < abs(dy)
    if swap:
        x1,y1=y1,x1
        x2,y2=y2,x2
        dx,dy=dy,dx
    m = abs(dy/dx)
    m *= -1 if y2 < y1 else 1
    xx = x1
    yy : float = y1
    direction = -1 if x1>x2 else 1
    result = [(xx,int(yy))]
    while xx != x2:
        xx += direction
        yy += m
        result.append((xx,round(yy))) # some inconsistency with negative number, but good enough
    if swap:
        result = [(y,x) for x,y in result]
    return result

@numba.njit(parallel=True)
def construct_LTL_ijv(lab: np.ndarray, 
    edge_preserving: int,
    grad: np.ndarray, 
    half_window: int,
    presearch_window: List[List[tuple[int,int]]],
    eta: float,
    sigma: float,
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray
    ):

    wsize = (half_window*2+1)
    w2size = wsize*wsize

    h, w = lab.shape[:2]
    sig2ma = 1/sigma/sigma/2
    def fixi(i: int):
        if i < 0:
            i = -i
        elif i >= h:
            i = h-2 - (i-h)
        return i
    def fixj(j: int):
        if j < 0:
            j = -j
        elif j >= w:
            j = w-2 - (j-w)
        return j
    def getidx(i: int, j: int):
        return i*w+j
    
    for i in prange(h):
        for j in prange(w):
            for di in prange(-half_window, half_window+1):
                for dj in prange(-half_window, half_window+1):
                    didx = (di+half_window)*wsize + (dj+half_window)
                    max_grad = 0
                    if edge_preserving:
                        for ddi, ddj in presearch_window[didx]:
                            ni, nj = fixi(i+ddi), fixj(j+ddj)
                            _grad = grad[ni,nj]
                            if _grad > max_grad:
                                max_grad = _grad
                    ni, nj = fixi(i+di), fixj(j+dj)
                    ww: float = max(eta * max_grad * max_grad, np.dot(lab[ni,nj]-lab[i,j], lab[ni,nj]-lab[i,j]))
                    ww = np.exp(-ww*sig2ma)

                    idx = ((i*w+j)*w2size + didx) * 4

                    row[idx] = getidx(i,j)
                    col[idx] = getidx(i,j)
                    val[idx] = ww*ww

                    row[idx+1] = getidx(ni,nj)
                    col[idx+1] = getidx(ni,nj)
                    val[idx+1] = ww*ww
                        
                    row[idx+2] = getidx(i,j)
                    col[idx+2] = getidx(ni,nj)
                    val[idx+2] = -ww*ww

                    row[idx+3] = getidx(ni,nj)
                    col[idx+3] = getidx(i,j)
                    val[idx+3] = -ww*ww

@numba.njit(parallel=True)
def construct_GTG_ijv(lab: np.ndarray,
    superpixels: np.ndarray,
    sigma: float,
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray
    ):
    h, w = lab.shape[:2]
    sig2ma = 1/sigma/sigma/2
    n = superpixels.shape[0]
    def getidx(i: int, j: int):
        return i*w+j
    for i in prange(n):
        y1, x1 = superpixels[i]
        for j in prange(n):
            if j == i:
                continue
            idx = (i*(n-1)+j)*4
            y2, x2 = superpixels[j]
            WW = lab[y1,x1]-lab[y2,x2]
            ww: float = np.dot(WW, WW)
            ww = np.exp(-ww*sig2ma)
            
            row[idx] = getidx(y1,x1)
            col[idx] = getidx(y1,x1)
            val[idx] = ww*ww

            row[idx+1] = getidx(y2,x2)
            col[idx+1] = getidx(y2,x2)
            val[idx+1] = ww*ww

            row[idx+2] = getidx(y1,x1)
            col[idx+2] = getidx(y2,x2)
            val[idx+2] = -ww*ww

            row[idx+3] = getidx(y2,x2)
            col[idx+3] = getidx(y1,x1)
            val[idx+3] = -ww*ww
    
@numba.njit(parallel=True)
def compute_LTdb(lab: np.ndarray,
    edge_preserving: int,
    grad: np.ndarray,
    half_window: int,
    presearch_window: List[List[tuple[int,int]]],
    eta: float,
    sigma: float,
    db: np.ndarray,
    data_out: np.ndarray):

    wsize = (half_window*2+1)
    w2size = wsize*wsize

    h, w = lab.shape[:2]
    data_out.fill(0)
    sig2ma = 1/2/sigma/sigma
    def fixi(i: int):
        if i < 0:
            i = -i
        elif i >= h:
            i = h-2 - (i-h)
        return i
    def fixj(j: int):
        if j < 0:
            j = -j
        elif j >= w:
            j = w-2 - (j-w)
        return j
    def getidx(i: int, j: int):
        return i*w+j
    
    for i in prange(h):
        for j in prange(w):
            for di in prange(-half_window, half_window+1):
                for dj in prange(-half_window, half_window+1):
                    didx = (di+half_window)*wsize + (dj+half_window)
                    max_grad = 0
                    if edge_preserving:
                        for ddi, ddj in presearch_window[didx]:
                            ni, nj = fixi(i+ddi), fixj(j+ddj)
                            _grad = grad[ni,nj]
                            if _grad > max_grad:
                                max_grad = _grad
                    ni, nj = fixi(i+di), fixj(j+dj)
                    ww: float = max(eta * max_grad * max_grad, np.dot(lab[ni,nj]-lab[i,j], lab[ni,nj]-lab[i,j]))
                    ww = np.exp(-ww*sig2ma)

                    kidx = (i*w+j)*w2size + didx

                    idx1: int = getidx(i, j)
                    data_out[idx1] += ww * db[kidx]
                    idx2: int = getidx(ni, nj)
                    data_out[idx2] += -ww * db[kidx]

@numba.njit(parallel=True)
def compute_GTdb(lab: np.ndarray,
    superpixels: np.ndarray,
    sigma: float,
    db: np.ndarray,
    data_out: np.ndarray):
    h, w = lab.shape[:2]
    data_out.fill(0)
    sig2ma = 1/sigma/sigma/2
    n = superpixels.shape[0]
    def getidx(i: int, j: int):
        return i*w+j
    for i in prange(n):
        y1, x1 = superpixels[i]
        for j in prange(n):
            if j == i:
                continue
            y2, x2 = superpixels[j]
            kidx = i * (n-1) + j

            WW = lab[y1,x1]-lab[y2,x2]
            ww: float = np.dot(WW,WW)
            ww = np.exp(-ww*sig2ma)

            idx1: int = getidx(y1,x1)
            data_out[idx1] += ww * db[kidx]
            idx2: int = getidx(y2,x2)
            data_out[idx2] -= ww * db[kidx]

@numba.njit(parallel=True)
def compute_Lz(lab:np.ndarray,
    edge_preserving: int,
    grad: np.ndarray,
    half_window: int,
    presearch_window: np.ndarray,
    eta: float,
    sigma: float,
    z: np.ndarray,
    data_out: np.ndarray):

    wsize = (half_window*2+1)
    w2size = wsize*wsize

    h, w = lab.shape[:2]
    data_out.fill(0)
    sig2ma = 1/2/sigma/sigma
    def fixi(i: int):
        if i < 0:
            i = -i
        elif i >= h:
            i = h-2 - (i-h)
        return i
    def fixj(j: int):
        if j < 0:
            j = -j
        elif j >= w:
            j = w-2 - (j-w)
        return j
    def getidx(i: int, j: int):
        return i*w+j

    for i in prange(h):
        for j in prange(w):
            for di in prange(-half_window, half_window+1):
                for dj in prange(-half_window, half_window+1):
                    max_grad = 0
                    didx = (di+half_window)*wsize + (dj+half_window)
                    if edge_preserving:
                        for ddi, ddj in presearch_window[didx]:
                            ni, nj = fixi(i+ddi), fixj(j+ddj)
                            _grad = grad[ni,nj]
                            if _grad > max_grad:
                                max_grad = _grad
                    ni, nj = fixi(i+di), fixj(j+dj)
                    ww: float = max(eta * max_grad * max_grad, np.dot(lab[ni,nj]-lab[i,j], lab[ni,nj]-lab[i,j]))
                    ww = np.exp(-ww*sig2ma)

                    kidx = (i*w+j)*w2size + didx

                    idx1: int = getidx(i, j)
                    data_out[kidx] += ww * z[idx1]
                    idx2: int = getidx(ni, nj)
                    data_out[kidx] += -ww * z[idx2]

@numba.njit(parallel=True)
def compute_Gz(lab: np.ndarray,
    superpixels: np.ndarray,
    sigma: float,
    z: np.ndarray,
    data_out: np.ndarray):

    h, w = lab.shape[:2]
    data_out.fill(0)
    sig2ma = 1/sigma/sigma/2
    n = superpixels.shape[0]
    def getidx(i: int, j: int):
        return i*w+j
    for i in prange(n):
        y1, x1 = superpixels[i]
        for j in prange(n):
            if j == i:
                continue
            y2, x2 = superpixels[j]
            kidx = i * (n-1) + j

            WW = lab[y1,x1]-lab[y2,x2]
            ww: float = np.dot(WW, WW)
            ww = np.exp(-ww*sig2ma)

            idx1: int = getidx(y1,x1)
            idx2: int = getidx(y2,x2)

            data_out[kidx] += ww * (z[idx1]-z[idx2])

# https://stackoverflow.com/a/17678303
def find_nearest_vector(array, value):
    idx = np.array([np.linalg.norm(array-value, axis=-1)]).argmin()
    return idx

def L1Smooth(img: np.ndarray, 
            alpha: float = 50.0, # local weight
            beta: float = 5.0, # global weight (if not edge-preserve)
            gamma: float = 2.5, # approximate
            _lambda: float = 5.0, # regulation
            maxIter: int = 4, # maximum iteration step
            kappa: float = 0.3, # L weight in LAB
            sigma: float = 1, # lab weight
            half_window: int = 3, # half window size

            eta: float = 15, # gradient weight in edge preserving mode
            edge_preserving: int = 0, # enable edge preserving mode (ignore global if true)

            global_size: int = 20, # max global superpixels
            threshold: float = 0.001,
            msk: Optional[np.ndarray] = None,
    ):
    h, w = img.shape[:2]
    c = 1 if len(img.shape) < 3 else img.shape[2]
    tot_size = h*w
    w_size = (half_window*2+1)**2

    if edge_preserving:
        presearch_window = [[] for _ in range((half_window*2+1)**2)]
        idx = 0
        for di in range(-half_window, half_window+1):
            for dj in range(-half_window, half_window+1):
                presearch_window[idx] = List(DrawLine(0,0,di,dj))
                
                idx += 1
        presearch_window = List(presearch_window)
    else:
        presearch_window = List([List([(0,0)]) for _ in range(1)])
    
    if edge_preserving:
        if c == 1:
            imggray = img.copy()
        else:
            imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(imggray, (3,3), 0)
        grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)
        grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)
        grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    else:
        grad = np.array([[1]], dtype=np.float32)
    if c == 1:
        lab = cv2.merge([img * kappa, np.zeros_like(img), np.zeros_like(img)])
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l *= kappa
        lab = cv2.merge([l,a,b])

    rowL = np.zeros((tot_size*w_size*4), dtype=np.int32)
    colL = np.zeros((tot_size*w_size*4), dtype=np.int32)
    valL = np.zeros((tot_size*w_size*4), dtype=np.float32)
    construct_LTL_ijv(lab,edge_preserving,grad,half_window,presearch_window,eta,sigma,rowL,colL,valL)
    valL *= alpha

    if msk is not None:
        rowI = np.array([i for i in range(tot_size)], dtype=np.int32)
        colI = np.array([i for i in range(tot_size)], dtype=np.int32)
        valI = (img.flatten() * (1 - msk.flatten())).astype(np.float32)
    else:
        rowI = np.array([i for i in range(tot_size)], dtype=np.int32)
        colI = np.array([i for i in range(tot_size)], dtype=np.int32)
        valI = np.array([1.0] * tot_size, dtype=np.float32)
    valI *= gamma

    if not edge_preserving:
        global_segmentation, global_real_size = GraphBasedSegmentation(img, 0.5, 500, global_size)
        superpixels = []
        for i in range(global_real_size):
            repmsk = global_segmentation==i
            if c == 3:
                repmsk = repmsk[..., np.newaxis]
                repmsk = np.repeat(repmsk, 3, axis=2)
            img_seg = np.ma.masked_where(repmsk, img)
            avg_col = np.mean(img_seg, axis=(0,1))
            rep_pixel = find_nearest_vector(img_seg, avg_col)
            rep_h = rep_pixel // w
            rep_w = rep_pixel % w
            superpixels.append((h,w))
        superpixels = np.array(superpixels)
        rowG = np.zeros((global_real_size * (global_real_size-1)), dtype=np.int32)
        colG = np.zeros((global_real_size * (global_real_size-1)), dtype=np.int32)
        valG = np.zeros((global_real_size * (global_real_size-1)), dtype=np.float32)
        construct_GTG_ijv(lab,superpixels,sigma,rowG,colG,valG)
        valG *= beta
    else:
        superpixels = np.array([(1,1)], dtype=np.int32)

    row = (rowG, rowL, rowI) if not edge_preserving else (rowL, rowI)
    col = (colG, colL, colI) if not edge_preserving else (colL, colI)
    val = (valG, valL, valI) if not edge_preserving else (valL, valI)

    row = np.concatenate(row)
    col = np.concatenate(col)
    val = np.concatenate(val)
    A = scipy.sparse.coo_matrix((val, (row,col)), shape=(tot_size, tot_size))
    
    zIn = []
    if c == 1:
        if msk is not None:
            zIn.append(np.ravel(img) * (1 - msk.flatten()))
        else:
            zIn.append(np.ravel(img))
    else:
        for x in cv2.split(img):
            if msk is not None:
                zIn.append(np.ravel(x) * (1 - msk.flatten()))
            else:
                zIn.append(np.ravel(x))
    zOut = [zin.copy() for zin in zIn]
    zOut2 = [zin.copy() for zin in zIn]

    d1 = [np.zeros((tot_size*w_size), dtype=np.float32) for _ in range(c)]
    b1 = [np.zeros_like(dx) for dx in d1]

    if not edge_preserving:
        d2 = [np.zeros(global_real_size*(global_real_size-1), dtype=np.float32) for _ in range(c)]
        b2 = [np.zeros_like(dx) for dx in d2]
    
    def Shrink(y, _gamma):
        ind = y < 0
        y = np.abs(y) - _gamma
        y[y<0] = 0
        y[ind] *= -1
        return y
    
    dd1 = np.zeros_like(d1[0])
    bb1 = np.zeros_like(b1[0])
    db1 = np.zeros_like(zIn[0])
    Lz = np.zeros_like(d1[0])
    if not edge_preserving:
        dd2 = np.zeros_like(d2[0])
        bb2 = np.zeros_like(b2[0])
        db2 = np.zeros_like(zIn[0])
        Gz = np.zeros_like(d2[0])
    for _ in range(maxIter):
        zDiff = 0
        for i in range(c):
            compute_LTdb(lab, edge_preserving, grad, half_window, presearch_window, eta, sigma, d1[i]-b1[i], db1)
            if not edge_preserving:
                compute_GTdb(lab, superpixels, sigma, d2[i]-b2[i], db2)
                v = gamma * zIn[i] + _lambda * db1 + db2
            else:
                v = gamma * zIn[i] + _lambda * db1
            zOut2[i], _ = scipy.sparse.linalg.cg(A, v)
            zDiff += np.dot(zOut2[i] - zOut[i], zOut2[i] - zOut[i])
            zOut[i] = zOut2[i].copy()
        if zDiff <= threshold:
            break
        for i in range(c):
            compute_Lz(lab, edge_preserving, grad, half_window, presearch_window, eta, sigma, zOut[i], Lz)
            dd1 = Shrink(Lz + b1[i], 1 / _lambda)
            bb1 = Lz + b1[i] - dd1
            d1[i] = dd1.copy()
            b1[i] = bb1.copy()

            if not edge_preserving:
                compute_Gz(lab, superpixels, sigma, zOut[i], Gz)
                dd2 = Shrink(Gz + b2[i], 1 / _lambda)
                bb2 = Gz + b2[i] - dd2
                d2[i] = dd2.copy()
                b2[i] = bb2.copy()
    if c == 3:
        img2 = cv2.merge([x.reshape((h,w)) for x in zOut])
    else:
        img2 = zOut[0].reshape((h,w))
    return img2


if __name__ == '__main__':
    img = cv2.imread('bg.png', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300,300))
    h,w,c = img.shape[:3]
    S = img.astype(np.float32) / 255.
    img2 = L1Smooth(S, edge_preserving=1)
    img2 = (img2*255).clip(0,255).astype(np.uint8)
    cv2.imshow('1', img)
    cv2.imshow('2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()