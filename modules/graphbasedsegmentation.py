import numpy as np
import numba
import cv2

# stuck with 2e9 pixels for now
edge_type = np.dtype([
    ('u', np.int32),
    ('v', np.int32),
    ('w', np.float32)
])

@numba.njit
def populate_edge_list(imgFlatten: np.ndarray, H: int, W: int, edgeList: np.ndarray, using_8_conn: int):
    edgeLen = 0
    normw = 3 if imgFlatten.shape[1] == 1 else 1
    offsetList = ((-1, 0), (0, -1), (-1, 1), (-1 ,-1))
    offsetLen = 4 if using_8_conn != 0 else 2
    for y in range(H):
        for x in range(W):
            ii = y*W+x
            for i in range(offsetLen):
                dy, dx = offsetList[i]
                yy = y + dy
                xx = x + dx
                if (yy < 0 or xx < 0 or xx >= W):
                    continue
                jj = yy*W+xx
                edgeList[edgeLen]['u'] = ii
                edgeList[edgeLen]['v'] = jj
                WW = (imgFlatten[ii] - imgFlatten[jj])
                edgeList[edgeLen]['w'] = np.linalg.norm(WW) * normw
                edgeLen += 1
    return edgeLen

@numba.njit
def graphSegment(H: int, W: int, edgeList: np.ndarray, K: float, max_cc: int):
    n = H*W
    DSU_p  = np.array([i for i in range(n)], dtype=np.int32)
    DSU_sz = np.array([1 for i in range(n)], dtype=np.int32)
    THRESH = np.array([K for i in range(n)], dtype=np.float32)
    def DSU_find(u: int) -> int:
        Stk = [u]
        while DSU_p[Stk[-1]] != Stk[-1]:
            Stk.append(DSU_p[Stk[-1]])
        p = Stk[-1]
        while Stk:
            a = Stk.pop()
            DSU_p[a] = p
        return p
    def DSU_join(u: int, v: int) -> int:
        u = DSU_find(u)
        v = DSU_find(v)
        if u == v:
            return 0
        if DSU_sz[u] > DSU_sz[v]:
            u, v = v, u
        DSU_p[u] = v
        DSU_sz[v] += DSU_sz[u]
        return 1
    num_cc = n
    for e in edgeList:
        u, v, w = e['u'], e['v'], e['w']
        u = DSU_find(u)
        v = DSU_find(v)
        if u == v:
            continue
        if w <= min(THRESH[u],THRESH[v]):
            num_cc -= DSU_join(u, v)
            u = DSU_find(u)
            THRESH[u] = w + (K / DSU_sz[u])

    merge_threshold = int(H*W*0.01 + 1)
    for e in edgeList:
        u, v, w = e['u'], e['v'], e['w']
        u = DSU_find(u)
        v = DSU_find(v)
        if u == v:
            continue
        if min(DSU_sz[u], DSU_sz[v]) <= merge_threshold:
            num_cc -= DSU_join(u, v)            
    
    while num_cc > max_cc:
        merge_threshold *= 2
        for e in edgeList:
            u, v, w = e['u'], e['v'], e['w']
            u = DSU_find(u)
            v = DSU_find(v)
            if u == v:
                continue
            if min(DSU_sz[u], DSU_sz[v]) <= merge_threshold:
                num_cc -= DSU_join(u, v)
                if num_cc <= max_cc:
                    break

    Label = np.array([-1 for i in range(n)], dtype=np.int32)
    LabelSet = np.array([-1 for i in range(n)], dtype=np.int32)
    LabelSz = 0
    for i in range(n):
        lbl = DSU_find(i)
        if LabelSet[lbl] == -1:
            LabelSet[lbl] = LabelSz
            LabelSz += 1
        Label[i] = LabelSet[lbl]
    return Label.reshape((H, W)), num_cc

def GraphBasedSegmentation(image: np.ndarray, sigma: float, K: float, max_cc: int, using_8_conn: int = True):
    H, W = image.shape[:2]
    C = 1 if len(image.shape) < 3 else image.shape[2]
    img_smooth = cv2.GaussianBlur(image, ksize=(0,0), sigmaX = max(0.01, sigma))
    imgFlatten = img_smooth.reshape((H*W, C))
    if using_8_conn:
        edgeList = np.empty(H*W*4, dtype=edge_type)
    else:
        edgeList = np.empty(H*W*2, dtype=edge_type)
    edgeLen = populate_edge_list(imgFlatten, H, W, edgeList, using_8_conn)
    edgeList = edgeList[:edgeLen]
    sorted_indexes = np.argsort(edgeList, order=('w','u','v'))
    edgeList = edgeList[sorted_indexes]
    label, num_label = graphSegment(H, W, edgeList, K, max_cc)
    return label, num_label
    
if __name__ == '__main__':
    import random
    img = cv2.imread('bg.png', cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    seg, nseg = GraphBasedSegmentation(img, 0.5, 500, 20)
    rand_col = lambda : np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
    img2 = np.zeros_like(img)
    for lbl in range(nseg):
        img2[seg==lbl] = rand_col()
    cv2.imshow('i', img)
    cv2.imshow('2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()