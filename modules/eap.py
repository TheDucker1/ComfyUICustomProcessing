import cv2
from typing import Callable, Any, Optional
import numpy as np

def EAP(image: np.ndarray, func: Callable[..., Any], scaleConst: float, iter: Optional[int] = 5):
    H, W = image.shape[:2]
    msk = np.random.uniform(size=(H,W)).astype(np.float32)
    msk[msk > 0.5] = 1
    msk[msk != 1] = 0
    eps_weight = 1.0
    for i in range(iter):
        output = func(image, msk)
        if i+1 == iter:
            break
        value = np.sum(np.square((output - image)), axis=2)
        weight = np.sum(np.square((output - cv2.boxFilter(output, ddepth=-1, ksize=(3,3), borderType=cv2.BORDER_REFLECT_101))), axis=2)
        knapsack = value / (weight + eps_weight)
        thresh = np.sort(knapsack.flatten())
        thresh = thresh[int(H*W*(1.0 - (i+1) * scaleConst / iter))]
        knapsack = knapsack / thresh
        msk = np.ones((H,W), dtype=np.float32)
        alpha = (iter - i - 1) / max(1, (iter-1))
        msk[np.random.uniform(size=(H,W)).astype(np.float32)*alpha + 1.0 - alpha >= knapsack] = 0
    return output