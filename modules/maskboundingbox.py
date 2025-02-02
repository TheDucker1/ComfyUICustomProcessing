import cv2
import numpy as np

def MaskBoundingBoxExtract(msk):
    msk = (msk*255).clip(0,255).astype(np.uint8)
    _, thresh = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
    cnt, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(cnt[0])
    return (x,y,w,h)