# get model here https://github.com/lllyasviel/DanbooRegion

import h5py
import numpy as np

import torch
import torch.nn as nn

import cv2
from scipy.ndimage import label
import random

class _r_block(nn.Module):
    def __init__(self, in_filters, nb_filters):
        super(_r_block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_filters, nb_filters, stride=(1, 1), kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(nb_filters, nb_filters, stride=(1, 1), kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.model(x)

class _dog(nn.Module):
    def __init__(self):
        super(_dog, self).__init__()
    def forward(self, x):
        down = nn.functional.avg_pool2d(x, 2)
        up = nn.functional.interpolate(down, scale_factor=2)
        return down, x-up

class _cat(nn.Module):
    def __init__(self):
        super(_cat, self).__init__()
    def forward(self, x, y):
        return torch.cat([nn.functional.interpolate(x, scale_factor=2), y], axis=1)

class diff_net_torch(nn.Module):
    def __init__(self):
        super(diff_net_torch, self).__init__()

        self.r_block_1 = _r_block(3, 16)

        self.dog_1 = _dog()
        self.r_block_2 = _r_block(16, 32)

        self.dog_2 = _dog()
        self.r_block_3 = _r_block(32, 64)

        self.dog_3 = _dog()
        self.r_block_4 = _r_block(64, 128)

        self.dog_4 = _dog()
        self.r_block_5 = _r_block(128, 256)

        self.dog_5 = _dog()
        self.r_block_6 = _r_block(256, 512)

        self.cat_1 = _cat()
        self.r_block_7 = _r_block(512+256, 256)

        self.cat_2 = _cat()
        self.r_block_8 = _r_block(256+128, 128)

        self.cat_3 = _cat()
        self.r_block_9 = _r_block(128+64, 64)

        self.cat_4 = _cat()
        self.r_block_10 = _r_block(64+32, 32)

        self.cat_5 = _cat()
        self.r_block_11 = _r_block(32+16, 16)

        self.conv = nn.Conv2d(16, 1, stride=(1, 1), kernel_size=(3, 3), padding='same')

    def forward(self, x):
        c512 = self.r_block_1(x)
        c256, l512 = self.dog_1(c512)
        c256 = self.r_block_2(c256)
        c128, l256 = self.dog_2(c256)
        c128 = self.r_block_3(c128)
        c64, l128 = self.dog_3(c128)
        c64 = self.r_block_4(c64)
        c32, l64 = self.dog_4(c64)
        c32 = self.r_block_5(c32)
        c16, l32 = self.dog_5(c32)
        c16 = self.r_block_6(c16)
        d32 = self.cat_1(c16, l32)
        d32 = self.r_block_7(d32)
        d64 = self.cat_2(d32, l64)
        d64 = self.r_block_8(d64)
        d128 = self.cat_3(d64, l128)
        d128 = self.r_block_9(d128)
        d256 = self.cat_4(d128, l256)
        d256 = self.r_block_10(d256)
        d512 = self.cat_5(d256, l512)
        d512 = self.r_block_11(d512)
        y = self.conv(d512)
        return y

MAPPING = {
    'c512_c1': ('r_block_1', 'model', '0',),
    'c512_c2': ('r_block_1', 'model', '2',),
    'c256_c1': ('r_block_2', 'model', '0',),
    'c256_c2': ('r_block_2', 'model', '2',),
    'c128_c1': ('r_block_3', 'model', '0',),
    'c128_c2': ('r_block_3', 'model', '2',),
    'c64_c1': ('r_block_4', 'model', '0',),
    'c64_c2': ('r_block_4', 'model', '2',),
    'c32_c1': ('r_block_5', 'model', '0',),
    'c32_c2': ('r_block_5', 'model', '2',),
    'c16_c1': ('r_block_6', 'model', '0',),
    'c16_c2': ('r_block_6', 'model', '2',),
    'd32_c1': ('r_block_7', 'model', '0',),
    'd32_c2': ('r_block_7', 'model', '2',),
    'd64_c1': ('r_block_8', 'model', '0',),
    'd64_c2': ('r_block_8', 'model', '2',),
    'd128_c1': ('r_block_9', 'model', '0',),
    'd128_c2': ('r_block_9', 'model', '2',),
    'd256_c1': ('r_block_10', 'model', '0',),
    'd256_c2': ('r_block_10', 'model', '2',),
    'd512_c1': ('r_block_11', 'model', '0',),
    'd512_c2': ('r_block_11', 'model', '2',),
    'op': ('conv',)
}

def count_all(labeled_array: np.ndarray, all_counts):
    a = labeled_array.copy() - 1
    for i in range(len(all_counts)):
        all_counts[i] = np.sum(labeled_array==i)

def trace_all(labeled_array: np.ndarray, xs, ys, cs):
    a = labeled_array.copy() - 1
    for i in range(len(cs)):
        xx, yy = np.where(a == i)
        xs[i] = xx
        ys[i] = yy
        cs[i] = len(xx)

def find_all(labeled_array: np.ndarray):
    hist_size = int(np.max(labeled_array))
    if hist_size == 0:
        return []
    all_counts = [0 for _ in range(hist_size)]
    count_all(labeled_array, all_counts)
    xs = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
    ys = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
    cs = [0 for item in all_counts]
    trace_all(labeled_array, xs, ys, cs)
    filled_area = []
    for _ in range(hist_size):
        filled_area.append((xs[_], ys[_]))
    return filled_area
def get_fill(image):
    labeled_array, num_features = label(image / 255)
    filled_area = find_all(labeled_array)
    return filled_area
def up_fill(fills, cur_fill_map):
    new_fillmap = cur_fill_map.copy()
    padded_fillmap = np.pad(cur_fill_map, [[1, 1], [1, 1]], 'constant', constant_values=0)
    max_id = np.max(cur_fill_map)
    for item in fills:
        points0 = padded_fillmap[(item[0] + 1, item[1] + 0)]
        points1 = padded_fillmap[(item[0] + 1, item[1] + 2)]
        points2 = padded_fillmap[(item[0] + 0, item[1] + 1)]
        points3 = padded_fillmap[(item[0] + 2, item[1] + 1)]
        all_points = np.concatenate([points0, points1, points2, points3], axis=0)
        pointsets, pointcounts = np.unique(all_points[all_points > 0], return_counts=True)
        if len(pointsets) == 1 and item[0].shape[0] < 128:
            new_fillmap[item] = pointsets[0]
        else:
            max_id += 1
            new_fillmap[item] = max_id
    return new_fillmap

def thinning(fillmap, max_iter=100):
    """Fill area of line with surrounding fill color.
    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.copy()

    for iterNum in range(max_iter):
        # Get points of line. if there is not point, stop.
        line_points = np.where(result == line_id)
        if not len(line_points[0]) > 0:
            break

        # Get points between lines and fills.
        line_mask = np.full((h, w), 255, np.uint8)
        line_mask[line_points] = 0
        line_border_mask = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE,
                                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), anchor=(-1, -1),
                                            iterations=1) - line_mask
        line_border_points = np.where(line_border_mask == 255)

        result_tmp = result.copy()
        # Iterate over points, fill each point with nearest fill's id.
        for i, _ in enumerate(line_border_points[0]):
            x, y = line_border_points[1][i], line_border_points[0][i]

            if x - 1 > 0 and result[y][x - 1] != line_id:
                result_tmp[y][x] = result[y][x - 1]
                continue

            if x - 1 > 0 and y - 1 > 0 and result[y - 1][x - 1] != line_id:
                result_tmp[y][x] = result[y - 1][x - 1]
                continue

            if y - 1 > 0 and result[y - 1][x] != line_id:
                result_tmp[y][x] = result[y - 1][x]
                continue

            if y - 1 > 0 and x + 1 < w and result[y - 1][x + 1] != line_id:
                result_tmp[y][x] = result[y - 1][x + 1]
                continue

            if x + 1 < w and result[y][x + 1] != line_id:
                result_tmp[y][x] = result[y][x + 1]
                continue

            if x + 1 < w and y + 1 < h and result[y + 1][x + 1] != line_id:
                result_tmp[y][x] = result[y + 1][x + 1]
                continue

            if y + 1 < h and result[y + 1][x] != line_id:
                result_tmp[y][x] = result[y + 1][x]
                continue

            if y + 1 < h and x - 1 > 0 and result[y + 1][x - 1] != line_id:
                result_tmp[y][x] = result[y + 1][x - 1]
                continue

        result = result_tmp.copy()

    return result

class DanbooRegion():
    def __init__(self, model):
        self.model = model
    def go_vector(self, img):
        tensor = torch.from_numpy(img.astype(np.float32).transpose(2,0,1) / 255.).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(tensor)
        return 1. - pred.cpu().detach().numpy()[0].transpose(1,2,0)
    def go_flipped_vector(self, x):
        a = self.go_vector(x)
        b = np.fliplr(self.go_vector(np.fliplr(x)))
        c = np.flipud(self.go_vector(np.flipud(x)))
        d = np.flipud(np.fliplr(self.go_vector(np.flipud(np.fliplr(x)))))
        return (a + b + c + d) / 4.0
    def go_transposed_vector(self, x):
        a = self.go_flipped_vector(x)
        b = np.transpose(self.go_flipped_vector(np.transpose(x, [1, 0, 2])), [1, 0, 2])
        return (a + b) / 2.0
    
    def go_process(self, img):
        img = img.copy()
        H, W = img.shape[:2]
        padw = (512 - W % 512) % 512
        padh = (512 - H % 512) % 512
        img = cv2.copyMakeBorder(img, 0, padh, 0, padw, cv2.BORDER_REFLECT_101)
        img2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
        
        # linear weight
        Weight = (np.arange(256) + 1) / 512
        Weight = np.concatenate([Weight, Weight[::-1]])
        Weight = Weight[np.newaxis, :]
        WeightHorizontal = np.repeat(Weight, 512, axis=0)
        WeightVertical = cv2.rotate(WeightHorizontal, cv2.ROTATE_90_CLOCKWISE)
        Weight = np.minimum(WeightHorizontal, WeightVertical)
        Weight = Weight[:,:,np.newaxis]
        w2 = img2.copy()
        w3 = img2.copy()
        w4 = img2.copy()
        
        for i in range(0, img.shape[0], 512):
            for j in range(0, img.shape[1], 512):
                img2[i:i+512, j:j+512, :] = self.go_transposed_vector(img[i:i+512,j:j+512,:])
                w2[i:i+512, j:j+512, :] = Weight

        img3 = img2.copy()
        img4 = img2.copy()
        for i in range(-256, img.shape[0] + 256, 512):
            for j in range(0, img.shape[1], 512):
                I = img[max(i,0):min(i+512,img.shape[0]), max(j,0):min(j+512,img.shape[1]), :]
                padU = 256 if i < 0 else 0
                padD = 256 if i + 256 == img.shape[0] else 0
                I = cv2.copyMakeBorder(I, padU, padD, 0, 0, cv2.BORDER_REFLECT_101)
                I = self.go_transposed_vector(I)
                ww = Weight.copy()
                if padD:
                    I = I[padU:-padD,:,:]
                    ww = ww[padU:-padD,:,:]
                else:
                    I = I[padU:,:,:]
                    ww = ww[padU:,:,:]
                img3[max(i,0):min(i+512,img.shape[0]), max(j,0):min(j+512,img.shape[1]), :] = I
                w3[max(i,0):min(i+512,img.shape[0]), max(j,0):min(j+512,img.shape[1]), :] = ww
                
        for i in range(0, img.shape[0], 512):
            for j in range(-256, img.shape[1] + 256, 512):
                I = img[max(i,0):min(i+512,img.shape[0]), max(j,0):min(j+512,img.shape[1]), :]
                padL = 256 if j < 0 else 0
                padR = 256 if j + 256 == img.shape[1] else 0
                I = cv2.copyMakeBorder(I, 0, 0, padL, padR, cv2.BORDER_REFLECT_101)
                I = self.go_transposed_vector(I)
                ww = Weight.copy()
                if padR:
                    I = I[:,padL:-padR,:]
                    ww = ww[:,padL:-padR,:]
                else:
                    I = I[:,padL:,:]
                    ww = ww[:,padL:,:]
                img4[max(i,0):min(i+512,img.shape[0]), max(j,0):min(j+512,img.shape[1]), :] = I
                w4[max(i,0):min(i+512,img.shape[0]), max(j,0):min(j+512,img.shape[1]), :] = ww        
        
        return ((img2*w2+img3*w3+img4*w4)/(w2+w3+w4))[:H,:W,0]
    
    def __call__(self, x):
        raw_img = (x.copy()*255).clip(0,255).astype(np.uint8)
        height = self.go_process(raw_img) * 255
        height += (height - cv2.GaussianBlur(height, (0,0), 3.0)) * 10.0
        marker = height.clip(0, 255).astype(np.uint8)
        marker[marker>135]=255
        marker[marker<255]=0
        fills = get_fill(marker / 255)
        for fill in fills:
            if fill[0].shape[0] < 64:
                marker[fill] = 0
        filt = np.array([
            [0,1,0],
            [1,1,1],
            [0,1,0]
        ], dtype=np.uint8)
        big_marker = cv2.erode(marker, filt, iterations=5)
        fills = get_fill(big_marker / 255)
        for fill in fills:
            if fill[0].shape[0] < 64:
                big_marker[fill] = 0
        big_marker = cv2.dilate(big_marker, filt, iterations=5)
        small_marker = marker.copy()
        small_marker[big_marker > 127] = 0
        fin_labels, _ = label(big_marker / 255)
        fin_labels = up_fill(get_fill(small_marker), fin_labels)
        water = cv2.watershed(raw_img, fin_labels.astype(np.int32)) + 1
        water = thinning(water)
        all_region_indices = find_all(water)
        return all_region_indices
    
def DanbooRegionLoadModel(model_path):
    model = diff_net_torch()
    vt_state_dict = model.state_dict()
    with h5py.File(model_path, "r") as F:
        F = F['model_weights']
        for keras_layer in MAPPING:
            Weight = F[keras_layer]['generator'][keras_layer]['kernel:0'][()]
            Bias = F[keras_layer]['generator'][keras_layer]['bias:0'][()]
            vt_state_dict['.'.join(MAPPING[keras_layer]) + '.weight'] = torch.from_numpy(Weight.transpose(3,2,0,1))
            vt_state_dict['.'.join(MAPPING[keras_layer]) + '.bias'] = torch.from_numpy(Bias)
    model.load_state_dict(vt_state_dict)
    model.eval()
    return model
def DanbooRegionGetRegion(img, model, rand_colour = False):
    rand_col = lambda : np.array([random.random() for _ in range(3)])
    
    DR = DanbooRegion(model)
    regions = DR(img)
    img2 = img.copy()

    for reg in regions:
        if len(reg[0]) == 0:
            continue
        col = rand_col() if rand_colour != 0 else np.average(img[reg], axis=0)
        img2[reg] = col
    return img2

if __name__ == '__main__':
    model = DanbooRegionLoadModel('../../../models/extra/DanbooRegion2020UNet.net')
    img = cv2.imread('00009-1104080113.png', cv2.IMREAD_COLOR)
    img2 = cv2.imread('06.jpg', cv2.IMREAD_GRAYSCALE)
    hh, ww = img2.shape[:2]
    def resize_crop(hh, ww, h2, w2):
        ratio_h = hh / h2
        ratio_w = ww / w2
        if ratio_h > ratio_w:
            w3 = ww
            h3 = int(h2 * ratio_w)
        else:
            h3 = hh
            w3 = int(w2 * ratio_h)
        return h3, w3
    h2, w2 = resize_crop(hh, ww, 1152, 896)
    
    I = img.astype(np.float32) / 255.
    R = DanbooRegionGetRegion(I, model).clip(0, 1)
    
    iMult = (I / (R + 1e-6)).clip(0, 1)
    iScr = 1 - ((1 - I) / (1 - R * iMult + 1e-6)).clip(0,1)
    
    R = cv2.resize(R, (w2,h2))
    iMult = cv2.resize(iMult, (w2,h2))
    iScr = cv2.resize(iScr, (w2,h2))
    
    cv2.imwrite('iR.png', (R*255).astype(np.uint8))
    cv2.imwrite('iMult.png', (iMult*255).astype(np.uint8))
    cv2.imwrite('iScr.png', (iScr*255).astype(np.uint8))
    
    
    #img2 = (DanbooRegionGetRegion(S, model) * 255).clip(0,255).astype(np.uint8)
    #cv2.imshow('1', img)
    #cv2.imshow('2', img2)
    #cv2.imwrite("o2.png", img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()