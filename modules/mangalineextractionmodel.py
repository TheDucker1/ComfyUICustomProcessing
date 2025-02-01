# get model here https://github.com/ljsabc/MangaLineExtraction_PyTorch

import torch
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)


class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)



class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(
                    nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        if self.process:
            y0 = self.model(x)
            return y0 + y
        else:
            return x + y

class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)

class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            l = None
            if i == 0: 
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MangaLineExtractionModel(nn.Module):

    def __init__(self):
        super(MangaLineExtractionModel, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y

#class MyDataset(Dataset):
#    def __init__(self, image_paths, transform=None):
#        self.image_paths = image_paths
#        self.transform = transform
#        
#    def get_class_label(self, image_name):
#        # your method here
#        head, tail = os.path.split(image_name)
#        #print(tail)
#        return tail
#        
#    def __getitem__(self, index):
#        image_path = self.image_paths[index]
#        x = Image.open(image_path)
#        y = self.get_class_label(image_path.split('/')[-1])
#        if self.transform is not None:
#            x = self.transform(x)
#        return x, y
#    
#    def __len__(self):
#        return len(self.image_paths)

#def loadImages(folder):
#    imgs = []
#    matches = []
#    for root, dirnames, filenames in os.walk(folder):
#        for filename in fnmatch.filter(filenames, '*'):
#            matches.append(os.path.join(root, filename))
#   
#    return matches

#if __name__ == "__main__":
#    model = res_skip()
#    model.load_state_dict(torch.load('erika.pth'))
#    is_cuda = torch.cuda.is_available()
#    if is_cuda:
#        model.cuda()
#    else:
#        model.cpu()
#    model.eval()
#    
#    filelists = loadImages(sys.argv[1])
#
#    with torch.no_grad():
#        for imname in filelists:
#            src = cv2.imread(imname,cv2.IMREAD_GRAYSCALE)
#            
#            rows = int(np.ceil(src.shape[0]/16))*16
#            cols = int(np.ceil(src.shape[1]/16))*16
#            
#            # manually construct a batch. You can change it based on your usecases. 
#            patch = np.ones((1,1,rows,cols),dtype="float32")
#            patch[0,0,0:src.shape[0],0:src.shape[1]] = src
#            
#            if is_cuda: 
#                tensor = torch.from_numpy(patch).cuda()
#            else:
#                tensor = torch.from_numpy(patch).cpu()
#            y = model(tensor)
#            print(imname, torch.max(y), torch.min(y))
#
#            yc = y.cpu().numpy()[0,0,:,:]
#            yc[yc>255] = 255
#            yc[yc<0] = 0
#
#            head, tail = os.path.split(imname)
#            cv2.imwrite(sys.argv[2]+"/"+tail.replace(".jpg",".png"),yc[0:src.shape[0],0:src.shape[1]])

def MangaLineExtract(img_batch, model):
    B, H, W, C = img_batch.shape
    if C == 3:
        img_batch = rgb_to_grayscale(img_batch.permute(0,3,1,2))
    else:
        img_batch = img_batch.permute(0,3,1,2)
    padW = (16 - W%16)%16
    padH = (16 - H%16)%16
    img_batch = F.pad(img_batch, (0, padW, 0, padH), 'reflect')
    img_batch *= 255
    with torch.no_grad():
        img_batch = model(img_batch)
    img_batch = img_batch.clip(0, 255)
    img_batch /= 255
    img_batch = img_batch[:,:,:H,:W].permute(0,2,3,1)
    if C == 3:
        img_batch = img_batch.repeat(1,1,1,3)
    return img_batch

def MangaLineModelLoad(model_path):
    model = MangaLineExtractionModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model