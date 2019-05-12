import os
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
from PIL import Image

from chainercv.transforms import random_crop
from chainercv.transforms import random_flip
from chainercv.transforms import resize
from chainercv.utils import read_image


## load images everytime from disk: slower but low memory usage
class DatasetOutMem(dataset_mixin.DatasetMixin):
    def __init__(self, path, baseA, rangeA, slice_range=None, crop=(256,256), random=0, forceSpacing=0, imgtype="png", dtype=np.float32):
        self.path = path
        self.ids = []
        self.flip = random
        self.crop = crop
        self.color=True
        self.ch = 3 if self.color else 1
        self.imgtype=imgtype
        self.dtype = dtype
        for file in glob.glob(self.path+"/**/*.{}".format(imgtype), recursive=True):
            fn, ext = os.path.splitext(file)
            self.ids.append(fn)
        print("Loaded: {} images".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return '{:s}.{}'.format(self.ids[i],self.imgtype)

    def var2img(self,var):
        return(0.5*(1.0+var)*255)

    def get_example(self, i):
        img = read_image(self.get_img_path(i),color=self.color)
        img = img * 2 / 255.0 - 1.0  # [-1, 1)
#        img = resize(img, (self.resize_to, self.resize_to))
        img = random_crop(img, self.crop)
        if self.flip:
            img = random_flip(img, x_random=True)
        return img.astype(self.dtype)

## load dataset onto memory: faster
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, baseA, rangeA, crop=(256,256), random=0, scale=1.0):
        self.path = path
        self.ids = []
        self.imgs = []
        self.crop = crop
        self.flip = random
        self.color=False
        self.suffix="png"

        for file in glob.glob(self.path+"/**/*.{}".format(self.suffix), recursive=True):
            fn, ext = os.path.splitext(file)
            img = Image.open(file).resize((self.resize_to, self.resize_to))
            # mask
            if img.mode == 'RGBA':
                mask = (np.array(img.split()[-1]) > 0)
            else:
                mask = np.ones( (img.shape[1],img.shape[2]), dtype=bool)
            # convert to [C,H,W]
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 2:
                img = img[np.newaxis]
            else:
                img = img.transpose((2, 0, 1))[:3,:,:]
            img = img * mask
            img = img * 2 / 255.0 - 1.0  # [-1, 1)
#            img = img[:,self.cut:-self.cut,self.cut:-self.cut]
            self.ids.append(fn)
            self.imgs.append(img.copy())

    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return '{:s}.{}'.format(self.ids[i],self.suffix)

    def get_example(self, i):
        img = self.imgs[i]
        out = random_crop(img, (self.crop_to, self.crop_to))
        if self.flip:
            out = random_flip(out, x_random=True)
        return out
