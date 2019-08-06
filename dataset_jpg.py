import os
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
from PIL import Image

from chainercv.transforms import random_crop,center_crop,random_flip
from chainercv.transforms import resize
from chainercv.utils import read_image

from consts import dtypes

## load images everytime from disk: slower but low memory usage
class DatasetOutMem(dataset_mixin.DatasetMixin):
    def __init__(self, path, args, random=0, forceSpacing=0):
        self.path = path
        self.names = []
        self.random = random
        self.color=True
        self.ch = 3 if self.color else 1
        self.imgtype=args.imgtype
        self.dtype = dtypes[args.dtype]
        for fn in glob.glob(os.path.join(self.path,"**/*.{}".format(self.imgtype)), recursive=True):
            self.names.append(fn)
        if not args.crop_height or not args.crop_width:
            img = read_image(self.names[0])
            self.crop = ( 16*((img.shape[1]-args.random_translate)//16), 16*((img.shape[2]-args.random_translate)//16) )
        else:
            self.crop = (args.crop_height,args.crop_width)
        self.names = sorted(self.names)
        print("Cropped to: ",self.crop)
        print("Loaded: {} images".format(len(self.names)))

    def __len__(self):
        return len(self.names)

    def get_img_path(self, i):
        return(self.names[i])

    def var2img(self,var):
        return(0.5*(1.0+var)*255)

    def get_example(self, i):
        img = read_image(self.get_img_path(i),color=self.color)
        img = img * 2 / 255.0 - 1.0  # [-1, 1)
#        img = resize(img, (self.resize_to, self.resize_to))
        img = random_crop(center_crop(img, (self.crop[0]+self.random,self.crop[1]+self.random)),self.crop)
        if self.random:
            img = random_flip(img, x_random=True)
        return img.astype(self.dtype)

    def mask(self,fn):
        img = Image.open(fn)
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
        return img
