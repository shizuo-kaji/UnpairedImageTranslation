import os
import pydicom as dicom
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
from skimage.transform import rescale
from scipy.misc import imresize
from chainercv.transforms import random_crop,center_crop

## load images everytime from disk: slower but low memory usage
class DatasetOutMem(dataset_mixin.DatasetMixin):
    def __init__(self, path, baseA, rangeA, slice_range=None, crop=(256,256), random=0, forceSpacing=0.7634, imgtype="dcm", dtype=np.float32):
        self.path = path
        self.base = baseA
        self.range = rangeA
        self.ids = []
        self.random = random
        self.crop = crop
        self.ch = 1
        self.forceSpacing = forceSpacing
        self.dtype = dtype
        self.imgtype=imgtype

        print("Load Dataset from disk: {}".format(path))
        for file in glob.glob(os.path.join(self.path,"**/*.{}".format(imgtype)), recursive=True):
            fn, ext = os.path.splitext(file)
            if slice_range:
                # slice location from dicom header: slow
#                ref_dicom = dicom.read_file(file, force=True)
#                loc = float(ref_dicom.SliceLocation)
                # slice location from filename
                loc = int(fn[-3:])
                if slice_range[0] < loc < slice_range[1]:
                    self.ids.append(fn)
            else:
                self.ids.append(fn)
        print("Loaded: {} images".format(len(self.ids)))
        
    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return '{:s}.{}'.format(self.ids[i],self.imgtype)

    def img2var(self,img):
        # cut off mask [-1,1] or [0,1] output
        return(2*(np.clip(img,self.base,self.base+self.range)-self.base)/self.range-1.0)
#        return((np.clip(img,self.base,self.base+self.range)-self.base)/self.range)
    
    def var2img(self,var):
        return(0.5*(1.0+var)*self.range + self.base)
#        return(np.round(var*self.range + self.base))

    def overwrite(self,new,fn,salt):
        ref_dicom = dicom.read_file(fn, force=True)
        dt=ref_dicom.pixel_array.dtype
        img = np.full(ref_dicom.pixel_array.shape, self.base, dtype=np.float32)
        ch,cw = new.shape
        h,w = self.crop
        if np.min(img - ref_dicom.RescaleIntercept)<0:
            ref_dicom.RescaleIntercept = -1024
        img[np.newaxis,(ch-h)//2:(ch+h)//2,(cw-w)//2:(cw+w)//2] = new
        img -= ref_dicom.RescaleIntercept
        img = img.astype(dt)           
        print("min {}, max {}, intercept {}".format(np.min(img),np.max(img),ref_dicom.RescaleIntercept))
#            print(img.shape, img.dtype)
        ref_dicom.PixelData = img.tostring()
        ## UID should be changed for dcm's under different dir
        #                uid=dicom.UID.generate_uid()
        #                uid = dicom.UID.UID(uid.name[:-len(args.suffix)]+args.suffix)
        uid = ref_dicom[0x8,0x18].value.split(".")
        uid[-2] = salt
        uidn = ".".join(uid)
        uid = ".".join(uid[:-1])

#            ref_dicom[0x2,0x3].value=uidn  # Media SOP Instance UID                
        ref_dicom[0x8,0x18].value=uidn  #(0008, 0018) SOP Instance UID              
        ref_dicom[0x20,0xd].value=uid  #(0020, 000d) Study Instance UID       
        ref_dicom[0x20,0xe].value=uid  #(0020, 000e) Series Instance UID
        ref_dicom[0x20,0x52].value=uid  # Frame of Reference UID
        return(ref_dicom)

    def get_example(self, i):
        ref_dicom = dicom.read_file(self.get_img_path(i), force=True)
#        print(ref_dicom)
        ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        img = ref_dicom.pixel_array.astype(self.dtype)+ref_dicom.RescaleIntercept
        if self.forceSpacing>0:
            scaling = float(ref_dicom.PixelSpacing[0])/self.forceSpacing
            img = rescale(img,scaling,mode="reflect",preserve_range=True)
#            img = imresize(img,(int(img.shape[0]*self.scale), int(img.shape[1]**self.scale)), interp='bicubic')            
        img = self.img2var(img)
        img = img[np.newaxis,:,:]
        h,w = self.crop
        img = center_crop(img,(h+self.random, w+self.random))
        img = random_crop(img,self.crop).astype(self.dtype)
        return img
