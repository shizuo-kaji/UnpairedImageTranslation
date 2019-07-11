import os
import pydicom as dicom
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
from skimage.transform import rescale
from scipy.misc import imresize
from chainercv.transforms import random_crop,center_crop
from consts import dtypes

class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, args, random=0, forceSpacing=0):
        self.path = path
        self.base = args.HU_base
        self.range = args.HU_range
        self.random = random
        self.crop = (args.crop_height,args.crop_width)
        self.ch = args.num_slices
        self.forceSpacing = forceSpacing
        self.dtype = dtypes[args.dtype]
        self.imgtype=args.imgtype
        self.dcms = []
        self.names = []
        self.idx = [] 

        if not args.crop_height:
            self.crop = (384,480)  ## default for the CBCT dataset
        print("Load Dataset from disk: {}".format(path))
        dirlist = [path]
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                dirlist.append(os.path.join(path,f))
        skipcount = 0
        j = 0
        for dirname in sorted(dirlist):
            files = [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname)) if fname.endswith(args.imgtype)]
            slices = []
            filenames = []
            loc = []
            for f in files:
                ds = dicom.dcmread(f, force=True)
                ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
                if hasattr(ds, 'SliceLocation'):
                    # loc.append(float(ds.SliceLocation))
                    loc.append(int(f[-7:-4])) # slice location from filename
                    if args.slice_range:
                        if args.slice_range[0] < loc < args.slice_range[1]:
                            slices.append(ds)
                            filenames.append(f)
                    else:
                        slices.append(ds)
                        filenames.append(f)
                else:
                    skipcount = skipcount + 1
            s = sorted(range(len(slices)), key=lambda k: loc[k])
            if len(s)>0:
                volume = self.img2var(np.stack([slices[i].pixel_array.astype(self.dtype)+slices[i].RescaleIntercept for i in s]))
                if self.forceSpacing>0:
                    scaling = float(slices[0].PixelSpacing[0])/self.forceSpacing
                    img = rescale(volume,scaling,mode="reflect",preserve_range=True)
        #            img = imresize(img,(int(img.shape[0]*self.scale), int(img.shape[1]**self.scale)), interp='bicubic')            
                self.dcms.append(volume)
                self.names.append( [filenames[i] for i in s] )
                self.idx.extend([(j,k) for k in range((self.ch-1)//2,len(slices)-self.ch//2)])
                j = j + 1

        print("#dir {}, #file {}, #skipped {}".format(len(dirlist),len(self.idx),skipcount))
        
    def __len__(self):
        return len(self.idx)

    def get_img_path(self, i):
        j,k=self.idx[i]
        return self.names[j][k]

    def img2var(self,img):
        # cut off mask [-1,1] or [0,1] output
        return(2*(np.clip(img,self.base,self.base+self.range)-self.base)/self.range-1.0)
#        return((np.clip(img,self.base,self.base+self.range)-self.base)/self.range)
    
    def var2img(self,var):
        return(0.5*(1.0+var)*self.range + self.base)
#        return(np.round(var*self.range + self.base))

    def overwrite(self,new,i,salt):
        ref_dicom = dicom.dcmread(self.get_img_path(i), force=True)
        ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
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
        j,k = self.idx[i]
        img = self.dcms[j][(k-(self.ch-1)//2):(k+(self.ch+1)//2)]
        h,w = self.crop
        return random_crop(center_crop(img,(h+self.random, w+self.random)),self.crop).astype(self.dtype)
