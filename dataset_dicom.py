import os,re
import pydicom as dicom
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
import PIL
from skimage.transform import rescale
from chainercv.transforms import random_crop,center_crop,resize,rotate
from consts import dtypes

class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, args, base, rang, random_tr=0, random_rot=0, random_scale=0):
        self.path = path
        self.base = base
        self.range = rang
        self.random_tr = random_tr
        self.random_rot = random_rot
        self.random_scale = random_scale
        self.ch = args.num_slices
        self.forceSpacing = args.forceSpacing
        self.dtype = dtypes[args.dtype]
        self.imgtype=args.imgtype
        self.dcms = []
        self.names = []
        self.idx = []
        if args.crop_height and args.crop_width:
            self.crop = (args.crop_height,args.crop_width)
        else:
            self.crop=( 16*((512-2*self.random_tr)//16), 16*((512-2*self.random_tr)//16) )
        num = lambda val : int(re.sub("\\D", "", val+"0"))

        print("Loading Dataset from: {}".format(path))
        dirlist = [path]
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                dirlist.append(os.path.join(path,f))
        j = 0  # dir index
        for dirname in sorted(dirlist):
            files = [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname), key=num) if fname.endswith(args.imgtype)]
            slices = []
            filenames = []
            loc = []
            for f in files:
                ds = dicom.dcmread(f, force=True)
                ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
                # sort slices according to SliceLocation header
                if hasattr(ds, 'ImagePositionPatient') and (args.slice_range is not None): # Thanks to johnrickman for letting me know to use this DICOM entry
#                if hasattr(ds, 'SliceLocation'):
                    z = float(ds.ImagePositionPatient[2])
                    if (args.slice_range[0] < z < args.slice_range[1]):
                        slices.append(ds)
                        filenames.append(f)
                        loc.append(z)   # sort by z-coord
                else:
                    slices.append(ds)
                    filenames.append(f)
                    loc.append(f)  # sort by filename
            s = sorted(range(len(slices)), key=lambda k: loc[k])

            # if the current dir contains at least one slice
            if len(s)>0:
                vollist = []
                for i in s:
                    sl = slices[i].pixel_array.astype(self.dtype)+slices[i].RescaleIntercept
                    if self.forceSpacing>0:
                        scaling = float(slices[i].PixelSpacing[0])/self.forceSpacing
                        sl = rescale(sl,scaling,mode="reflect",preserve_range=True)
                    vollist.append(sl)
                volume = self.img2var(np.stack(vollist))   # shape = (z,x,y)
                print("Loaded volume {} of size {}".format(dirname,volume.shape))
                if volume.shape[1]<self.crop[0]+2*self.random_tr or volume.shape[2] < self.crop[1]+2*self.random_tr:
                    p = max(self.crop[0]+2*self.random_tr-volume.shape[1],self.crop[1]+2*self.random_tr-volume.shape[2])
                    volume = np.pad(volume,((0,0),(p,p),(p,p)),'edge')
                volume = center_crop(volume,(self.crop[0]+2*self.random_tr, self.crop[1]+2*self.random_tr))
                self.dcms.append(volume)
                self.names.append( [filenames[i] for i in s] )
                self.idx.extend([(j,k) for k in range((self.ch-1)//2,len(slices)-self.ch//2)])
                j = j + 1

        print("#dir {}, #file {}, #slices {}".format(len(dirlist),len(self.idx),sum([len(fd) for fd in self.names])))

    def __len__(self):
        return len(self.idx)

    def get_img_path(self, i):
        j,k=self.idx[i]
        return self.names[j][k]

    def img2var(self,img):
        # output clipped and scaled to [-1,1]
        return(2*(np.clip(img,self.base,self.base+self.range)-self.base)/self.range-1.0)
    
    def var2img(self,var):
        # inverse of img2var
        return(0.5*(1.0+var)*self.range + self.base)

    def overwrite(self,new,i,salt):
        ref_dicom = dicom.dcmread(self.get_img_path(i), force=True)
        ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ExplicitVRLittleEndian #dicom.uid.ImplicitVRLittleEndian
        #ref_dicom.is_little_endian = True
        #ref_dicom.is_implicit_VR = False
        dt=ref_dicom.pixel_array.dtype
        img = np.full(ref_dicom.pixel_array.shape, self.base, dtype=np.float32)
        ch,cw = img.shape
        h,w = new.shape
        print(img.shape)
#        if np.min(img - ref_dicom.RescaleIntercept)<0:
#            ref_dicom.RescaleIntercept = -1024
        img[(ch-h)//2:(ch+h)//2,(cw-w)//2:(cw+w)//2] = new
        img -= ref_dicom.RescaleIntercept
        img = img.astype(dt)           
        print("min {}, max {}, intercept {}".format(np.min(img),np.max(img),ref_dicom.RescaleIntercept))
#            print(img.shape, img.dtype)
        ref_dicom.PixelData = img.tostring()
        ## UID should be changed for dcm's under different dir
        #                uid=dicom.UID.generate_uid()
        #                uid = dicom.UID.UID(uid.name[:-len(args.suffix)]+args.suffix)
#        uid = ref_dicom[0x8,0x18].value.split(".")
        uid = ref_dicom[0x20,0x52].value.split(".")  # Frame of Reference UID
        uid[-1] = salt
        uidn = ".".join(uid)
#        uid = ".".join(uid[:-1])
#        ref_dicom[0x2,0x3].value=uidn  # Media SOP Instance UID                
        # ref_dicom[0x8,0x18].value=uidn  #(0008, 0018) SOP Instance UID              
        # ref_dicom[0x20,0xd].value=uid  #(0020, 000d) Study Instance UID       
        # ref_dicom[0x20,0xe].value=uid  #(0020, 000e) Series Instance UID
        ref_dicom[0x20,0x52].value=uidn  # Frame of Reference UID
        return(ref_dicom)

    def get_example(self, i):
        j,k = self.idx[i]
        img = self.dcms[j][(k-(self.ch-1)//2):(k+(self.ch+1)//2)]
        ## TODO: multi channel
        if self.random_scale>0:
            r = np.random.uniform(1-self.random_scale,1+self.random_scale)
            img = resize(img, (int(img.shape[1]*r),int(img.shape[2]*r)), interpolation=PIL.Image.LANCZOS)
        if self.random_rot>0:
            img = rotate(img, np.random.uniform(-self.random_rot,self.random_rot),expand=False, fill=-1)
        if img.shape[1]<self.crop[0]+2*self.random_tr or img.shape[2] < self.crop[1]+2*self.random_tr:
            p = max(self.crop[0]+2*self.random_tr-img.shape[1],self.crop[1]+2*self.random_tr-img.shape[2])
            img = np.pad(img,((0,0),(p,p),(p,p)),'edge')
        return random_crop(img,self.crop).astype(self.dtype)
