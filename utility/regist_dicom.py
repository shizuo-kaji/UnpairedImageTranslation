#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import random

import numpy as np
import sys
import os
import glob
import pydicom as dicom
from PIL import Image
from skimage.transform import rescale
from skimage import measure,filters

def std_image(img):
    mean = np.mean(img)
    std = np.std(img)
    return( (img-mean)/std )

# pattern matching by euclidean norm
def find_translation_ref(img,pattern):
    min_res = float('inf')
    w = args.window_X
    v = args.window_Y
    for i in range(-args.range_X,args.range_X+1):
        for j in range(-args.range_Y,args.range_Y+1):
            diff = img[args.ystart+j:args.ystart+j+v,args.xstart+i:args.xstart+i+w]-pattern
            res = np.linalg.norm(diff)
            if res<min_res:
                x=i
                y=j
                min_res=res
    return x,y

# barycentre
def find_translation_centre(img, threshold):
    h,w = img.shape
    Y,X = np.mgrid[0:h,0:w]
    cx,cy = np.mean(X[img>threshold]),np.mean(Y[img>threshold])
    return( int(w/2-cx), int(h/2-cy))
#    bg = measure.label(img < threshold)
#    for i in range(bg.max()+1):

# translating image
def translate(img, x, y):
    height, width = img.shape
    p = abs(x)
    q = abs(y)
    nimg = np.pad(img,[(q,q),(p,p)],"edge")
    return(nimg[q-y:q-y+height,p-x:p-x+width])

# load and scale DICOM
def load_dicom(fn, pad=0, forceSpacing=None, scaling=None, cropw=None, croph=None, random_translate=0):
    ref_dicom = dicom.dcmread(fn, force=True)
    ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    dt=ref_dicom.pixel_array.dtype
    img = ref_dicom.pixel_array.astype(np.float32)
    if forceSpacing is not None:
        scaling = float(ref_dicom.PixelSpacing[0])/forceSpacing
    if scaling is not None:
        img = rescale(img,scaling,mode="reflect",preserve_range=True)
    if random_translate>0:
        x = random.randint(-random_translate,random_translate)
        y = random.randint(-random_translate,random_translate)
        img = translate(img, x, y)
    # padding
    if pad>0:
        img = np.pad(img,((pad,pad),(pad,pad)),constant_values=img[0,0])        
#        img = np.pad(img,((pad,pad),(pad,pad)),'edge')        
    # centre crop
    if cropw is not None:
        left = img.shape[1]//2 - cropw//2
        top = img.shape[0]//2 - croph//2
        img = img[top:top+croph,left:left+cropw]
#    print(img.shape)
    return(img.astype(dt))

#fix = np.clip(fix,-1024,1000)+1024  ## correction for plan 

parser = argparse.ArgumentParser(description='Simple image registration by translation')
parser.add_argument('--fixed_img', '-f', help='fixed reference image')
parser.add_argument('--mask_img', '-m', help='mask image')
parser.add_argument('--window_X', '-wx', type=int, default=400)
parser.add_argument('--window_Y', '-wy', type=int, default=300)
parser.add_argument('--ystart', '-y', type=int, default=100)
parser.add_argument('--xstart', '-x', type=int, default=40)
parser.add_argument('--range_X', '-rx', type=int, default=40)
parser.add_argument('--range_Y', '-ry', type=int, default=15)
parser.add_argument('--translate', '-tr', type=int, nargs=2, default=None)
parser.add_argument('--threshold', '-th', type=float, default=100)
parser.add_argument('--root', '-R', default='./target/', help='Root directory path of image files')
parser.add_argument('--output', '-o', default='./out/',help='output directory')
parser.add_argument('--scale', '-s', type=float, default=None)
parser.add_argument('--forceSpacing', '-fs', type=float, default=None, help='scale dicom to match the specified spacing (pixel size)')
parser.add_argument('--padding', '-p', type=int, default=0)
parser.add_argument('--random_translate', '-rt', default=0, type=int, help='jitter input images by random translation (in pixel)')
parser.add_argument('--crop_width', '-cw', type=int, help='this value may have to be divisible by a large power of two (if you encounter errors)')
parser.add_argument('--crop_height', '-ch', type=int, help='this value may have to be divisible by a large power of two (if you encounter errors)')
parser.add_argument('--anonimise', action="store_true", help='strip personal information from saved DICOM')
args = parser.parse_args()

# load reference image
if args.fixed_img is not None:
    print("Fixed reference image: {}".format(args.fixed_img))
    fix = load_dicom(args.fixed_img, args.padding, args.forceSpacing, args.scale, args.crop_width, args.crop_height)

if args.mask_img is not None:
    print("mask image: {}".format(args.mask_img))
    mask = load_dicom(args.mask_img, args.padding, args.forceSpacing, args.scale, args.crop_width, args.crop_height)>0
    args.crop_width = mask.shape[1]
    args.crop_height = mask.shape[0]


dns = [args.root]
for root_dir in [args.root, os.path.join(args.root,"trainA"),os.path.join(args.root,"trainB"),os.path.join(args.root,"testA"),os.path.join(args.root,"testB")]:
    for root, dirs, files in os.walk(root_dir):
        for dirname in dirs:
            dns.append(os.path.join(root, dirname))
            
for dir in dns:
    print("\n Processing {}".format(dir))
    sum_x = 0
    sum_y = 0
    cnt = 0
    if args.translate is None:  # determine the average translation vector for each volume
        for fn in glob.glob(os.path.join(dir,"*.dcm")):
            move = load_dicom(fn, args.padding, args.forceSpacing, args.scale, args.crop_width, args.crop_height)
            # matching pattern
            if args.fixed_img:
                pattern = move[args.ystart:args.ystart+args.window_Y,args.xstart:args.xstart+args.window_X]
                x,y = find_translation_ref(fix,pattern)
            else:
                x,y = find_translation_centre(move,args.threshold)
            print("file {}, x {}, y {}".format(fn,x,y))
            sum_x += x
            sum_y += y
            cnt += 1
        x = sum_x // max(cnt,1)
        y = sum_y // max(cnt,1)
        print("\n mean x:{} y:{}\n ".format(x,y))
    else:
        x,y = args.translate

    # save
    os.makedirs(dir.replace(args.root,args.output,1), exist_ok=True)
    for fn in glob.glob(os.path.join(dir,"*.dcm")):
        #print("Writing: {}".format(fn))
        move = load_dicom(fn, args.padding, args.forceSpacing, args.scale, args.crop_width, args.crop_height, args.random_translate)
        move = translate(move, x, y)
        if args.mask_img:
            move[mask==0]=0
        # write back image
        ds = dicom.dcmread(fn, force=True)
#        ds.PixelData = move.tostring()
        ds.PixelData = move.tobytes()
    # update the information regarding the shape of the data array
        ds.Rows, ds.Columns = move.shape
        if args.forceSpacing is not None:
            ds.PixelSpacing[0] = args.forceSpacing
            ds.PixelSpacing[1] = args.forceSpacing
        if args.anonimise:
            ds.remove_private_tags()
            for tag in ds.keys():
                print(ds[tag])
                if 'Name' in ds[tag].name:
                    ds[tag].value = '00'
        ds.save_as(fn.replace(args.root,args.output,1))
