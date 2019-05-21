#!/usr/bin/env python
#############################
##
## Image converter by learned models
##
#############################

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import glob
import json
import codecs
from datetime import datetime as dt
import time
import chainer.cuda
from chainer import serializers, Variable
import numpy as np
import net
import random
import chainer.functions as F
from chainercv.utils import write_image
from chainercv.transforms import resize
from chainerui.utils import save_args
from arguments import arguments 
from consts import activation,dtypes

def gradimg(img):
    grad = xp.tile(xp.asarray([[[[1,0,-1],[2,0,-2],[1,0,-1]]]],dtype=img.dtype),(img.data.shape[1],1,1))
    dx = F.convolution_2d(img,grad)
    dy = F.convolution_2d(img,xp.transpose(grad,(0,1,3,2)))
    return(F.sqrt(dx**2+dy**2))

if __name__ == '__main__':
    args = arguments()
    args.suffix = "out"
    outdir = os.path.join(args.out, dt.now().strftime('out_%m%d_%H%M'))

    args.gpu = args.gpu[0]
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print('use gpu {}'.format(args.gpu))

    ## load arguments from "arg" file used in training
    if args.argfile:
        with open(args.argfile, 'r') as f:
            larg = json.load(f)
            root=os.path.dirname(args.argfile)
            for x in ['HU_base','HU_range','forceSpacing',
              'dis_norm','dis_activation','dis_chs','dis_ksize','dis_sample','dis_down',
              'gen_norm','gen_activation','gen_out_activation','gen_nblock','gen_chs','gen_sample','gen_down','gen_up','gen_ksize','unet',
              'conditional_discriminator','gen_fc','gen_fc_activation','spconv','eqconv','wgan','dtype']:
                if x in larg:
                    setattr(args, x, larg[x])
            if not args.load_models:
                if larg["epoch"]:
                    args.load_models=os.path.join(root,'gen_g{}.npz'.format(larg["epoch"]))
                else:
                    args.load_models=os.path.join(root,'gen_g{}.npz'.format(larg["lrdecay_start"]+larg["lrdecay_period"]))
                    
    args.random_translate = 0
    save_args(args, outdir)
    args.dtype = dtypes[args.dtype]
    args.dis_activation = activation[args.dis_activation]
    args.gen_activation = activation[args.gen_activation]
    args.gen_out_activation = activation[args.gen_out_activation]
    args.gen_fc_activation = activation[args.gen_fc_activation]
    print(args)
    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = args.dtype

    ## load images
    if args.imgtype=="dcm":
        from dataset_dicom import DatasetOutMem as Dataset 
    else:
        from dataset_jpg import DatasetOutMem as Dataset   

    dataset = Dataset(path=args.root, baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=0, forceSpacing=0, imgtype=args.imgtype, dtype=args.dtype)
    args.ch = dataset.ch
#    iterator = chainer.iterators.MultiprocessIterator(dataset, args.batch_size, n_processes=3, repeat=False, shuffle=False)
    iterator = chainer.iterators.MultithreadIterator(dataset, args.batch_size, n_threads=3, repeat=False, shuffle=False)   ## best performance
#    iterator = chainer.iterators.SerialIterator(dataset, args.batch_size,repeat=False, shuffle=False)

    ## load generator models
    if "gen" in args.load_models:
            gen = net.Generator(args)
            print('Loading {:s}..'.format(args.load_models))
            serializers.load_npz(args.load_models, gen)
            if args.gpu >= 0:
                gen.to_gpu()
            xp = gen.xp
            is_AE = False
    elif "enc" in args.load_models:
        args.gen_nblock = args.gen_nblock // 2  # to match ordinary cycleGAN
        enc = net.Encoder(args)
        print('Loading {:s}..'.format(args.load_models))
        serializers.load_npz(args.load_models, enc)
        dec = net.Decoder(args)
        modelfn = args.load_models.replace('enc_x','dec_y')
        modelfn = modelfn.replace('enc_y','dec_x')
        print('Loading {:s}..'.format(modelfn))
        serializers.load_npz(modelfn, dec)
        if args.gpu >= 0:
            enc.to_gpu()
            dec.to_gpu()
        xp = enc.xp
        is_AE = True
    else:
        print("Specify a learnt model.")
        exit()        

    ## prepare networks for analysis 
    if args.output_analysis:
        if is_AE:
            enc_y = net.Encoder(args)
            dec_x = net.Decoder(args)
            dis_x = net.Discriminator(args)
            dis_y = net.Discriminator(args)
            models = {'enc_y': enc_y, 'dec_x': dec_x, 'dis_x': dis_x, 'dis_y': dis_y}
        else:
            gen_f = net.Generator(args)
            dis_y = net.Discriminator(args)
            dis_x = net.Discriminator(args)
            models = {'gen_f':gen_f, 'dis_x':dis_x, 'dis_y':dis_y}
        for e in models:
            path = args.load_models.replace('gen_g',e)
            path = path.replace('enc_x',e)
            if os.path.exists(path):
                print('Loading {:s}..'.format(path))
                serializers.load_npz(path, models[e])
                if args.gpu >= 0:
                    models[e].to_gpu()
        

    ## start measuring timing
    os.makedirs(outdir, exist_ok=True)
    start = time.time()

    cnt = 0
    prevdir = "RaNdOmDir"
    for batch in iterator:
        imgs = Variable(chainer.dataset.concat_examples(batch, device=args.gpu))
        with chainer.using_config('train', False),chainer.function.no_backprop_mode():
            if is_AE:
                out_v = dec(enc(imgs))
            else:
                out_v = gen(imgs)
        if args.output_analysis:
            img_disx = dis_x(imgs)
            ## gradcam for input
            img_disx.grad=xp.full(img_disx.data.shape, 1.0, dtype=imgs.dtype)
            dis_x.cleargrads()
            img_disx.backward(retain_grad=True)
            #
#            loss = F.sum((img_disx-xp.full(img_disx.data.shape, 1.0, dtype=img_disx.dtype))**2)
#            gd_x, = chainer.grad([loss],[imgs])
#            gd_x = np.clip(np.absolute(xp.asnumpy(gd_x.data)),0,10)
            ## gradcam for output
            img_disy = dis_y(out_v)
            img_disy.grad=xp.full(img_disy.data.shape, 1.0, dtype=imgs.dtype)
            dis_y.cleargrads()
            img_disy.backward(retain_grad=True)

            ## cycle
            with chainer.using_config('train', False):
                if is_AE:
                    cycle = dec_x(enc_y(out_v))
                else:
                    cycle = gen_f(out_v)
#            diff = cycle - imgs
            diff = gradimg(cycle)-gradimg(imgs)
            if args.gpu >= 0:
                weights_x = xp.asnumpy(xp.mean(imgs.grad, axis=(2, 3)))
                img_disx = np.abs(xp.asnumpy(img_disx.data)-1)
                weights_y = xp.asnumpy(xp.mean(out_v.grad, axis=(2, 3)))
                img_disy = np.abs(xp.asnumpy(img_disy.data)-1)
                cycle_diff = xp.asnumpy(diff.data)
                cycle = xp.asnumpy(cycle.data)
            else:
                weights_x = xp.mean(imgs.grad, axis=(2, 3))
                img_disx = np.abs(img_disx.data-1)
                weights_y = xp.mean(out_v.grad, axis=(2, 3))
                img_disy = np.abs(img_disy.data-1)
                cycle_diff = diff.data
                cycle = cycle.data

        if args.gpu >= 0:
            imgs = xp.asnumpy(imgs.data)
            out = xp.asnumpy(out_v.data)
        else:
            imgs = imgs.data
            out = out_v.data

        
        ## output images
        for i in range(len(out)):
            fn = dataset.get_img_path(cnt)
            print("\nProcessing {}".format(fn))
            new = dataset.var2img(out[i]) 
            print("raw value: {} {}".format(np.min(out[i]),np.max(out[i])))
            #print(new.shape)
            if len(new.shape)==3:
                cc,ch,cw = new.shape
            else:
                cc=1
                ch,cw = new.shape
            h,w = dataset.crop

            # converted image
            if args.imgtype=="dcm":
                if os.path.dirname(fn) != prevdir:
                    salt = str(random.randint(1000, 999999))
                    prevdir = os.path.dirname(fn)
                ref_dicom = dataset.overwrite(new[0],fn,salt)
                path = os.path.join(outdir,'{:s}_{}.dcm'.format(os.path.basename(fn),args.suffix))
                ref_dicom.save_as(path)
                ch,cw=ref_dicom.pixel_array.shape
            else:
                path = os.path.join(outdir,'{:s}_{}.jpg'.format(os.path.basename(fn),args.suffix))
                write_image(new, path)

            ## images for analysis
            if args.output_analysis:
                # original
                path = os.path.join(outdir,'{:s}_0org.jpg'.format(os.path.basename(fn)))
                write_image( (imgs[i]*127.5+127.5).astype(np.uint8), path)
                # converted
                path = os.path.join(outdir,'{:s}_3out.jpg'.format(os.path.basename(fn)))
                write_image( (out[i]*127.5+127.5).astype(np.uint8), path)
                # cycle
                path = os.path.join(outdir,'{:s}_1cycle.jpg'.format(os.path.basename(fn)))
                write_image( (cycle[i]*127.5+127.5).astype(np.uint8), path)
                # cycle grad image difference
                path = os.path.join(outdir,'{:s}_2cycle_diff.jpg'.format(os.path.basename(fn)))
#                cycle_diff[i] = (cycle_diff[i]+1)/(imgs[i]+2)   # [0,2]/[1,3] = (0.0,1.5)
                cycle_diff[i] = (cycle_diff[i]+2)/4
                print("cycle diff: {} {} {}".format(np.min(cycle_diff[i]),np.mean(cycle_diff[i]),np.max(cycle_diff[i])))
                write_image( (np.clip(cycle_diff[i],0,1)*255).astype(np.uint8), path) 
                # gradcam for dis_x
#                gd_x = np.clip(np.tensordot(weights_x[i], img_disx[i], axes=(0, 0))*100,0,1)
#                print("dis dx: {} {} {}".format(np.min(gd_x),np.mean(gd_x),np.max(gd_x)))
#                path = os.path.join(outdir,'{:s}_5disdx.jpg'.format(os.path.basename(fn)))
#                img = np.zeros((cc,ch,cw), dtype=np.uint8)
#                img[:,(ch-h)//2:(ch+h)//2,(cw-w)//2:(cw+w)//2] = 
#                write_image(resize(gd_x[np.newaxis,]*255,(h,w)), path)
                # discriminator for original
                path = os.path.join(outdir,'{:s}_4disx.jpg'.format(os.path.basename(fn)))
                print("dis x: {} {} {}".format(np.min(img_disx[i]),np.mean(img_disx[i]),np.max(img_disx[i])))
                write_image(resize(np.clip(img_disx[i],0,1)*255,(h,w)), path)
                # gradcam for dis_y
#                gd_y = np.clip(np.tensordot(weights_y[i], img_disy[i], axes=(0, 0))*30,0,1)
#                print("dis dy: {} {} {}".format(np.min(gd_y[i]),np.mean(gd_y[i]),np.max(gd_y[i])))
#                path = os.path.join(outdir,'{:s}_6disdy.jpg'.format(os.path.basename(fn)))
#                write_image(resize(gd_y[np.newaxis,]*255,(h,w)), path)
                # discriminator for converted
                path = os.path.join(outdir,'{:s}_7disy.jpg'.format(os.path.basename(fn)))
                print("dis y: {} {} {}".format(np.min(img_disy[i]),np.mean(img_disy[i]),np.max(img_disy[i])))
                write_image(resize(np.clip(img_disy[i],0,1)*255,(h,w)), path)

            cnt += 1
        ####

    elapsed_time = time.time() - start
    print ("{} images in {} sec".format(cnt,elapsed_time))



