#!/usr/bin/env python
#############################
##
## Image converter by learned models
##
#############################

import warnings
warnings.filterwarnings("ignore")

import argparse
import os,glob
import cv2
from datetime import datetime as dt
import time
import numpy as np
import net
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, Variable, cuda
from chainercv.utils import write_image
from chainercv.transforms import resize
from chainerui.utils import save_args
from arguments import arguments 
from consts import dtypes

def gradimg(img):
    grad = xp.tile(xp.asarray([[[[1,0,-1],[2,0,-2],[1,0,-1]]]],dtype=img.dtype),(img.array.shape[1],1,1))
    dx = F.convolution_2d(img,grad)
    dy = F.convolution_2d(img,xp.transpose(grad,(0,1,3,2)))
    return(F.sqrt(dx**2+dy**2))

def heatmap(heat,src):  ## heat [0,1], src [-1,1] grey
#    h = cv2.normalize(heat[0], h, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    h = np.uint8(np.clip(heat,0,1)*255)
    h = cv2.resize(h, (src.shape[2],src.shape[1]))
    h = cv2.applyColorMap(np.uint8(h), cv2.COLORMAP_JET)
    h = np.transpose(127.5*(src+1),(1,2,0)) + h
    h = np.uint8(255 * h / h.max())
    return(h)

if __name__ == '__main__':
    args = arguments()
    args.random_translate = 0 ## necessary to infer crop size
    outdir = os.path.join(args.out, dt.now().strftime('out_%m%d_%H%M'))

    if args.gpu[0] >= 0:
        cuda.get_device_from_id(args.gpu[0]).use()
        print('use gpu {}'.format(args.gpu))

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]

    # infer model names
    if not args.load_models:
        root=os.path.dirname(args.argfile)
        args.load_models=os.path.join(root,'enc_x{}.npz'.format(args.epoch))
                    
    ## load images
    if args.imgtype=="dcm":
        from dataset_dicom import Dataset as Dataset
    else:
        from dataset_jpg import DatasetOutMem as Dataset   

    dataset = Dataset(path=args.root, args=args, base=args.HU_baseA, rang=args.HU_rangeA)
    if args.ch != dataset.ch:
        print("number of input channels is different during training.")
    print("Input channels {}, Output channels {}".format(args.ch,args.out_ch))

    print(args)
    save_args(args, outdir)

#    iterator = chainer.iterators.MultiprocessIterator(dataset, args.batch_size, n_processes=3, repeat=False, shuffle=False)
    iterator = chainer.iterators.MultithreadIterator(dataset, args.batch_size, n_threads=3, repeat=False, shuffle=False)   ## best performance
#    iterator = chainer.iterators.SerialIterator(dataset, args.batch_size,repeat=False, shuffle=False)

    # shared pretrained layer
    if (args.gen_pretrained_encoder and args.gen_pretrained_lr_ratio == 0):
            if "resnet" in args.gen_pretrained_encoder:
                pretrained = L.ResNet50Layers()
                print("Pretrained ResNet model loaded.")
            else:
                pretrained = L.VGG16Layers()
                print("Pretrained VGG model loaded.")
            if args.gpu[0] >= 0:
                pretrained.to_gpu()

    ## load generator models
    if "gen" in args.load_models:
            gen = net.Generator(args)
            print('Loading {:s}..'.format(args.load_models))
            serializers.load_npz(args.load_models, gen)
            if args.gpu[0] >= 0:
                gen.to_gpu()
            xp = gen.xp
            is_AE = False
    elif "enc" in args.load_models:
        if (args.gen_pretrained_encoder and args.gen_pretrained_lr_ratio == 0) :
            enc = net.Encoder(args, pretrained)
        else:
            enc = net.Encoder(args)
        print('Loading {:s}..'.format(args.load_models))
        serializers.load_npz(args.load_models, enc)
        dec = net.Decoder(args)
        modelfn = args.load_models.replace('enc_x','dec_y')
        modelfn = modelfn.replace('enc_y','dec_x')
        print('Loading {:s}..'.format(modelfn))
        serializers.load_npz(modelfn, dec)
        if args.gpu[0] >= 0:
            enc.to_gpu()
            dec.to_gpu()
        xp = enc.xp
        is_AE = True
    else:
        gen = F.identity
        xp = np
        is_AE = False
        print("Identity...")

    ## prepare networks for analysis 
    if args.output_analysis:
        vgg = L.VGG16Layers()  # for perceptual loss
        vgg.to_gpu()
        if is_AE:
            enc_i = net.Encoder(args)
            dec_i = net.Decoder(args)
            dis = net.Discriminator(args)
            dis_i = net.Discriminator(args)
            if "enc_x" in args.load_models:
                models = {'enc_y': enc_i, 'dec_x': dec_i, 'dis_x': dis_i, 'dis_y': dis}
            else:
                models = {'enc_x': enc_i, 'dec_y': dec_i, 'dis_y': dis_i, 'dis_x': dis}
        else:
            gen_i = net.Generator(args)
            dis = net.Discriminator(args)
            dis_i = net.Discriminator(args)
            if "gen_f" in args.load_models:
                models = {'gen_i':gen_f,'dis_x':dis_i, 'dis_y':dis}
            else:
                models = {'gen_i':gen_g,'dis_y':dis_i, 'dis_x':dis}
        for e in models:
            path = args.load_models.replace('gen_g',e)
            path = path.replace('gen_f',e)
            path = path.replace('enc_x',e)
            path = path.replace('enc_y',e)
            if os.path.exists(path):
                print('Loading {:s}..'.format(path))
                serializers.load_npz(path, models[e])
                if args.gpu[0] >= 0:
                    models[e].to_gpu()
        

    ## start measuring timing
    os.makedirs(outdir, exist_ok=True)
    start = time.time()

    cnt = 0
    prevdir = "RaNdOmDir"
    for batch in iterator:
        imgs = Variable(chainer.dataset.concat_examples(batch, device=args.gpu[0]))
        with chainer.using_config('train', False),chainer.function.no_backprop_mode():
            if is_AE:
                out = dec(enc(imgs))
            else:
                out = gen(imgs)
        if args.output_analysis:
            img_disx = dis_i(imgs)
            img_disy = dis(out)
            # perceptual diff
            layer = args.perceptual_layer
            if imgs.shape[1] == 1:
                perc_diff = vgg(F.concat([imgs,imgs,imgs]), layers=[layer])[layer] - vgg(F.concat([out,out,out]), layers=[layer])[layer]
            else:
                perc_diff = vgg(imgs, layers=[layer])[layer] - vgg(out, layers=[layer])[layer]
            # tv
            dx = out[:, :, 1:, :-1] - out[:, :, :-1, :-1]
            dy = out[:, :, :-1, 1:] - out[:, :, :-1, :-1]
            tv = F.sqrt(dx**2 + dy**2 + 1e-8)
            ## cycle
            with chainer.using_config('train', False):
                if is_AE:
                    cycle = dec_i(enc_i(out))
                else:
                    cycle = gen_i(out)
            grad_diff = gradimg(out)-gradimg(imgs)
            air_diff = ((out.array <= args.air_threshold)^(imgs.array <= args.air_threshold)).astype(out.xp.float32) * (out-imgs)## XOR

            for var in [img_disx, img_disy, imgs, cycle, tv, perc_diff, grad_diff, air_diff]:
                var.to_cpu()
            img_disx = img_disx.array
            img_disy = img_disy.array
            imgs = imgs.array
            cycle = cycle.array
            tv = tv.array
            perc_diff = 0.05*np.abs(perc_diff.array)
            grad_diff = 0.5*np.abs(grad_diff.array)
            air_diff = 0.5*np.abs(air_diff.array)
            cycle_diff = 0.5*np.abs(cycle - imgs)


        ##
        out.to_cpu()
        out = out.array        
        ## output images
        for i in range(len(out)):
            fn = dataset.get_img_path(cnt)
            dname = os.path.dirname(fn)
            fn = os.path.basename(os.path.splitext(fn)[0])
            print("\nProcessing {}".format(fn))
            if args.imgtype=="dcm":
                new = 0.5*(1.0+out[i])*args.HU_rangeB + args.HU_baseB
            else:
                new = dataset.var2img(out[i]) 
            print("raw value: {} {}".format(np.min(out[i]),np.max(out[i])))
            #print(new.shape)

            # converted image
            if args.imgtype=="dcm":
                if  dname != prevdir:
                    salt = str(random.randint(1000, 999999))
                    prevdir = dname
                for j in range(args.num_slices):
                    ref_dicom = dataset.overwrite(new[j],cnt,salt)
                    path = os.path.join(outdir,'{:s}_out_{}.dcm'.format(fn,j))
                    ref_dicom.save_as(path)
            else:
                path = os.path.join(outdir,'{:s}_out.jpg'.format(fn))
                write_image(new, path)

            ## images for analysis
            if args.output_analysis:
                # original
                path = os.path.join(outdir,'{:s}_0orig.png'.format(fn))
                write_image( (imgs[i]*127.5+127.5).astype(np.uint8), path)
                # cycle
                path = os.path.join(outdir,'{:s}_1cycle.png'.format(fn))
                write_image( (cycle[i]*127.5+127.5).astype(np.uint8), path)
                # cycle difference
                path = os.path.join(outdir,'{:s}_2cycle_diff.png'.format(fn))
#                cycle_diff[i] = (cycle_diff[i]+1)/(imgs[i]+2)   # [0,2]/[1,3] = (0.0,1.5)
                print("cycle diff: {} {} {}".format(np.min(cycle_diff[i]),np.mean(cycle_diff[i]),np.max(cycle_diff[i])))
                cv2.imwrite(path, heatmap(cycle_diff[i,0],imgs[i]))
                # converted
                path = os.path.join(outdir,'{:s}_2out.png'.format(fn))
                write_image( (out[i]*127.5+127.5).astype(np.uint8), path)
                # perceptual difference
                path = os.path.join(outdir,'{:s}_3perc_diff.png'.format(fn))
                print("perc diff: {} {} {}".format(np.min(perc_diff[i]),np.mean(perc_diff[i]),np.max(perc_diff[i])))
                cv2.imwrite(path, heatmap(perc_diff[i,0],out[i]))
                # gradient difference
                path = os.path.join(outdir,'{:s}_4grad_diff.png'.format(fn))
                print("grad diff: {} {} {}".format(np.min(grad_diff[i]),np.mean(grad_diff[i]),np.max(grad_diff[i])))
                cv2.imwrite(path, heatmap(grad_diff[i,0],out[i]))
                # air region difference
                path = os.path.join(outdir,'{:s}_5air_diff.png'.format(fn))
                print("air diff: {} {} {}".format(np.min(air_diff[i]),np.mean(air_diff[i]),np.max(air_diff[i])))
                cv2.imwrite(path, heatmap(air_diff[i,0],out[i]))
                # discriminator for original
                if(img_disx[i].shape[0]==2):
                    wg=np.tanh(img_disx[i,1])+1
                    path = os.path.join(outdir,'{:s}_5disx_weight.png'.format(fn))
                    print("dis x_w: {} {} {}".format(np.min(wg),np.mean(wg),np.max(wg)))
                    cv2.imwrite(path, heatmap(wg,imgs[i]))
                    d = (1-img_disx[i,0])*wg
                else:
                    d = 1-img_disx[i,0]
                path = os.path.join(outdir,'{:s}_5disx.png'.format(fn))
                print("dis x: {} {} {}".format(np.min(d),np.mean(d),np.max(d)))
                cv2.imwrite(path, heatmap(d,imgs[i]))
                # discriminator for converted
                if(img_disy[i].shape[0]==2):
                    wg=np.tanh(img_disy[i,1])+1
                    path = os.path.join(outdir,'{:s}_6disy_weight.png'.format(fn))
                    print("dis y_w: {} {} {}".format(np.min(wg),np.mean(wg),np.max(wg)))
                    cv2.imwrite(path, heatmap(wg,out[i]))
                    d = (1-img_disy[i,0])*wg
                else:
                    d = 1-img_disy[i,0]
                path = os.path.join(outdir,'{:s}_6disy.png'.format(fn))
                print("dis y: {} {} {}".format(np.min(d),np.mean(d),np.max(d)))
                cv2.imwrite(path, heatmap(d,out[i]))
                # total variation
                path = os.path.join(outdir,'{:s}_9tv.png'.format(fn))
                print("TV: {} {} {}".format(np.min(tv[i]),np.mean(tv[i]),np.max(tv[i])))
                cv2.imwrite(path, heatmap(tv[i,0],out[i]))

            cnt += 1
        ####

    elapsed_time = time.time() - start
    print ("{} images in {} sec".format(cnt,elapsed_time))
    print ("Output: {}".format(outdir))



