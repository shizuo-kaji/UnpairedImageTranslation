#!/usr/bin/env python
import os
from datetime import datetime as dt

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import serializers, training
from chainer.training import extensions
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainer.dataset import convert
import chainer.functions as F

import net
from arguments import arguments 
import numpy as np

from updaterAE import Updater
from visualization import VisEvaluator
from consts import activation,dtypes

def main():
    args = arguments()
    out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M_AE'))
    print(args)
    print(out)
    save_args(args, out)
    args.dtype = dtypes[args.dtype]
    args.dis_activation = activation[args.dis_activation]
    args.gen_activation = activation[args.gen_activation]
    args.gen_out_activation = activation[args.gen_out_activation]
    args.gen_nblock = args.gen_nblock // 2  # to match ordinary cycleGAN

    if args.imgtype=="dcm":
        from dataset_dicom import DatasetOutMem as Dataset 
    else:
        from dataset_jpg import DatasetOutMem as Dataset   


    if not chainer.cuda.available:
        print("CUDA required")

    if len(args.gpu)==1 and args.gpu[0] >= 0:
        chainer.cuda.get_device_from_id(args.gpu[0]).use()

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = args.dtype
    chainer.print_runtime_info()
    # Turn off type check
#    chainer.config.type_check = False
#    print('Chainer version: ', chainer.__version__)
#    print('GPU availability:', chainer.cuda.available)
#    print('cuDNN availablility:', chainer.cuda.cudnn_enabled)

    ## dataset iterator
    print("Setting up data iterators...")
    train_A_dataset = Dataset(
        path=os.path.join(args.root, 'trainA'), baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width),random=args.random_translate, forceSpacing=0, imgtype=args.imgtype, dtype=args.dtype)
    train_B_dataset = Dataset(
        path=os.path.join(args.root, 'trainB'),  baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=args.random_translate, forceSpacing=args.forceSpacing, imgtype=args.imgtype, dtype=args.dtype)
    test_A_dataset = Dataset(
        path=os.path.join(args.root, 'testA'), baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=0, forceSpacing=0, imgtype=args.imgtype, dtype=args.dtype)
    test_B_dataset = Dataset(
        path=os.path.join(args.root, 'testB'),  baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=0, forceSpacing=args.forceSpacing, imgtype=args.imgtype, dtype=args.dtype)

    args.ch = train_A_dataset.ch
    test_A_iter = chainer.iterators.SerialIterator(test_A_dataset, args.nvis_A, shuffle=False)
    test_B_iter = chainer.iterators.SerialIterator(test_B_dataset, args.nvis_B, shuffle=False)

    
    if args.batch_size > 1:
        train_A_iter = chainer.iterators.MultiprocessIterator(
            train_A_dataset, args.batch_size, n_processes=3, shuffle=not args.conditional_discriminator)
        train_B_iter = chainer.iterators.MultiprocessIterator(
            train_B_dataset, args.batch_size, n_processes=3, shuffle=not args.conditional_discriminator)
    else:
        train_A_iter = chainer.iterators.SerialIterator(
            train_A_dataset, args.batch_size, shuffle=not args.conditional_discriminator)
        train_B_iter = chainer.iterators.SerialIterator(
            train_B_dataset, args.batch_size, shuffle=not args.conditional_discriminator)

    # setup models
    enc_x = net.Encoder(args)
    enc_y = net.Encoder(args)
    dec_x = net.Decoder(args)
    dec_y = net.Decoder(args)
    dis_x = net.Discriminator(args)
    dis_y = net.Discriminator(args)
    dis_z = net.DiscriminatorZ(args)
    models = {'enc_x': enc_x, 'enc_y': enc_y, 'dec_x': dec_x, 'dec_y': dec_y, 'dis_x': dis_x, 'dis_y': dis_y, 'dis_z': dis_z}
    optimiser_files = []

    ## load learnt models
    if args.load_models:
        for e in models:
            m = args.load_models.replace('enc_x',e)
            try:
                serializers.load_npz(m, models[e])
                print('model loaded: {}'.format(m))
            except:
                print("couldn't load {}".format(m))
                pass
            optimiser_files.append(m.replace(e,'opt_'+e).replace('dis_',''))

    # select GPU
    if len(args.gpu) == 1:
        for e in models:
            models[e].to_gpu()
        print('use gpu {}, cuDNN {}'.format(args.gpu, chainer.cuda.cudnn_enabled))
    else:
        print("mandatory GPU use: currently only a single GPU can be used")
        exit()

    # Setup optimisers
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        eps = 1e-5 if args.dtype==np.float16 else 1e-8
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, eps=eps)
        optimizer.setup(model)
        if args.weight_decay>0:
            if args.weight_decay_norm =='l2':
                optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
            else:
                optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))
        return optimizer

    opt_enc_x = make_optimizer(enc_x, alpha=args.learning_rate_g)
    opt_dec_x = make_optimizer(dec_x, alpha=args.learning_rate_g)
    opt_enc_y = make_optimizer(enc_y, alpha=args.learning_rate_g)
    opt_dec_y = make_optimizer(dec_y, alpha=args.learning_rate_g)
    opt_x = make_optimizer(dis_x, alpha=args.learning_rate_d)
    opt_y = make_optimizer(dis_y, alpha=args.learning_rate_d)
    opt_z = make_optimizer(dis_z, alpha=args.learning_rate_d)
    optimizers = {'opt_enc_x': opt_enc_x,'opt_dec_x': opt_dec_x,'opt_enc_y': opt_enc_y,'opt_dec_y': opt_dec_y,'opt_x': opt_x,'opt_y': opt_y,'opt_z': opt_z}
    if args.load_optimizer:
        for (m,e) in zip(optimiser_files,optimizers):
            if m:
                try:
                    serializers.load_npz(m, optimizers[e])
                    print('optimiser loaded: {}'.format(m))
                except:
                    print("couldn't load {}".format(m))
                    pass

    # Set up an updater: TODO: multi gpu updater
    print("Preparing updater...")
    updater = Updater(
        models=(enc_x,dec_x,enc_y,dec_y, dis_x, dis_y, dis_z),
        iterator={
            'main': train_A_iter,
            'train_B': train_B_iter,
        },
        optimizer=optimizers,
        converter=convert.ConcatWithAsyncTransfer(),
        device=args.gpu[0],
        params={
            'args': args
        })

    if not args.snapinterval:
        args.snapinterval = (args.lrdecay_start+args.lrdecay_start)//5
    log_interval = (200, 'iteration')
    model_save_interval = (args.snapinterval, 'epoch')
    vis_interval = (args.vis_freq, 'iteration')
    plot_interval = (500, 'iteration')
    
    # Set up a trainer
    print("Preparing trainer...")
    trainer = training.Trainer(updater, (args.lrdecay_start + args.lrdecay_period, 'epoch'), out=out)
    for e in models:
        trainer.extend(extensions.snapshot_object(
            models[e], e+'{.updater.epoch}.npz'), trigger=model_save_interval)
    for e in optimizers:
        trainer.extend(extensions.snapshot_object(
            optimizers[e], e+'{.updater.epoch}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration']
    log_keys_cycle = ['opt_enc_x/loss_cycle', 'opt_enc_y/loss_cycle', 'opt_dec_x/loss_cycle',  'opt_dec_y/loss_cycle', 'myval/cycle_y_l1']
    log_keys_d = ['opt_x/loss_real','opt_x/loss_fake','opt_y/loss_real','opt_y/loss_fake','opt_z/loss_x','opt_z/loss_y']
    log_keys_adv = ['opt_enc_y/loss_adv','opt_dec_y/loss_adv','opt_enc_x/loss_adv','opt_dec_x/loss_adv']
    log_keys.extend([ 'opt_dec_y/loss_id', 'opt_enc_x/loss_dom','opt_enc_y/loss_dom', 'opt_dec_y/loss_grad','opt_dec_x/loss_grad'])
    log_keys.extend([ 'opt_enc_x/loss_reg','opt_enc_y/loss_reg', 'opt_dec_x/loss_air','opt_dec_y/loss_air', 'opt_dec_y/loss_tv'])
    log_keys_d.extend(['opt_x/loss_gp','opt_y/loss_gp'])

    log_keys_all = log_keys+log_keys_d+log_keys_adv+log_keys_cycle
    trainer.extend(extensions.LogReport(keys=log_keys_all, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys_all), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(CommandsExtension())
    ## to dump graph, set -lix 1 --warmup 0
#    trainer.extend(extensions.dump_graph('opt_g/loss_id', out_name='gen.dot'))
#    trainer.extend(extensions.dump_graph('opt_x/loss', out_name='dis.dot'))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(log_keys[2:], 'iteration',trigger=plot_interval, file_name='loss.png'))
        trainer.extend(extensions.PlotReport(log_keys_d, 'iteration', trigger=plot_interval, file_name='loss_d.png'))
        trainer.extend(extensions.PlotReport(log_keys_adv, 'iteration', trigger=plot_interval, file_name='loss_adv.png'))
        trainer.extend(extensions.PlotReport(log_keys_cycle, 'iteration', trigger=plot_interval, file_name='loss_cyc.png'))

    ## output filenames of training dataset
    with open(os.path.join(out, 'trainA.txt'),'w') as output:
        output.writelines("\n".join(train_A_dataset.ids))
    with open(os.path.join(out, 'trainB.txt'),'w') as output:
        output.writelines("\n".join(train_B_dataset.ids))
    # archive the scripts
    rundir = os.path.dirname(os.path.realpath(__file__))
    import zipfile
    with zipfile.ZipFile(os.path.join(out,'script.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for f in ['trainAE.py','net.py','updaterAE.py','consts.py','losses.py','arguments.py','convert.py']:
            new_zip.write(os.path.join(rundir,f),arcname=f)

    ## visualisation
    vis_folder = os.path.join(out, "vis")
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
#    trainer.extend(visualize( (enc_x, enc_y, dec_y), vis_folder, test_A_iter, test_B_iter),trigger=(1, 'epoch'))
    trainer.extend(VisEvaluator({"main":test_A_iter, "testB":test_B_iter}, {"enc_x":enc_x, "enc_y":enc_y,"dec_x":dec_x,"dec_y":dec_y},
            params={'vis_out': vis_folder, 'single_encoder': args.single_encoder}, device=args.gpu[0]),trigger=vis_interval)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
