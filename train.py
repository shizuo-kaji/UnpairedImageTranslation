#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime as dt
import numpy as np

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainer.dataset import convert
import chainer.functions as F

import net
from updater import Updater
from visualization import VisEvaluator
from arguments import arguments 
from consts import activation,dtypes

def main():
    args = arguments()
    out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M'))
    print(args)
    print(out)
    save_args(args, out)
    args.dtype = dtypes[args.dtype]
    args.dis_activation = activation[args.dis_activation]
    args.gen_activation = activation[args.gen_activation]
    args.gen_out_activation = activation[args.gen_out_activation]

    if args.imgtype=="dcm":
        from dataset_dicom import DatasetOutMem as Dataset 
    else:
        from dataset_jpg import DatasetOutMem as Dataset   

    if not chainer.cuda.available:
        print("CUDA required")
        exit()

    if len(args.gpu)==1 and args.gpu[0] >= 0:
        chainer.cuda.get_device_from_id(args.gpu[0]).use()

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = args.dtype
    chainer.print_runtime_info()
#    print('Chainer version: ', chainer.__version__)
#    print('GPU availability:', chainer.cuda.available)
#    print('cuDNN availability:', chainer.cuda.cudnn_enabled)


    ## dataset iterator
    print("Setting up data iterators...")
    train_A_dataset = Dataset(
        path=os.path.join(args.root, 'trainA'), baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width),random=args.random_translate, forceSpacing=0, imgtype=args.imgtype, dtype=args.dtype)
    train_B_dataset = Dataset(
        path=os.path.join(args.root, 'trainB'),  baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=args.random_translate, forceSpacing=args.forceSpacing, imgtype=args.imgtype,dtype=args.dtype)
    test_A_dataset = Dataset(
        path=os.path.join(args.root, 'testA'), baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=0, forceSpacing=0, imgtype=args.imgtype,dtype=args.dtype)
    test_B_dataset = Dataset(
        path=os.path.join(args.root, 'testB'),  baseA=args.HU_base, rangeA=args.HU_range, slice_range=args.slice_range, crop=(args.crop_height,args.crop_width), random=0, forceSpacing=args.forceSpacing, imgtype=args.imgtype,dtype=args.dtype)

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
    gen_g = net.Generator(args)
    gen_f = net.Generator(args)
    dis_y = net.Discriminator(args)
    dis_x = net.Discriminator(args)
    models = {'gen_g':gen_g, 'gen_f':gen_f, 'dis_x':dis_x, 'dis_y':dis_y}

    ## load learnt models
    optimiser_files = []
    if args.load_models:
        for e in models:
            m = args.load_models.replace('gen_g',e)
            try:
                serializers.load_npz(m, models[e])
                print('model loaded: {}'.format(m))
            except:
                print("couldn't load {}".format(m))
                pass       
            optimiser_files.append(m.replace(e,'opt_'+e[-1]))

    # select GPU
    if len(args.gpu)==1:
        for e in models:
            models[e].to_gpu()
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

    opt_g = make_optimizer(gen_g, alpha=args.learning_rate_g)
    opt_f = make_optimizer(gen_f, alpha=args.learning_rate_g)
    opt_y = make_optimizer(dis_y, alpha=args.learning_rate_d)
    opt_x = make_optimizer(dis_x, alpha=args.learning_rate_d)
#    opt_y.add_hook(chainer.optimizer_hooks.GradientClipping(5))
    optimizers = {'opt_g':opt_g, 'opt_f':opt_f, 'opt_x':opt_x, 'opt_y':opt_y}
    if args.load_optimizer:
        for (m,e) in zip(optimiser_files,optimizers):
            if m:
                try:
                    serializers.load_npz(m, optimizers[e])
                    print('optimiser loaded: {}'.format(m))
                except:
                    print("couldn't load {}".format(m))
                    pass

    # Set up an updater
    print("Preparing updater...")
    updater = Updater(
        models=(gen_g, gen_f, dis_x, dis_y),
        iterator={
            'main': train_A_iter,
            'train_B': train_B_iter,
        },
        optimizer=optimizers,
        device=args.gpu[0],
        converter=convert.ConcatWithAsyncTransfer(),
        params={'args': args}
        )

    if args.snapinterval<0:
        args.snapinterval = args.lrdecay_start+args.lrdecay_period
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

    ## log
    log_keys = ['epoch', 'iteration']
    log_keys_cycle = ['opt_g/loss_cycle_y', 'opt_f/loss_cycle_x','myval/cycle_y_l1','myval/cycle_x_l1',  'opt_g/loss_tv']
    log_keys_d = ['opt_x/loss_real','opt_x/loss_fake','opt_y/loss_real','opt_y/loss_fake'] # ,'opt_x/loss_gp','opt_y/loss_gp']
    log_keys_adv = ['opt_g/loss_adv','opt_f/loss_adv']
    log_keys.extend([ 'opt_g/loss_id','opt_f/loss_id']) # ,'opt_g/loss_idem', 'opt_f/loss_idem','opt_g/loss_dom', 'opt_f/loss_dom',
    log_keys.extend([ 'opt_g/loss_air','opt_f/loss_air'])   # 'opt_g/loss_grad','opt_f/loss_grad', 

    log_keys_all = log_keys+log_keys_d+log_keys_adv+log_keys_cycle
    trainer.extend(extensions.LogReport(keys=log_keys_all, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys_all), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    ## to dump graph, set -lix 1 --warmup 0
#    trainer.extend(extensions.dump_graph('opt_g/loss_id', out_name='gen.dot'))
#    trainer.extend(extensions.dump_graph('opt_x/loss_real', out_name='dis.dot'))

    # ChainerUI
    trainer.extend(CommandsExtension())

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(log_keys[2:], 'iteration', trigger=plot_interval, file_name='loss.png'))
        trainer.extend(extensions.PlotReport(log_keys_d, 'iteration', trigger=plot_interval, file_name='loss_d.png'))
        trainer.extend(extensions.PlotReport(log_keys_adv, 'iteration', trigger=plot_interval, file_name='loss_adv.png'))
        trainer.extend(extensions.PlotReport(log_keys_cycle, 'iteration', trigger=plot_interval, file_name='loss_cyc.png'))
    ## visualisation
    os.makedirs(out, exist_ok=True)
    vis_folder = os.path.join(out, "vis")
    os.makedirs(vis_folder, exist_ok=True)

    ## output filenames of training dataset
    with open(os.path.join(out, 'trainA.txt'),'w') as output:
        output.writelines("\n".join(train_A_dataset.ids))
    with open(os.path.join(out, 'trainB.txt'),'w') as output:
        output.writelines("\n".join(train_B_dataset.ids))
    # archive the scripts
    rundir = os.path.dirname(os.path.realpath(__file__))
    import zipfile
    with zipfile.ZipFile(os.path.join(out,'script.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for f in ['train.py','net.py','updater.py','consts.py','losses.py','arguments.py','convert.py']:
            new_zip.write(os.path.join(rundir,f),arcname=f)

#    trainer.extend(visualize( (gen_g, gen_f), vis_folder, test_A_iter, test_B_iter), trigger=(1, 'epoch'))
    trainer.extend(VisEvaluator({"main":test_A_iter, "testB":test_B_iter}, {"gen_g":gen_g, "gen_f":gen_f},
            params={'vis_out': vis_folder, 'single_encoder': None}, device=args.gpu[0]),trigger=vis_interval )

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
