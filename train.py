#!/usr/bin/env python
## TODO: detect failure from mismatch between channels
import os
from datetime import datetime as dt
import numpy as np

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import serializers, training, cuda
from chainer.training import extensions
#from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainer.dataset import convert
import chainer.functions as F

import net
from arguments import arguments 
from updater import Updater
from visualization import VisEvaluator
from consts import dtypes,optim

def main():
    args = arguments()
    out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M'))
    print(args)
    print("\nresults are saved under: ",out)
    save_args(args, out)

    if args.imgtype=="dcm":
        from dataset_dicom import Dataset as Dataset 
    else:
        from dataset_jpg import DatasetOutMem as Dataset   

    # CUDA
    if not chainer.cuda.available:
        print("CUDA required")
        exit()
    if len(args.gpu)==1 and args.gpu[0] >= 0:
        chainer.cuda.get_device_from_id(args.gpu[0]).use()
#        cuda.cupy.cuda.set_allocator(cuda.cupy.cuda.MemoryPool().malloc)

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]
    chainer.print_runtime_info()
    # Turn off type check
#    chainer.config.type_check = False
#    print('Chainer version: ', chainer.__version__)
#    print('GPU availability:', chainer.cuda.available)
#    print('cuDNN availablility:', chainer.cuda.cudnn_enabled)

    ## dataset iterator
    print("Setting up data iterators...")
    train_A_dataset = Dataset(
        path=os.path.join(args.root, 'trainA'), args=args, random=args.random_translate, forceSpacing=0)
    train_B_dataset = Dataset(
        path=os.path.join(args.root, 'trainB'), args=args, random=args.random_translate, forceSpacing=args.forceSpacing)
    test_A_dataset = Dataset(
        path=os.path.join(args.root, 'testA'), args=args, random=0, forceSpacing=0)
    test_B_dataset = Dataset(
        path=os.path.join(args.root, 'testB'), args=args, random=0, forceSpacing=args.forceSpacing)

    args.ch = train_A_dataset.ch
    test_A_iter = chainer.iterators.SerialIterator(test_A_dataset, args.nvis_A, shuffle=False)
    test_B_iter = chainer.iterators.SerialIterator(test_B_dataset, args.nvis_B, shuffle=False)

    
    if args.batch_size > 1:
        train_A_iter = chainer.iterators.MultiprocessIterator(
            train_A_dataset, args.batch_size, n_processes=3)
        train_B_iter = chainer.iterators.MultiprocessIterator(
            train_B_dataset, args.batch_size, n_processes=3)
    else:
        train_A_iter = chainer.iterators.SerialIterator(
            train_A_dataset, args.batch_size)
        train_B_iter = chainer.iterators.SerialIterator(
            train_B_dataset, args.batch_size)

    # setup models
    enc_x = net.Encoder(args)
    enc_y = enc_x if args.single_encoder else net.Encoder(args)
    dec_x = net.Decoder(args)
    dec_y = net.Decoder(args)
    dis_x = net.Discriminator(args)
    dis_y = net.Discriminator(args)
    dis_z = net.Discriminator(args) if args.lambda_dis_z>0 else chainer.links.Linear(1,1)
    models = {'enc_x': enc_x, 'dec_x': dec_x, 'enc_y': enc_y, 'dec_y': dec_y, 'dis_x': dis_x, 'dis_y': dis_y, 'dis_z': dis_z}

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

    # select GPU
    if len(args.gpu) == 1:
        for e in models:
            models[e].to_gpu()
        print('using gpu {}, cuDNN {}'.format(args.gpu, chainer.cuda.cudnn_enabled))
    else:
        print("mandatory GPU use: currently only a single GPU can be used")
        exit()

    # Setup optimisers
    def make_optimizer(model, lr, opttype='Adam'):
#        eps = 1e-5 if args.dtype==np.float16 else 1e-8
        optimizer = optim[opttype](lr)
        if args.weight_decay>0:
            if opttype in ['Adam','AdaBound','Eve']:
                optimizer.weight_decay_rate = args.weight_decay
            else:
                if args.weight_decay_norm =='l2':
                    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
                else:
                    optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))
        optimizer.setup(model)
        return optimizer

    opt_enc_x = make_optimizer(enc_x, args.learning_rate_g, args.optimizer)
    opt_dec_x = make_optimizer(dec_x, args.learning_rate_g, args.optimizer)
    opt_enc_y = make_optimizer(enc_y, args.learning_rate_g, args.optimizer)
    opt_dec_y = make_optimizer(dec_y, args.learning_rate_g, args.optimizer)
    opt_x = make_optimizer(dis_x, args.learning_rate_d, args.optimizer)
    opt_y = make_optimizer(dis_y, args.learning_rate_d, args.optimizer)
    opt_z = make_optimizer(dis_z, args.learning_rate_d, args.optimizer)
    optimizers = {'opt_enc_x': opt_enc_x,'opt_dec_x': opt_dec_x,'opt_enc_y': opt_enc_y,'opt_dec_y': opt_dec_y,'opt_x': opt_x,'opt_y': opt_y,'opt_z': opt_z}
    if args.load_optimizer:
        for e in optimizers:
            try:
                m = args.load_models.replace('enc_x',e)
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
#        converter=convert.ConcatWithAsyncTransfer(),
        device=args.gpu[0],
        params={
            'args': args
        })

    if args.snapinterval<0:
        args.snapinterval = args.lrdecay_start+args.lrdecay_period
    log_interval = (200, 'iteration')
    model_save_interval = (args.snapinterval, 'epoch')
    plot_interval = (500, 'iteration')
    
    # Set up a trainer
    print("Preparing trainer...")
    trainer = training.Trainer(updater, (args.lrdecay_start + args.lrdecay_period, 'epoch'), out=out)
    for e in models:
        trainer.extend(extensions.snapshot_object(
            models[e], e+'{.updater.epoch}.npz'), trigger=model_save_interval)
        trainer.extend(extensions.ParameterStatistics(models[e]))
    for e in optimizers:
        trainer.extend(extensions.snapshot_object(
            optimizers[e], e+'{.updater.epoch}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration','lr']
    log_keys_cycle = ['opt_enc_x/loss_cycle', 'opt_enc_y/loss_cycle', 'opt_dec_x/loss_cycle',  'opt_dec_y/loss_cycle', 'myval/cycle_x_l1', 'myval/cycle_y_l1']
    log_keys_d = ['opt_x/loss_real','opt_x/loss_fake','opt_y/loss_real','opt_y/loss_fake','opt_z/loss_x','opt_z/loss_y']
    log_keys_adv = ['opt_enc_y/loss_adv','opt_dec_y/loss_adv','opt_enc_x/loss_adv','opt_dec_x/loss_adv']
    log_keys.extend([ 'opt_enc_x/loss_reg','opt_enc_y/loss_reg', 'opt_dec_y/loss_tv'])
    if args.lambda_air>0:
        log_keys.extend(['opt_dec_x/loss_air','opt_dec_y/loss_air'])
    if args.lambda_grad>0:
        log_keys.extend([ 'opt_dec_x/loss_grad','opt_dec_y/loss_grad'])
    if args.lambda_identity_x>0:
        log_keys.extend(['opt_dec_x/loss_id','opt_dec_y/loss_id'])
    if args.dis_reg_weighting>0:
        log_keys_d.extend(['opt_x/loss_reg','opt_y/loss_reg','opt_z/loss_reg'])
    if args.wgan:
        log_keys_d.extend(['opt_x/loss_gp','opt_y/loss_gp','opt_z/loss_gp'])

    log_keys_all = log_keys+log_keys_d+log_keys_adv+log_keys_cycle
    trainer.extend(extensions.LogReport(keys=log_keys_all, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys_all), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(extensions.observe_lr(optimizer_name='opt_enc_x'), trigger=log_interval)
    # learning rate scheduling
    decay_start_iter = len(train_A_dataset) * args.lrdecay_start
    decay_end_iter = len(train_A_dataset) * (args.lrdecay_start+args.lrdecay_period)
    for e in [opt_enc_x,opt_enc_y,opt_dec_x,opt_dec_y]:
        trainer.extend(extensions.LinearShift('alpha', (args.learning_rate_g,0), (decay_start_iter,decay_end_iter), optimizer=e))
    for e in [opt_x,opt_y,opt_z]:
        trainer.extend(extensions.LinearShift('alpha', (args.learning_rate_d,0), (decay_start_iter,decay_end_iter), optimizer=e))
    ## dump graph
    if args.report_start < 1:
        if args.lambda_tv>0:
            trainer.extend(extensions.dump_graph('opt_dec_y/loss_tv', out_name='dec.dot'))
        if args.lambda_reg>0:
            trainer.extend(extensions.dump_graph('opt_enc_x/loss_reg', out_name='enc.dot'))            
        trainer.extend(extensions.dump_graph('opt_x/loss_fake', out_name='dis.dot'))

    # ChainerUI
#    trainer.extend(CommandsExtension())

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(log_keys[3:], 'iteration',trigger=plot_interval, file_name='loss.png'))
        trainer.extend(extensions.PlotReport(log_keys_d, 'iteration', trigger=plot_interval, file_name='loss_d.png'))
        trainer.extend(extensions.PlotReport(log_keys_adv, 'iteration', trigger=plot_interval, file_name='loss_adv.png'))
        trainer.extend(extensions.PlotReport(log_keys_cycle, 'iteration', trigger=plot_interval, file_name='loss_cyc.png'))

    ## visualisation
    vis_folder = os.path.join(out, "vis")
    os.makedirs(vis_folder, exist_ok=True)
    if not args.vis_freq:
        args.vis_freq = len(train_d)//2        
    s = [k for k in range(args.num_slices)] if args.num_slices>0 and args.imgtype=="dcm" else None
    trainer.extend(VisEvaluator({"testA":test_A_iter, "testB":test_B_iter}, {"enc_x":enc_x, "enc_y":enc_y,"dec_x":dec_x,"dec_y":dec_y},
            params={'vis_out': vis_folder, 'slice':s}, device=args.gpu[0]),trigger=(args.vis_freq, 'iteration'))

    ## output filenames of training dataset
    with open(os.path.join(out, 'trainA.txt'),'w') as output:
        for f in train_A_dataset.names:
            output.writelines("\n".join(f))
            output.writelines("\n")
    with open(os.path.join(out, 'trainB.txt'),'w') as output:
        for f in train_B_dataset.names:
            output.writelines("\n".join(f))
            output.writelines("\n")

    # archive the scripts
    rundir = os.path.dirname(os.path.realpath(__file__))
    import zipfile
    with zipfile.ZipFile(os.path.join(out,'script.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for f in ['train.py','net.py','updater.py','consts.py','losses.py','arguments.py','convert.py']:
            new_zip.write(os.path.join(rundir,f),arcname=f)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
