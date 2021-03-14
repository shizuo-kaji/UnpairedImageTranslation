#!/usr/bin/env python
import os,sys
from datetime import datetime as dt
import numpy as np

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import serializers, training, cuda
from chainer.training import extensions
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L

from net import Discriminator
from net import Encoder,Decoder
#from net_dp import Encoder,Decoder
from arguments import arguments 
from updater import Updater
from visualization import VisEvaluator
from consts import dtypes,optim
from cosshift import CosineShift

#from chainer_profutil import create_marked_profile_optimizer

def plot_ylimit(f,a,summary):
    a.set_ylim(top=0.1)
def plot_log(f,a,summary):
    a.set_yscale('log')

def main():
    args = arguments()
    outdir = os.path.join(args.out, dt.now().strftime('%m%d_%H%M'))

    if args.imgtype=="dcm":
        from dataset_dicom import Dataset as Dataset 
    else:
        from dataset_jpg import DatasetOutMem as Dataset   

    # CUDA
    if not chainer.cuda.available:
        print("This program runs hopelessly slow without CUDA!!")
    if len(args.gpu)==1 and args.gpu[0] >= 0:
        chainer.cuda.get_device_from_id(args.gpu[0]).use()

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
        path=os.path.join(args.root, 'trainA'), args=args, base=args.HU_baseA, rang=args.HU_rangeA, random_tr=args.random_translate, random_rot=args.random_rotation, random_scale=args.random_scale)
    train_B_dataset = Dataset(
        path=os.path.join(args.root, 'trainB'), args=args, base=args.HU_baseB, rang=args.HU_rangeB, random_tr=args.random_translate, random_rot=args.random_rotation, random_scale=args.random_scale)
    test_A_dataset = Dataset(
        path=os.path.join(args.root, 'testA'), args=args, base=args.HU_baseA, rang=args.HU_rangeA, random_tr=args.random_translate, random_rot=args.random_rotation, random_scale=args.random_scale)
#        path=os.path.join(args.root, 'testA'), args=args, base=args.HU_baseA, rang=args.HU_rangeA, random_tr=0, random_rot=0)
    test_B_dataset = Dataset(
        path=os.path.join(args.root, 'testB'), args=args, base=args.HU_baseB, rang=args.HU_rangeB, random_tr=args.random_translate, random_rot=args.random_rotation, random_scale=args.random_scale)
#        path=os.path.join(args.root, 'testB'), args=args, base=args.HU_baseB, rang=args.HU_rangeB, random_tr=0, random_rot=0)

    args.ch = train_A_dataset.ch
    args.out_ch = train_B_dataset.ch
    print("channels in A {}, channels in B {}".format(args.ch,args.out_ch))
    if(len(train_A_dataset)*len(train_B_dataset)==0):
        print("No images found!")
        exit()

#    test_A_iter = chainer.iterators.SerialIterator(test_A_dataset, args.nvis_A, shuffle=False)
#    test_B_iter = chainer.iterators.SerialIterator(test_B_dataset, args.nvis_B, shuffle=False)
    test_A_iter = chainer.iterators.MultithreadIterator(test_A_dataset, args.nvis_A, shuffle=False, n_threads=3)
    test_B_iter = chainer.iterators.MultithreadIterator(test_B_dataset, args.nvis_B, shuffle=False, n_threads=3)   
    train_A_iter = chainer.iterators.MultithreadIterator(train_A_dataset, args.batch_size, n_threads=3)
    train_B_iter = chainer.iterators.MultithreadIterator(train_B_dataset, args.batch_size, n_threads=3)

    # shared pretrained layer
    if (args.gen_pretrained_encoder and args.gen_pretrained_lr_ratio == 0) \
        or (args.dis_pretrained and args.dis_pretrained_lr_ratio == 0) \
        or args.lambda_identity_x > 0 or args.lambda_identity_y > 0:
            if "resnet" in args.gen_pretrained_encoder:
                pretrained = L.ResNet50Layers()
                print("Pretrained ResNet model loaded.")
            else:
                pretrained = L.VGG16Layers()
                print("Pretrained VGG model loaded.")
            if args.gpu[0] >= 0:
                pretrained.to_gpu()
    else:
        pretrained = None

    # setup models
    if (args.gen_pretrained_encoder and args.gen_pretrained_lr_ratio == 0) :
        enc_x = Encoder(args, pretrained)
        enc_y = enc_x if args.single_encoder else Encoder(args, pretrained)
    else:
        enc_x = Encoder(args)
        enc_y = enc_x if args.single_encoder else Encoder(args)
    if(args.dis_pretrained and args.dis_pretrained_lr_ratio == 0):
        dis_x = Discriminator(args, pretrained)
        dis_y = Discriminator(args, pretrained)
    else:
        dis_x = Discriminator(args)
        dis_y = Discriminator(args)
    dec_x = Decoder(args)
    dec_y = Decoder(args)
    dis_z = Discriminator(args, pretrained_off=True) if args.lambda_dis_z>0 else chainer.links.Linear(1,1)
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
    def make_optimizer(model, lr, opttype='Adam', pretrained_lr_ratio=1.0):
#        eps = 1e-5 if args.dtype==np.float16 else 1e-8
        optimizer = optim[opttype](lr)
        #from profiled_optimizer import create_marked_profile_optimizer
#        optimizer = create_marked_profile_optimizer(optim[opttype](lr), sync=True, sync_level=2)
        optimizer.setup(model)
        if args.weight_decay>0:
            if opttype in ['Adam','AdaBound','Eve']:
                optimizer.weight_decay_rate = args.weight_decay
            else:
                if args.weight_decay_norm =='l2':
                    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
                else:
                    optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))
        # finetuning
        if hasattr(model,'base') and pretrained_lr_ratio != 1.0:
            if pretrained_lr_ratio == 0:
                model.base.disable_update()
            elif opttype in ['Adam','AdaBound','Eve']:
                for func_name in model.base._children:
                    for param in model.base[func_name].params():
                        param.update_rule.hyperparam.eta *= pretrained_lr_ratio
            else:
                for func_name in model.base._children:
                    for param in model.base[func_name].params():
                        param.update_rule.hyperparam.lr *= pretrained_lr_ratio

        return optimizer

    opt_enc_x = make_optimizer(enc_x, args.learning_rate_g, args.optimizer, args.gen_pretrained_lr_ratio)
    opt_dec_x = make_optimizer(dec_x, args.learning_rate_g, args.optimizer)
    opt_enc_y = opt_enc_x if args.single_encoder else make_optimizer(enc_y, args.learning_rate_g, args.optimizer,args.gen_pretrained_lr_ratio)
    opt_dec_y = make_optimizer(dec_y, args.learning_rate_g, args.optimizer)
    opt_x = make_optimizer(dis_x, args.learning_rate_d, args.optimizer, args.dis_pretrained_lr_ratio)
    opt_y = make_optimizer(dis_y, args.learning_rate_d, args.optimizer, args.dis_pretrained_lr_ratio)
    opt_z = make_optimizer(dis_z, args.learning_rate_d, args.optimizer)
    optimizers = {'enc_x': opt_enc_x,'dec_x': opt_dec_x,'enc_y': opt_enc_y,'dec_y': opt_dec_y,'dis_x': opt_x,'dis_y': opt_y,'dis_z': opt_z}
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
        converter=convert.ConcatWithAsyncTransfer(),
        device=args.gpu[0],
        params={
            'args': args,
            'perceptual_model': pretrained
        })

    if args.snapinterval<0:
        args.snapinterval = args.epoch
    log_interval = (200, 'iteration')
    model_save_interval = (args.snapinterval, 'epoch')
    plot_interval = (500, 'iteration')
    
    # Set up a trainer
    print("Preparing trainer...")
    if args.iteration:
        stop_trigger = (args.iteration, 'iteration')
    else:
        stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=outdir)
    for e in models:
        trainer.extend(extensions.snapshot_object(
            models[e], e+'{.updater.epoch}.npz'), trigger=model_save_interval)
#        trainer.extend(extensions.ParameterStatistics(models[e]))   ## very slow
    for e in optimizers:
        trainer.extend(extensions.snapshot_object(
            optimizers[e], 'opt_'+e+'{.updater.epoch}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration','lr']
    log_keys_cycle = ['enc_x/loss_cycle', 'enc_y/loss_cycle', 'dec_x/loss_cycle',  'dec_y/loss_cycle', 'myval/cycle_x_l1', 'myval/cycle_y_l1']
    log_keys_adv = ['dec_y/loss_adv','dec_x/loss_adv']
    log_keys_d = []
    if args.lambda_dis_z>0:
        log_keys_adv.extend(['enc_y/loss_adv','enc_x/loss_adv'])
    if args.lambda_reg>0:
        log_keys.extend(['enc_x/loss_reg','enc_y/loss_reg'])
    if args.lambda_tv>0:
        log_keys.extend(['dec_y/loss_tv'])
    if args.lambda_air>0:
        log_keys.extend(['dec_x/loss_air','dec_y/loss_air'])
    if args.lambda_grad>0:
        log_keys.extend(['dec_x/loss_grad','dec_y/loss_grad'])
    if args.lambda_identity_x>0: # perceptual
        log_keys.extend(['enc_x/loss_id','enc_y/loss_id'])
    if args.lambda_domain>0:
        log_keys_cycle.extend(['dec_x/loss_dom','dec_y/loss_dom'])
    if args.dis_reg_weighting>0:
        log_keys_d.extend(['dis_x/loss_reg','dis_y/loss_reg','dis_z/loss_reg'])
    if args.dis_wgan:
        log_keys_d.extend(['dis_x/loss_dis','dis_x/loss_gp','dis_y/loss_dis','dis_y/loss_gp'])
        if args.lambda_dis_z>0:
            log_keys_d.extend(['opt_z/loss_dis','opt_z/loss_gp'])
    else:
        log_keys_d.extend(['dis_x/loss_real','dis_x/loss_fake','dis_y/loss_real','dis_y/loss_fake'])
        if args.lambda_dis_z>0:
            log_keys_d.extend(['dis_z/loss_x','dis_z/loss_y'])

    log_keys_all = log_keys[3:]+log_keys_d+log_keys_adv+log_keys_cycle
    trainer.extend(extensions.LogReport(keys=log_keys_all, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys_all), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))
    trainer.extend(extensions.observe_lr(optimizer_name='enc_x'), trigger=log_interval)

    # learning rate scheduling
    if args.optimizer in ['Adam','AdaBound','Eve']:
        lr_target = 'eta'
    else:
        lr_target = 'lr'
    if args.lr_drop > 0:  ## cosine annealing
        for e in [opt_enc_x,opt_enc_y,opt_dec_x,opt_dec_y,opt_x,opt_y,opt_z]:
            trainer.extend(CosineShift(lr_target, args.epoch//args.lr_drop, optimizer=e), trigger=(1, 'epoch'))
    else:
        decay_start_iter = len(train_A_dataset) * args.epoch // 2
        decay_end_iter = len(train_A_dataset) * args.epoch
        for e in [opt_enc_x,opt_enc_y,opt_dec_x,opt_dec_y,opt_x,opt_y,opt_z]:
            trainer.extend(extensions.LinearShift(lr_target, (1.0,0.0), (decay_start_iter,decay_end_iter), optimizer=e))

    ## dump graph
    if args.lambda_Az>0:
        trainer.extend(extensions.dump_graph('enc_y/loss_cycle', out_name='gen.dot'))
    if args.lambda_dis_x>0:
        if args.dis_wgan:
            trainer.extend(extensions.dump_graph('dis_x/loss_dis', out_name='dis.dot'))
        else:
            trainer.extend(extensions.dump_graph('dis_x/loss_fake', out_name='dis.dot'))

    # ChainerUI
    trainer.extend(CommandsExtension())

    if extensions.PlotReport.available():
#        trainer.extend(extensions.PlotReport(['lr'], 'iteration',trigger=plot_interval, file_name='lr.png'))
        trainer.extend(extensions.PlotReport(log_keys[3:], 'iteration',trigger=plot_interval, file_name='loss.png', postprocess=plot_log))
        trainer.extend(extensions.PlotReport(log_keys_d, 'iteration', trigger=plot_interval, file_name='loss_d.png'))
        trainer.extend(extensions.PlotReport(log_keys_adv, 'iteration', trigger=plot_interval, file_name='loss_adv.png'))
        trainer.extend(extensions.PlotReport(log_keys_cycle, 'iteration', trigger=plot_interval, file_name='loss_cyc.png', postprocess=plot_log))

    ## visualisation
    vis_folder = os.path.join(outdir, "vis")
    os.makedirs(vis_folder, exist_ok=True)
    if not args.vis_freq:
        args.vis_freq = len(train_A_dataset)//2        
    s = [k for k in range(args.num_slices)] if args.num_slices>0 and args.imgtype=="dcm" else None
    trainer.extend(VisEvaluator({"testA":test_A_iter, "testB":test_B_iter}, {"enc_x":enc_x, "enc_y":enc_y,"dec_x":dec_x,"dec_y":dec_y},
            params={'vis_out': vis_folder, 'slice':s, 'args':args}, device=args.gpu[0]),trigger=(args.vis_freq, 'iteration'))

    ## output filenames of training dataset
    with open(os.path.join(outdir, 'trainA.txt'),'w') as output:
        for f in train_A_dataset.names:
            output.writelines("\n".join(f))
            output.writelines("\n")
    with open(os.path.join(outdir, 'trainB.txt'),'w') as output:
        for f in train_B_dataset.names:
            output.writelines("\n".join(f))
            output.writelines("\n")

    # archive the scripts
    rundir = os.path.dirname(os.path.realpath(__file__))
    import zipfile
    with zipfile.ZipFile(os.path.join(outdir,'script.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for f in ['train.py','net.py','updater.py','consts.py','losses.py','arguments.py','convert.py']:
            new_zip.write(os.path.join(rundir,f),arcname=f)

    # Run the training
    print("\nresults are saved under: ",outdir)
    save_args(args, outdir)
    with open(os.path.join(outdir,"args.txt"), 'w') as fh:
        fh.write(" ".join(sys.argv))
    trainer.run()


if __name__ == '__main__':
    main()
