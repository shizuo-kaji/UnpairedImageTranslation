import argparse
import numpy as np
import chainer.functions as F
from consts import activation_func,dtypes,norm_layer,unettype,optim
import os
from datetime import datetime as dt
import json,codecs

default_values = {'root': 'data', 'batch_size': 1, 'gpu': [0], 'out': 'result', 'imgtype': 'jpg', \
    'learning_rate': None, 'learning_rate_g': 2e-4, 'learning_rate_d': 1e-4, 'learning_freq_d': 1, 'lr_drop': 1, 'epoch': 50, 'iteration': None, \
    'snapinterval': -1, 'weight_decay': 1e-7, 'optimizer': 'Adam', \
    'crop_width': None, 'crop_height': None, 'grey': None, 'dtype': 'fp32', 'load_optimizer': False, \
    'eqconv': False, 'spconv': False, 'senet': False, \
    'random_translate': 4, 'random_rotation': 0, 'random_scale': 0, 'noise': 0, 'noise_z': 0, \
    'HU_baseA': -1024, 'HU_rangeA': 1200, 'HU_baseB': -1024, 'HU_rangeB': 1200, 'slice_range': None, 'forceSpacing': -1, 'num_slices': 1, \
    'dis_pretrained': '', 'dis_pretrained_lr_ratio': 0, 'dis_activation': 'lrelu', 'dis_out_activation': 'none', 'dis_chs': None, \
    'dis_basech': 64, 'dis_ndown': 3, 'dis_ksize': 4, 'dis_down': 'down', 'dis_sample': 'down', 'dis_jitter': 0.2, 'dis_dropout': None, \
    'dis_norm': 'instance', 'dis_reg_weighting': 0, 'dis_attention': False, \
    'gen_pretrained_encoder': '', 'gen_pretrained_lr_ratio': 0, 'gen_activation': 'relu', 'gen_out_activation': 'tanh', 'gen_chs': None, \
    'gen_ndown': 3, 'gen_basech': 32, 'gen_fc': 0, 'gen_fc_activation': 'relu', 'gen_nblock': 9, 'gen_ksize': 3, \
    'gen_sample': 'none-7', 'gen_down': 'down', 'gen_up': 'unpool', 'gen_dropout': None, 'gen_norm': 'instance', \
    'unet': 'none', 'skipdim': 4, 'latent_dim': -1, 'single_encoder': False, \
    'lambda_A': 10, 'lambda_B': 10, 'lambda_Az': 1, 'lambda_Bz': 1, 'lambda_identity_x': 0, 'lambda_identity_y': 0, \
    'perceptual_layer': 'conv1_2', 'lambda_grad': 0, 'lambda_air': 0, 'lambda_domain': 0.1, 'lambda_idempotence': 0, \
    'lambda_dis_y': 1, 'lambda_dis_x': 1, 'lambda_dis_z': 0, 'lambda_reg': 0, 'lambda_tv': 0, 'tv_tau': 1e-3, 'tv_method': 'usual', \
    'lambda_wgan_gp': 10, 'air_threshold': -0.997, \
    'nvis_A': 3, 'nvis_B': 3, 'vis_freq': None, 'HU_base_vis': 0, 'HU_range_vis': 0, \
    'ch': None, 'out_ch': None}

def arguments():
    parser = argparse.ArgumentParser(description='Image-to-image translation using an unpaired training dataset')
    parser.add_argument('--root', '-R', help='Directory containing trainA, trainB, testA, testB')
    parser.add_argument('--batch_size', '-b', type=int)
    parser.add_argument('--gpu', '-g', type=int, nargs="*", help='GPU IDs (currently, only single-GPU usage is supported')
    parser.add_argument('--out', '-o', help='Directory to output the result')
    parser.add_argument('--argfile', '-a', help="specify args file to load settings from")
    parser.add_argument('--imgtype', '-it', help="image file type (file extension)")

    parser.add_argument('--learning_rate', '-lr', type=float, help='Learning rate')
    parser.add_argument('--learning_rate_g', '-lrg', type=float, help='Learning rate for generator')
    parser.add_argument('--learning_rate_d', '-lrd', type=float, help='Learning rate for discriminator')
    parser.add_argument('--learning_freq_d', '-lfd', type=int, help='Learning frequency for discriminator')
    parser.add_argument('--lr_drop', type=int, help='How many times the learning rate drops in cosine annealing. Set to 0 for linear lr decay.')
    parser.add_argument('--epoch', '-e', type=int, help='number of epochs to train')
    parser.add_argument('--iteration', type=int, help='number of iterations to train')
    parser.add_argument('--snapinterval', '-si', type=int, help='take snapshot every this epoch')
    parser.add_argument('--weight_decay', '-wd', type=float, help='weight decay for regularization')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(), help='select optimizer')

    # input image
    parser.add_argument('--crop_width', '-cw', type=int, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--crop_height', '-ch', type=int, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--grey', action='store_true', help='greyscale')
    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), help='floating point precision')

    parser.add_argument('--load_optimizer', '-mo', action='store_true', help='load optimizer parameters from file')
    parser.add_argument('--load_models', '-m', default='', help='load models: specify enc_x/gen_g model file')

    parser.add_argument('--eqconv', '-eq', action='store_true', help='Enable Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true', help='Enable Separable Convolution')
    parser.add_argument('--senet', '-se', action='store_true', help='Enable Squeeze-and-Excitation mechanism')

    # data augmentation
    parser.add_argument('--random_translate', '-rt', type=int, help='jitter input images by random translation (in pixel)')
    parser.add_argument('--random_rotation', '-rr', type=int, help='jitter input images by random rotation (in degree)')
    parser.add_argument('--random_scale', '-rs', type=float, help='jitter input images by random scaling (in ratio)')
    parser.add_argument('--noise', '-n', type=float, help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, help='strength of noise injection for the latent variable')

    ## DICOM specific
    parser.add_argument('--HU_baseA', '-huba', type=int, help='minimum HU value to be accounted for')
    parser.add_argument('--HU_rangeA', '-hura', type=int, help='the maximum HU value to be accounted for will be HU_base+HU_range')
    parser.add_argument('--HU_baseB', '-hubb', type=int, help='minimum HU value to be accounted for')
    parser.add_argument('--HU_rangeB', '-hurb', type=int, help='the maximum HU value to be accounted for will be HU_base+HU_range')
    parser.add_argument('--slice_range', '-sr', type=float, nargs="*", help='z-coords of slices used')
    parser.add_argument('--forceSpacing', '-fs', type=float, help='scale dicom to match the specified spacing (pixel size)')
    parser.add_argument('--num_slices', '-ns', type=int, help='number of slices stacked together (for 2.5 dimensional model)')

    # discriminator
    parser.add_argument('--dis_pretrained', '-dp', type=str, choices=["","vgg","resnet"], help='Use pretrained ResNet/VGG as discriminator')
    parser.add_argument('--dis_pretrained_lr_ratio', '-dpr', type=float, help='learning rate multiplier for the pretrained part')
    parser.add_argument('--dis_activation', '-da', choices=activation_func.keys(), help='activation of middle layers discriminators')
    parser.add_argument('--dis_out_activation', '-do', choices=activation_func.keys(), help='activation of last layer of discriminators')
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, help='number of down layers in discriminator')
    parser.add_argument('--dis_ksize', '-dk', type=int, help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_down', '-dd', help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', choices=norm_layer, help='nomalisation layer for discriminator')
    parser.add_argument('--dis_reg_weighting', '-dw', type=float, help='regularisation of weighted discriminator. Set 0 to disable weighting')
    parser.add_argument('--dis_wgan', '-wgan', action='store_true',help='WGAN-GP')
    parser.add_argument('--dis_attention', action='store_true',help='attention mechanism for discriminator')

    # generator: G: A -> B, F: B -> A
    parser.add_argument('--gen_pretrained_encoder', '-gp', type=str, choices=["","vgg","resnet"], help='Use pretrained ResNet/VGG as encoder')
    parser.add_argument('--gen_pretrained_lr_ratio', '-gpr', type=float, help='learning rate multiplier for the pretrained part')
    parser.add_argument('--gen_activation', '-ga', choices=activation_func.keys(), help='activation for middle layers of generators')
    parser.add_argument('--gen_out_activation', '-go', choices=activation_func.keys(), help='activation for last layers of generators')
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_fc_activation', '-gfca', choices=activation_func.keys(), help='activation of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-gnb', type=int, help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', choices=norm_layer, help='nomalisation layer for generator')
    parser.add_argument('--unet', '-u', choices=unettype, help='use u-net skip connections for generator')
    parser.add_argument('--skipdim', '-sd', type=int, help='channel number for skip connections')
    parser.add_argument('--latent_dim', type=int, help='dimension of the latent space between encoder and decoder')
    parser.add_argument('--single_encoder', '-senc', action='store_true', help='use the same encoder enc_x = enc_y for both domains')

    ## loss function
    parser.add_argument('--lambda_A', '-lcA', type=float, help='weight for cycle loss FG=Id:A -> B -> A')
    parser.add_argument('--lambda_B', '-lcB', type=float, help='weight for cycle loss GF=Id:B -> A -> B')
    parser.add_argument('--lambda_Az', '-lcAz', type=float, help='weight for autoencoder loss Id:A -> Z -> A')
    parser.add_argument('--lambda_Bz', '-lcBz', type=float, help='weight for autoencoder loss Id:B -> Z -> B')
    parser.add_argument('--lambda_identity_x', '-lix', type=float, help='lambda for perceptual loss for A -> B')
    parser.add_argument('--lambda_identity_y', '-liy', type=float, help='lambda for perceptual loss for B -> A')
    parser.add_argument('--perceptual_layer', '-pl', type=str, help='The name of the layer of VGG16 used for perceptual loss')
    parser.add_argument('--lambda_grad', '-lg', type=float, help='lambda for gradient loss')
    parser.add_argument('--lambda_air', '-la', type=float, help='lambda for air comparison loss')
    parser.add_argument('--lambda_domain', '-ld', type=float, help='lambda for domain preservation: G (resp. F) restricted on A (resp. B) should be Id')
    parser.add_argument('--lambda_idempotence', '-lidm', type=float, help='lambda for idempotence: G^2=F^2=Id')
    parser.add_argument('--lambda_dis_y', '-ly', type=float, help='lambda for discriminator for domain B')
    parser.add_argument('--lambda_dis_x', '-lx', type=float, help='lambda for discriminator for domain A')
    parser.add_argument('--lambda_dis_z', '-lz', type=float, help='weight for discriminator for the latent variable')
    parser.add_argument('--lambda_reg', '-lreg', type=float, help='weight for regularisation for encoders')
    parser.add_argument('--lambda_tv', '-ltv', type=float, help='lambda for the total variation')
    parser.add_argument('--tv_tau', '-tt', type=float, help='smoothing parameter for total variation')
    parser.add_argument('--tv_method', '-tm', choices=['abs','sobel','usual'], help='method of calculating total variation')
    parser.add_argument('--lambda_wgan_gp', '-lwgp', type=float, help='lambda for the gradient penalty for WGAN')
    parser.add_argument('--air_threshold', '-at', type=float, help='values below this is considered as air for air comparison loss')

    ## visualisation during training
    parser.add_argument('--nvis_A', type=int, help='number of images in A to visualise after each epoch')
    parser.add_argument('--nvis_B', type=int, help='number of images in B to visualise after each epoch')
    parser.add_argument('--vis_freq', '-vf', type=int, help='visualisation frequency in iteration')
    parser.add_argument('--HU_base_vis', '-hubv', type=int, help='minimum HU value to be visualised')
    parser.add_argument('--HU_range_vis', '-hurv', type=int, help='the maximum HU value to be visualised will be HU_base+HU_range')

    # options for converter
    parser.add_argument('--output_analysis', '-oa', action='store_true',
                        help='Output analysis images in conversion')

    args = parser.parse_args()

    # number of channels in input/output images: infered from data or args file.
    args.ch = None
    args.out_ch = None

    ## set default values from file 
    if args.argfile:
        with open(args.argfile, 'r') as f:
            larg = json.load(f)
    else:
        larg = []
    for x in default_values:
        if getattr(args, x) is None:
            if x in larg:
                setattr(args, x, larg[x])
            else:
                setattr(args, x, default_values[x])

    if args.learning_rate:
        args.learning_rate_g = args.learning_rate
        args.learning_rate_d = args.learning_rate/2
    if "resnet" in args.gen_pretrained_encoder:
        args.gen_chs = [64,256,512,1024,2048][:args.gen_ndown]
    elif "vgg" in args.gen_pretrained_encoder:
        args.gen_chs = [64,128,256,512,512][:args.gen_ndown]
    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (2**i) for i in range(args.gen_ndown)]
    else:
        args.gen_ndown = len(args.gen_chs)
    if "resnet" in args.dis_pretrained:
        args.dis_chs = [64,256,512,1024,2048][:args.dis_ndown]
    elif "vgg" in args.dis_pretrained:
        args.dis_chs = [64,128,256,512,512][:args.dis_ndown]
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (2**i) for i in range(args.dis_ndown)]
    else:
        args.dis_ndown = len(args.dis_chs)

    if args.imgtype=="dcm":
        args.grey = True

    if args.gen_fc>0 and args.crop_width is None:
        print("Specify crop_width and crop_height!")
        exit()

    # convert.py
    if args.out_ch is None:
        args.out_ch = 1 if args.grey else 3
    if args.ch is None:
        args.ch = 1 if args.grey else 3



    ## temp
    # args.random_translate=50
    # args.random_rotation=20
    # args.HU_baseA, args.HU_rangeA = -600, 1000
    # args.HU_baseB, args.HU_rangeB = -600, 1000
    # args.HU_base_vis, args.HU_range_vis = -600, 800
    # args.forceSpacing = 0.7634

    print(args)

    return(args)
