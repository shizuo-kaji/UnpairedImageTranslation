import argparse
import numpy as np
import chainer.functions as F
from consts import activation,dtypes

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-R', default='data', help='Directory containing trainA, trainB, testA, testB')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--gpu', '-g', type=int, nargs="*", default=[0],
                        help='GPU IDs')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--argfile', '-a', help="specify args file to read")
    parser.add_argument('--imgtype', '-it', default="jpg", help="image file type (file extension)")

    parser.add_argument('--learning_rate', '-lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--learning_rate_g', '-lrg', type=float, default=1e-4,   # 2e-4 in the original paper
                        help='Learning rate for generator')
    parser.add_argument('--learning_rate_d', '-lrd', type=float, default=1e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--lrdecay_start', '-e1', type=int, default=25,
                        help='anneal the learning rate (by epoch)')
    parser.add_argument('--lrdecay_period', '-e2', type=int,default=25, 
                        help='period to anneal the learning')
    parser.add_argument('--epoch', '-e', type=int, default=None,
                        help='epoch')
    parser.add_argument('--snapinterval', '-si', type=int, default=-1, 
                        help='take snapshot every this epoch')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,
                        help='weight decay for regularization')
    parser.add_argument('--weight_decay_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')

    # 
    parser.add_argument('--crop_width', '-cw', type=int, default=480, help='better to have a value divisible by a large power of two')
    parser.add_argument('--crop_height', '-ch', type=int, default=384, help='better to have a value divisible by a large power of two')
    parser.add_argument('--grey', action='store_true',
                        help='greyscale')

    parser.add_argument('--load_optimizer', '-op', action='store_true', help='load optimizer parameters')
    parser.add_argument('--load_models', '-m', default='', 
                        help='load models: specify enc_x/gen_g model file')

    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    parser.add_argument('--eqconv', '-eq', action='store_true',
                        help='Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true',
                        help='Separable Convolution')

    # options for converter
    parser.add_argument('--output_analysis', '-oa', action='store_true',
                        help='Output analysis images in conversion')

    # data augmentation
    parser.add_argument('--random_translate', '-rt', type=int, default=5, help='jitter input images by random translation')
    parser.add_argument('--noise', '-n', type=float, default=0,
                        help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, default=0.0,
                        help='strength of noise injection for the latent variable')

    ## the parameters below are not used
    parser.add_argument('--HU_base', '-hub', type=int, default=-500, help='minimum HU value to be accounted for')
    parser.add_argument('--HU_range', '-hur', type=int, default=700, help='the maximum HU value to be accounted for will be HU_base+HU_range')
    parser.add_argument('--slice_range', '-sr', type=float, nargs="*", default=None, help='')
    parser.add_argument('--forceSpacing', '-fs', type=float, default=-1,   # 0.7634, 
                            help='rescale B to match the specified spacing')

    # discriminator
    parser.add_argument('--dis_activation', '-da', default='lrelu', choices=activation.keys())
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, default=64,
                        help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, default=3,
                        help='number of down layers in discriminator')
    parser.add_argument('--dis_ksize', '-dk', type=int, default=4,
                        help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_down', '-dd', default='down', choices=['down','maxpool','maxpool_res','avgpool','avgpool_res','none'],  ## default down
                        help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', default='down', 
                        help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, default=0,
                        help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, default=None, 
                        help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', default='instance',
                        choices=['instance', 'batch','batch_aff', 'rbatch', 'fnorm', 'none'])
    parser.add_argument('--conditional_discriminator', '-cd', action='store_true',
                        help='use paired dataset for training discriminator')
    parser.add_argument('--n_critics', '-nc', type=int, default=1,
                        help='discriminator is trained this times during a single training of generators')
    parser.add_argument('--wgan', action='store_true',help='WGAN-GP')

    # generator: G: A -> B, F: B -> A
    parser.add_argument('--gen_activation', '-ga', default='relu', choices=activation.keys())
    parser.add_argument('--gen_out_activation', '-go', default='tanh', choices=activation.keys())
    parser.add_argument('--gen_fc_activation', '-gfca', default='relu', choices=activation.keys())
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, default=3,
                        help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, default=32,
                        help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, default=0,
                        help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-nb', type=int, default=9,
                        help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, default=3,    # 4 in the original paper
                        help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', default='none-7',
                        help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', default='down', choices=['down','maxpool','maxpool_res','avgpool','avgpool_res','none'],
                        help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', default='resize', choices=['unpool','unpool_res','deconv','pixsh','resize','resize_res','none'],
                        help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, default=None, 
                        help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', default='instance',
                        choices=['instance','batch','batch_aff', 'rbatch', 'fnorm', 'none'])
    parser.add_argument('--unet', '-u', default='with_last', choices=['none','no_last','with_last'],
                        help='use u-net for generator')
    parser.add_argument('--gen_start', type=int, default=200,
                        help='start using discriminator for generator training after this number of iterations')
    parser.add_argument('--warmup', type=int, default=200,
                        help='add loss L2(x,x_y)+L2(y,y_x) for warming-up iterations')
    ## loss function
    parser.add_argument('--lambda_A', '-lcA', type=float, default=10.0,
                        help='weight for cycle loss FG=Id:A -> B -> A')
    parser.add_argument('--lambda_B', '-lcB', type=float, default=10.0,
                        help='weight for cycle loss GF=Id:B -> A -> B')
    parser.add_argument('--lambda_identity_x', '-lix', type=float, default=0,
                        help='lambda for perceptual loss for A -> B')
    parser.add_argument('--lambda_identity_y', '-liy', type=float, default=0,
                        help='lambda for perceptual loss for B -> A')
    parser.add_argument('--id_ksize', '-ik', type=int, default=0,
                        help='kernel size for G-Id')
    parser.add_argument('--lambda_grad', '-lg', type=float, default=0,
                        help='lambda for gradient loss')
    parser.add_argument('--lambda_air', '-la', type=float, default=0,
                        help='lambda for air comparison loss')
    parser.add_argument('--grad_norm', default='l2', choices=['l1','l2'],
                        help='norm for gradient loss')
    parser.add_argument('--lambda_domain', '-ld', type=float, default=0,
                        help='lambda for domain preservation: G (resp. F) restricted on A (resp. B) should be Id')
    parser.add_argument('--lambda_idempotence', '-lidm', type=float, default=0,
                        help='lambda for idempotence: G^2=F^2=Id')
    parser.add_argument('--lambda_dis_y', '-ly', type=float, default=1,
                        help='lambda for discriminator for domain B')
    parser.add_argument('--lambda_dis_x', '-lx', type=float, default=1,
                        help='lambda for discriminator for domain A')
    parser.add_argument('--lambda_tv', '-ltv', type=float, default=0, ## typically, 1e-3
                        help='lambda for the total variation')
    parser.add_argument('--lambda_wgan_gp', '-lwgp', type=float, default=10,
                        help='lambda for the gradient penalty for WGAN')
    parser.add_argument('--tv_tau', '-tt', type=float, default=1e-3,
                        help='smoothing parameter for total variation')

    ## visualisation during training
    parser.add_argument('--nvis_A', type=int, default=3,
                        help='number of images in A to visualise after each epoch')
    parser.add_argument('--nvis_B', type=int, default=3,
                        help='number of images in B to visualise after each epoch')
    parser.add_argument('--vis_freq', '-vf', type=int, default=1000,
                        help='visualisation frequency in iteration')

    ## latent space model specific
    parser.add_argument('--lambda_reg', '-lreg', type=float, default=0,
                        help='weight for regularisation for encoders')
    parser.add_argument('--lambda_dis_z', '-lz', type=float, default=0,
                        help='weight for discriminator for the latent variable')
    parser.add_argument('--single_encoder', '-se', action='store_true',
                        help='enc_x = enc_y')
    parser.add_argument('--z_ndown', type=int, default=2,
                        help='number of down layers in discriminator for latent')
    parser.add_argument('--dis_z_start', type=int, default=1000,
                        help='start using dis_z after this iteration')


    args = parser.parse_args()
    if args.epoch:
        args.lrdecay_period = args.epoch//2
        args.lrdecay_start = args.epoch - args.lrdecay_period
    if args.learning_rate:
        args.learning_rate_g = args.learning_rate
        args.learning_rate_d = args.learning_rate
    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (i+1) for i in range(args.gen_ndown)]
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (i+1) for i in range(args.dis_ndown)]
    return(args)

