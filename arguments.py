import argparse
import numpy as np
import chainer.functions as F
from consts import activation_func,dtypes,uplayer,downlayer,norm_layer,unettype,optim


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-R', default='data', help='Directory containing trainA, trainB, testA, testB')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--gpu', '-g', type=int, nargs="*", default=[0],
                        help='GPU IDs (currently, only single-GPU usage is supported')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--argfile', '-a', help="specify args file to load settings from")
    parser.add_argument('--imgtype', '-it', default="jpg", help="image file type (file extension)")

    parser.add_argument('--learning_rate', '-lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--learning_rate_g', '-lrg', type=float, default=1e-4,   # 2e-4 in the original paper
                        help='Learning rate for generator')
    parser.add_argument('--learning_rate_d', '-lrd', type=float, default=1e-4,
                        help='Learning rate for discriminator')
    parser.add_argument('--lrdecay_start', '-e1', type=int, default=25,
                        help='start lowering the learning rate (in epoch)')
    parser.add_argument('--lrdecay_period', '-e2', type=int,default=25, 
                        help='period in epoch for lowering the learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=None,
                        help='epoch')
    parser.add_argument('--snapinterval', '-si', type=int, default=-1, 
                        help='take snapshot every this epoch')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-8,
                        help='weight decay for regularization')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='select optimizer')

    # 
    parser.add_argument('--crop_width', '-cw', type=int, default=None, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--crop_height', '-ch', type=int, default=None, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--grey', action='store_true', help='greyscale')

    parser.add_argument('--load_optimizer', '-mo', action='store_true', help='load optimizer parameters from file')
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
    parser.add_argument('--random_translate', '-rt', type=int, default=4, help='jitter input images by random translation')
    parser.add_argument('--noise', '-n', type=float, default=0,
                        help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, default=0,
                        help='strength of noise injection for the latent variable')

    ## DICOM specific
    parser.add_argument('--HU_base', '-hub', type=int, default=-500, help='minimum HU value to be accounted for')
    parser.add_argument('--HU_range', '-hur', type=int, default=700, help='the maximum HU value to be accounted for will be HU_base+HU_range')
    parser.add_argument('--slice_range', '-sr', type=float, nargs="*", default=None, help='')
    parser.add_argument('--forceSpacing', '-fs', type=float, default=-1,   # 0.7634, 
                            help='rescale B to match the specified spacing')
    parser.add_argument('--num_slices', '-ns', type=int, default=1, help='number of slices stacked together')

    # discriminator
    parser.add_argument('--dis_activation', '-da', default='lrelu', choices=activation_func.keys())
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, default=64,
                        help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, default=3,
                        help='number of down layers in discriminator')
    parser.add_argument('--dis_ksize', '-dk', type=int, default=4,
                        help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_down', '-dd', default='down', choices=downlayer,  ## default down
                        help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', default='down', 
                        help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, default=0,
                        help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, default=None, 
                        help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--dis_reg_weighting', '-dw', type=float, default=0,
                        help='regularisation of weighted discriminator. Set 0 to disable weighting')
    parser.add_argument('--n_critics', '-nc', type=int, default=1,
                        help='(not recommended; instead, use different learning rate for dis and gen) discriminator is trained this times during a single training of generators')
    parser.add_argument('--wgan', action='store_true',help='WGAN-GP')

    # generator: G: A -> B, F: B -> A
    parser.add_argument('--gen_activation', '-ga', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_out_activation', '-go', default='tanh', choices=activation_func.keys())
    parser.add_argument('--gen_fc_activation', '-gfca', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, default=3,
                        help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, default=32,
                        help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, default=0,
                        help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-gnb', type=int, default=9,
                        help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, default=3,    # 4 in the original paper
                        help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', default='none-7',
                        help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', default='down', choices=downlayer,
                        help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', default='resize', choices=uplayer,
                        help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, default=None, 
                        help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--unet', '-u', default='concat', choices=unettype,
                        help='use u-net skip connections for generator')
    parser.add_argument('--gen_start', type=int, default=0,
                        help='start using discriminator for generator training after this number of iterations')
    parser.add_argument('--report_start', type=int, default=1000,
                        help='start reporting losses after this number of iterations')

    ## loss function
    parser.add_argument('--lambda_A', '-lcA', type=float, default=10.0,
                        help='weight for cycle loss FG=Id:A -> B -> A')
    parser.add_argument('--lambda_B', '-lcB', type=float, default=10.0,
                        help='weight for cycle loss GF=Id:B -> A -> B')
    parser.add_argument('--lambda_Az', '-lcAz', type=float, default=10.0,
                        help='weight for autoencoder loss Id:A -> Z -> A')
    parser.add_argument('--lambda_Bz', '-lcBz', type=float, default=10.0,
                        help='weight for autoencoder loss Id:B -> Z -> B')
    parser.add_argument('--lambda_identity_x', '-lix', type=float, default=0,
                        help='lambda for perceptual loss for A -> B')
    parser.add_argument('--lambda_identity_y', '-liy', type=float, default=0,
                        help='lambda for perceptual loss for B -> A')
    parser.add_argument('--perceptual_layer', '-pl', type=str, default="conv4_2",
                        help='The name of the layer of VGG16 used for perceptual loss')
    parser.add_argument('--lambda_grad', '-lg', type=float, default=0,
                        help='lambda for gradient loss')
    parser.add_argument('--lambda_air', '-la', type=float, default=0,
                        help='lambda for air comparison loss')
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
    parser.add_argument('--lambda_reg', '-lreg', type=float, default=0,
                        help='weight for regularisation for encoders')
    parser.add_argument('--lambda_dis_z', '-lz', type=float, default=0,
                        help='weight for discriminator for the latent variable')
    parser.add_argument('--single_encoder', '-se', action='store_true',
                        help='use the same encoder enc_x = enc_y for both domains')
    parser.add_argument('--tv_tau', '-tt', type=float, default=1e-3,
                        help='smoothing parameter for total variation')
    parser.add_argument('--tv_method', '-tm', default='abs', choices=['abs','sobel','usual'],
                        help='method of calculating total variation')

    ## visualisation during training
    parser.add_argument('--nvis_A', type=int, default=3,
                        help='number of images in A to visualise after each epoch')
    parser.add_argument('--nvis_B', type=int, default=3,
                        help='number of images in B to visualise after each epoch')
    parser.add_argument('--vis_freq', '-vf', type=int, default=1000,
                        help='visualisation frequency in iteration')


    args = parser.parse_args()
    if args.epoch:
        args.lrdecay_period = args.epoch//2
        args.lrdecay_start = args.epoch - args.lrdecay_period
    else:
        args.epoch = args.lrdecay_start + args.lrdecay_period
    if args.learning_rate:
        args.learning_rate_g = args.learning_rate
        args.learning_rate_d = args.learning_rate
    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (2**i) for i in range(args.gen_ndown)]
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (2**i) for i in range(args.dis_ndown)]
    if args.imgtype=="dcm":
        args.grey = True
    return(args)

