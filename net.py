import functools

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from consts import activation_func, norm_layer

try:
    from sn import SNConvolution2D,SNLinear
except:
    pass

class SEBlock(chainer.Chain):
    def __init__(self,ch,r=16):
        super(SEBlock, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(ch, ch//r)
            self.l2 = L.Linear(ch//r, ch)

    def __call__(self, x):
        b,c,height,width = x.data.shape
        h = F.average(x, axis=(2, 3))  # Global pooling
        h = F.relu(self.l1(h))
        h = F.sigmoid(self.l2(h))
        return(F.transpose(F.broadcast_to(h, (height,width,b,c)), (2, 3, 0, 1)))
##
class EqualizedConv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad, pad_type='zero', equalised=False, nobias=False,separable=False, senet=False):
        self.equalised = equalised
        self.separable = separable
        self.senet = senet
        self.pad_type = pad_type
        self.pad = pad if pad_type=='zero' else 0
        if equalised:
            w = chainer.initializers.Normal(1.0) # equalized learning rate
        else:
            w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        self.ksize = ksize
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            if self.separable:
                self.depthwise = L.Convolution2D(in_ch, in_ch, ksize, stride, pad, initialW=w, nobias=True, groups=in_ch)
                self.pointwise = L.Convolution2D(in_ch, out_ch, 1, 1, initialW=w, nobias=nobias, initial_bias=bias)
            else:
                self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w, nobias=nobias, initial_bias=bias)
            if self.senet and out_ch>15:
                self.se = SEBlock(out_ch)
    def __call__(self, x):
        if self.pad_type=='reflect':
            h = F.pad(x,[[0,0],[0,0],[self.pad,self.pad],[self.pad,self.pad]],mode='reflect')
        else:
            h=x
        if self.equalised:
            b,c,_,_ = h.shape
            inv_c = np.sqrt(2.0/c)/self.ksize
            h = inv_c * h
        if self.separable:
            h=self.pointwise(self.depthwise(h))
        else:
            h = self.c(h)
        if hasattr(self,'se'):
            h = h * self.se(h)
        return h

class EqualizedDeconv2d(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, stride, pad, equalised=False, nobias=False,separable=False):
        self.equalised = equalised
        self.separable = separable
        self.pad = pad
        self.ksize = ksize
        if equalised:
            w = chainer.initializers.Normal(1.0) # equalized learning rate
        else:
            w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        self.ksize = ksize
        super(EqualizedDeconv2d, self).__init__()
        with self.init_scope():
            if self.separable:
                self.depthwise = L.Deconvolution2D(in_ch, in_ch, ksize, stride, pad, initialW=w, nobias=True, groups=in_ch)
                self.pointwise = L.Deconvolution2D(in_ch, out_ch, 1, 1, initialW=w, nobias=nobias, initial_bias=bias)
            else:
                self.c = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w, nobias=nobias,initial_bias=bias)
    def __call__(self, x):
        h=x
        if self.equalised:
            b,c,_,_ = h.shape
            inv_c = np.sqrt(2.0/c)/self.ksize
            h = inv_c * h
        if self.separable:
            h = self.pointwise(self.depthwise(h))
        else:
            h = self.c(h)
        if(self.ksize==3):
            h = F.pad(h,[[0,0],[0,0],[0,1],[0,1]],mode='reflect')
        return h

def bilinear_upsampling(x):
    _, _, height, width = x.shape
    h = F.resize_images(x, (height*2, width*2))
    return h

### the num of input channels should be divisible by 4
# obsolete: use F.depth2space
class PixelShuffler(chainer.Chain):
    def __init__(self, in_ch, out_ch, ksize, pad, nobias=False):
        w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        super(PixelShuffler, self).__init__()
        with self.init_scope():
#            self.c1 = L.Convolution2D(in_ch, out_ch, 1, stride=1, pad=0, initialW=w, nobias=nobias,initial_bias=bias)
            self.c = L.Convolution2D( int(in_ch / 4), out_ch, ksize, stride=1, pad=pad, initialW=w, nobias=nobias,initial_bias=bias)
    def __call__(self, x):
        B,C,H,W = x.shape
        h = F.reshape(x, (B, 2, 2, int(C/4), H, W))
        h = F.transpose(h, (0, 3, 4, 1, 5, 2))
        h = F.reshape(h, (B, int(C/4), H*2, W*2))
        return self.c(h)

class NonLocalBlock(chainer.Chain):
    def __init__(self, ch):
        self.ch = ch
        super(NonLocalBlock, self).__init__()
        with self.init_scope():
            self.theta = SNConvolution2D(ch, ch // 8, 1, 1, 0, nobias=True)
            self.phi = SNConvolution2D(ch, ch // 8, 1, 1, 0, nobias=True)
            self.g = SNConvolution2D(ch, ch // 2, 1, 1, 0, nobias=True)
            self.o_conv = SNConvolution2D(ch // 2, ch, 1, 1, 0, nobias=True)
            self.gamma = L.Parameter(np.array(0, dtype="float32"))

    def __call__(self, x):
        batchsize, _, width, height = x.shape
        f = self.theta(x).reshape(batchsize, self.ch // 8, -1)
        g = self.phi(x)
        g = F.max_pooling_2d(g, 2, 2).reshape(batchsize, self.ch // 8, -1)
        attention = F.softmax(F.matmul(f, g, transa=True), axis=2)
        h = self.g(x)
        h = F.max_pooling_2d(h, 2, 2).reshape(batchsize, self.ch // 2, -1)
        o = F.matmul(h, attention, transb=True).reshape(batchsize, self.ch // 2, width, height)
        o = self.o_conv(o)
        return x + self.gamma.W * o

class ResBlock(chainer.Chain):
    def __init__(self, ch, norm='instance', activation='relu', equalised=False, separable=False, skip_conv=False):
        super(ResBlock, self).__init__()
        self.activation = activation_func[activation]
        nobias = False
#        nobias = True if 'batch' in norm or 'instance' in norm else False
        with self.init_scope():
            self.c0 = EqualizedConv2d(ch, ch, 3, 1, 1, pad_type='zero', equalised=equalised, nobias=nobias, separable=separable)
            self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1, pad_type='zero', equalised=equalised, nobias=nobias, separable=separable)
            if skip_conv:  # skip connection
                self.cs = EqualizedConv2d(ch, ch, 1, 1, 0)
            else:
                self.cs = F.identity
            self.norm0 = norm_layer[norm](ch)
            self.norm1 = norm_layer[norm](ch)

    def __call__(self, x):
        h = self.c0(x)
        h = self.norm0(h)
        h = self.activation(h)
        h = self.c1(h)
        h = self.norm1(h)
        return h + self.cs(x)


class CBR(chainer.Chain):
    def __init__(self, ch0, ch11, ksize=3, pad=1, norm='instance',
                 sample='down', activation='relu', dropout=False, equalised=False, separable=False, senet=False):
        super(CBR, self).__init__()
        self.activation = activation_func[activation]
        self.dropout = dropout
        self.sample = sample
#        nobias = True if 'batch' in norm or 'instance' in norm else False
        nobias = False
        ch1 = 4*ch11 if 'pixsh' in sample else ch11
        with self.init_scope():
            if 'down' in sample:
                self.c1 = EqualizedConv2d(ch0, ch1, ksize, 2, pad, equalised=equalised, nobias=nobias, separable=separable,senet=senet)
            elif sample == 'none-7':
                self.c1 = EqualizedConv2d(ch0, ch1, 7, 1, 3, pad_type='reflect', equalised=equalised, nobias=nobias, separable=separable,senet=senet) 
            elif 'deconv' in sample:
                self.c1 = EqualizedDeconv2d(ch0, ch1, ksize, 2, pad, equalised=equalised, nobias=nobias, separable=separable)
            else: ## maxpool,avgpool,resize,unpool,none
                self.c1 = EqualizedConv2d(ch0, ch1, ksize, 1, pad, equalised=equalised, nobias=nobias, separable=separable,senet=senet)
            self.n1 = norm_layer[norm](ch1)
            # down/up sample layer
            if 'maxpool' in sample:
                self.d = functools.partial(F.max_pooling_2d, ksize=2, stride=2)
            elif 'avgpool' in sample:
                self.d = functools.partial(F.average_pooling_2d, ksize=2, stride=2)
            elif 'resize' in sample:
                self.u = bilinear_upsampling
            elif 'pixsh' in sample:
                self.u = functools.partial(F.depth2space,r=2)
            elif 'unpool' in sample:
                self.u = functools.partial(F.unpooling_2d, ksize=2, stride=2, cover_all=False)
            # second convolution
            if '_conv' in sample or '_res' in sample:
                self.c2 = EqualizedConv2d(ch1, ch1, 3, 1, 1, equalised=equalised, nobias=nobias, separable=separable,senet=senet)
                self.n2 = norm_layer[norm](ch1)
            # skip connection
            if '_res' in sample:
                if 'maxpool' in sample or 'avgpool' in sample or 'down' in sample:
                    self.skip = EqualizedConv2d(ch0, ch1, 3, 2, 1, equalised=equalised, separable=True)
#                elif 'unpool' in sample or 'resize' in sample:
#                    self.skip = EqualizedDeconv2d(ch0, ch1, 3, 2, 1, equalised=equalised, separable=True)
                else:
                    self.skip = EqualizedConv2d(ch0, ch1, 1, 1, 0, equalised=equalised, separable=True)

    def __call__(self, x):
#        print("*:",x.shape)
        h = self.n1(self.c1(x))
        if hasattr(self,'c2') and self.activation is not None:
            h = self.activation(h)
        if hasattr(self,'d'):
            h = self.d(h)
        if hasattr(self,'c2'):
            h = self.n2(self.c2(h))
        if self.dropout:
            h = F.dropout(h, ratio=self.dropout)
        if self.activation is not None:
            h = self.activation(h)
        if hasattr(self, 'skip'):
            h = h + self.skip(x)
        if hasattr(self, 'u'):
            h = self.u(h)
        return h

class LBR(chainer.Chain):
    def __init__(self, out_ch, norm='none', activation='tanh', dropout=False):
        super(LBR, self).__init__()
        self.activation = activation_func[activation]
#        nobias = True if 'batch' in norm or 'instance' in norm else False
        nobias = False
        self.dropout = dropout
        with self.init_scope():
            self.l0 = L.Linear(None, out_ch, nobias=nobias)
            self.norm = norm_layer[norm](out_ch)

    def __call__(self, x):
        h = self.l0(x)
        h = self.norm(h)
#        print(F.max(h))  # bug? we always get zero if a normalization is applied
        if self.dropout:
            h = F.dropout(h, ratio=self.dropout)
        if self.activation is not None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, args, pretrained_model=None):
        super(Encoder, self).__init__()
        self.n_resblock = (args.gen_nblock+1) // 2 # half for Enc and half for Dec
        self.chs = args.gen_chs
        if hasattr(args,'unet'):
            self.unet = args.unet
        else:
            self.unet = 'none'
        self.nfc = args.gen_fc
        if pretrained_model:
            self.base=pretrained_model
            self.update_base = False

        with self.init_scope():
            for i in range(args.gen_fc):
                self.in_c = args.ch
                self.in_h = args.crop_height
                self.in_w = args.crop_width
#                print(args.ch,args.crop_height,args.crop_width)
                setattr(self, 'l' + str(i), LBR(args.crop_height*args.crop_width*args.ch, activation=args.gen_fc_activation))

            ## use pretrained network
            if hasattr(args,'gen_pretrained_encoder') and args.gen_pretrained_encoder:
                self.pretrained = True
                if "resnet" in args.gen_pretrained_encoder:
                    self.layers = ['conv1']
                    for i in range(2,args.gen_ndown+1):
                        self.layers.append('res{}'.format(i))
                else: ## VGG16
                    self.layers = ['conv{}_2'.format(i) for i in range(1,min(3,args.gen_ndown+1))]
                    self.layers.extend(['conv{}_3'.format(i) for i in range(3,args.gen_ndown+1)])
                if pretrained_model is None:
                    if "resnet" in args.gen_pretrained_encoder:
                        self.base = L.ResNet50Layers()
                    else:
                        self.base = L.VGG16Layers()
#                print(self.chs, self.layers)
            else:  ## new network
                self.pretrained = False
                self.c0 = CBR(args.ch, self.chs[0], norm=args.gen_norm, sample=args.gen_sample, activation=args.gen_activation, equalised=args.eqconv)
                for i in range(1,len(self.chs)):
                    setattr(self, 'd' + str(i), CBR(self.chs[i-1], self.chs[i], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_down, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            ## common part
            if self.unet=='conv':
                for i in range(len(self.chs)):
                    setattr(self, 's' + str(i), CBR(self.chs[i], args.skipdim, ksize=3, norm=args.gen_norm, sample='none', equalised=args.eqconv))
            for i in range(self.n_resblock):
                setattr(self, 'r' + str(i), ResBlock(self.chs[-1], norm=args.gen_norm, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            if hasattr(args,'latent_dim') and args.latent_dim>0:
                self.latent_fc = LBR(args.latent_dim, activation=args.gen_fc_activation)

    def __call__(self, x):
        h = x
        ## precomposed-FC layers (AUTOMAP)
        for i in range(self.nfc):
            h=F.reshape(getattr(self, 'l' + str(i))(h),(-1,self.in_c,self.in_h,self.in_w))
        ## down layers
        if self.pretrained:
            if h.shape[1]==1:
                e = F.concat([h,h,h])
            else:
                e = h
            if self.update_base:
                zz = self.base(e, layers=self.layers)
            else:
                with chainer.using_config('train', False) and chainer.no_backprop_mode():
                    zz = self.base(e, layers=self.layers)
            h = []
            for i,layer in enumerate(self.layers):
                if self.unet=='conv':
                    f = getattr(self, 's' + str(i))
                    h.append(f(zz[layer]))
                elif self.unet in ['concat','add']:
                    h.append(zz[layer])
                else:
                    h.append(0)
            e = zz[self.layers[-1]]
        else:
            e = self.c0(h)
            if self.unet=='conv':
                h = [self.s0(e)]
            elif self.unet in ['concat','add']:
                h = [e]
            else:
                h=[0]
            for i in range(1,len(self.chs)):
                e = getattr(self, 'd' + str(i))(e)
                if self.unet=='conv':
                    h.append(getattr(self, 's' + str(i))(e))
                elif self.unet in ['concat','add']:
                    h.append(e)
                else:
                    h.append(0)

        ## residual blocks
        for i in range(self.n_resblock):
            e = getattr(self, 'r' + str(i))(e)
        h.append(e)
        
        ## post-composed FC layer
        if hasattr(self,'latent_fc'):
            h.append(self.latent_fc(e))
#        print([e.shape for e in h])
        return h

class Decoder(chainer.Chain):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_resblock = args.gen_nblock // 2 # half for Enc and half for Dec
        self.chs = args.gen_chs
        if hasattr(args,'noise_z'):
            self.noise_z = args.noise_z
        else:
            self.noise_z = 0
        if hasattr(args,'unet'):
            self.unet = args.unet
        else:
            self.unet = 'none'
        if self.unet=='concat':
            up_chs = [2*self.chs[i] for i in range(len(self.chs))]
        elif self.unet=='conv':
            up_chs = [self.chs[i]+args.skipdim for i in range(len(self.chs))]
        else:    # ['add','none']:
            up_chs = self.chs
        with self.init_scope():
            if hasattr(args,'latent_dim') and args.latent_dim>0:
                self.latent_c = args.gen_chs[-1]
                self.latent_h = args.crop_height//(2**(len(args.gen_chs)-1))
                self.latent_w = args.crop_width//(2**(len(args.gen_chs)-1))
                print("Latent dimensions: ",self.latent_c,self.latent_h,self.latent_w)
                self.latent_fc = LBR(self.latent_c*self.latent_h*self.latent_w, activation=args.gen_fc_activation)
            for i in range(self.n_resblock):
                setattr(self, 'r' + str(i), ResBlock(self.chs[-1], norm=args.gen_norm, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            for i in range(1,len(self.chs)):
                setattr(self, 'ua' + str(i), CBR(up_chs[-i], self.chs[-i-1], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_up, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            if hasattr(args,'gen_pretrained_encoder') and "resnet" in args.gen_pretrained_encoder:
                setattr(self, 'ua'+str(len(self.chs)),CBR(up_chs[0], up_chs[0], norm=args.gen_norm, sample='resize', activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            else:
                setattr(self, 'ua'+str(len(self.chs)),CBR(up_chs[0], up_chs[0], norm=args.gen_norm, sample='none', activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            setattr(self, 'ul',CBR(up_chs[0], args.out_ch, norm='none', sample=args.gen_sample, activation=args.gen_out_activation, equalised=args.eqconv, separable=args.spconv))

    def __call__(self, h):
        if isinstance(h,list):
            e = h[-1]
        else:
            e = h
        if chainer.config.train and self.noise_z>0:   ## noise injection for latent
            e.data += self.noise_z * e.xp.random.randn(*e.data.shape, dtype=e.dtype)
        if hasattr(self,'latent_fc'):
            e = F.reshape(self.latent_fc(e),(-1,self.latent_c,self.latent_h,self.latent_w))
        for i in range(self.n_resblock):
            e = getattr(self, 'r' + str(i))(e)
        for i in range(1,len(self.chs)+1):
#            print(e.shape,h[-i-1].shape)
            if self.unet in ['conv','concat']:
                e = getattr(self, 'ua' + str(i))(F.concat([e,h[-i-1]]))
            elif self.unet=='add':
                e = getattr(self, 'ua' + str(i))(e+h[-i-1])
            else:
                e = getattr(self, 'ua' + str(i))(e)
        e = self.ul(e)
        return e


class Generator(chainer.Chain):
    def __init__(self, args, pretrained_model=None):
        super(Generator, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(args, pretrained_model=pretrained_model)
            self.decoder = Decoder(args)
    def __call__(self, x):
#        print(self.encoder.base.conv1_1.W.update_rule.hyperparam.eta,self.decoder.ua1.c1.c.W.update_rule.hyperparam.eta)
        h = self.encoder(x)
        return self.decoder(h)

class Discriminator(chainer.Chain):
    def __init__(self, args, pretrained_model=None, pretrained_off=False):
        super(Discriminator, self).__init__()
        self.n_down_layers = args.dis_ndown
        self.activation = args.dis_activation
        self.wgan = args.dis_wgan
        self.chs = args.dis_chs
        self.attention = args.dis_attention
        pad = args.dis_ksize//2
        dis_out = 2 if args.dis_reg_weighting>0 else 1  ## weighted discriminator
        if pretrained_model:
            self.base=pretrained_model
            self.update_base = False

        with self.init_scope():
            if hasattr(args,'dis_pretrained') and args.dis_pretrained and not pretrained_off:
                self.pretrained = True
                if "resnet" in args.dis_pretrained:
                    if args.dis_ndown==1:
                        self.layers = ['conv1']
                    else:
                        self.layers = ['res{}'.format(args.dis_ndown)]
                else: ## VGG16
                    if args.dis_ndown < 3:
                        self.layers = ['conv{}_2'.format(args.dis_ndown)]
                    else:
                        self.layers = ['conv{}_3'.format(args.dis_ndown)]
                if pretrained_model is None:
                    if "resnet" in args.dis_pretrained:
                        self.base = L.ResNet50Layers()
                    else:
                        self.base = L.VGG16Layers()
#                print(self.chs, self.layers)
            else:  ## new network
                self.pretrained = False
                self.c0 = CBR(None, self.chs[0], ksize=args.dis_ksize, pad=pad, norm='none', 
                            sample=args.dis_sample, activation=args.dis_activation,dropout=args.dis_dropout, equalised=args.eqconv,senet=args.senet) #separable=args.spconv)
                for i in range(1, len(self.chs)):
                    setattr(self, 'c' + str(i),
                            CBR(self.chs[i-1], self.chs[i], ksize=args.dis_ksize, pad=pad, norm=args.dis_norm,
                                sample=args.dis_down, activation=args.dis_activation, dropout=args.dis_dropout, equalised=args.eqconv, separable=args.spconv, senet=args.senet))
            ## common                                
            self.csl = CBR(self.chs[-1], 2*self.chs[-1], ksize=args.dis_ksize, pad=pad, norm=args.dis_norm, sample='none', activation=args.dis_activation, dropout=args.dis_dropout, equalised=args.eqconv, separable=args.spconv, senet=args.senet)
            if self.attention:
                setattr(self, 'a',  NonLocalBlock(2*self.chs[-1]))
            if self.wgan:
                self.fc1 = LBR(1024, activation='relu')
                self.fc2 = L.Linear(None, 1)
            else:
                self.cl = CBR(2*self.chs[-1], dis_out, ksize=args.dis_ksize, pad=pad, norm='none', sample='none', activation='none', dropout=False, equalised=args.eqconv, separable=args.spconv, senet=args.senet)

    def __call__(self, x):
        if self.pretrained:
            if x.shape[1]==1:
                h = F.concat([x,x,x])
            else:
                h = x
            if self.update_base:
                zz = self.base(h, layers=self.layers)
            else:
                with chainer.using_config('train', False) and chainer.no_backprop_mode():
                    zz = self.base(h, layers=self.layers)
            h = zz[self.layers[-1]]
        else:
            h = self.c0(x)
            for i in range(1, len(self.chs)):
                h = getattr(self, 'c' + str(i))(h)

        h = self.csl(h)
        if self.attention:
            h = getattr(self, 'a')(h)
        if self.wgan:
#            h = F.average(h, axis=(2, 3))   # global pooling
            h = self.fc1(h)
            h = self.fc2(h)
        else:
            h = self.cl(h)
        return h
