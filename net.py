import functools

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from consts import activation_func, norm_layer

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
        self.pad = pad
        if equalised:
            w = chainer.initializers.Normal(1.0) # equalized learning rate
        else:
            w = chainer.initializers.HeNormal()
        bias = chainer.initializers.Zero()
        self.ksize = ksize
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            if self.separable:
                self.depthwise = L.Convolution2D(in_ch, in_ch, ksize, stride, initialW=w, nobias=True, groups=in_ch)
                self.pointwise = L.Convolution2D(in_ch, out_ch, 1, 1, initialW=w, nobias=nobias, initial_bias=bias)
            else:
                self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, initialW=w, nobias=nobias, initial_bias=bias)
            if self.senet and out_ch>15:
                self.se = SEBlock(out_ch)
    def __call__(self, x):
        if self.pad_type=='reflect':
            h = F.pad(x,[[0,0],[0,0],[self.pad,self.pad],[self.pad,self.pad]],mode='reflect')
        else:
            h = F.pad(x,[[0,0],[0,0],[self.pad,self.pad],[self.pad,self.pad]],mode='constant',constant_values=0) 
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

### the num of input channles should be divisible by 4
# obsolite: use F.depth2space
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


class ResBlock(chainer.Chain):
    def __init__(self, ch, norm='instance', activation='relu', equalised=False, separable=False, skip_conv=False):
        super(ResBlock, self).__init__()
        self.activation = activation_func[activation]
        nobias = True if 'batch' in norm or 'instance' in norm else False
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
        return self.activation(h + self.cs(x))


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, ksize=3, pad=1, norm='instance',
                 sample='down', activation='relu', dropout=False, equalised=False, separable=False, senet=False):
        super(CBR, self).__init__()
        self.activation = activation_func[activation]
        self.dropout = dropout
        self.sample = sample
        nobias = True if 'batch' in norm or 'instance' in norm else False
        if 'pixsh' in sample:
            ch0 = int(ch0/4)

        with self.init_scope():
            if sample == 'down':
                self.c = EqualizedConv2d(ch0, ch1, ksize, 2, pad, equalised=equalised, nobias=nobias, separable=separable,senet=senet)
            elif sample == 'none-7':
                self.c = EqualizedConv2d(ch0, ch1, 7, 1, 3, pad_type='reflect', equalised=equalised, nobias=nobias, separable=separable,senet=senet) 
            elif sample == 'deconv':
                self.c = EqualizedDeconv2d(ch0, ch1, ksize, 2, pad, equalised=equalised, nobias=nobias, separable=separable)
            else: ## maxpool,avgpool,resize,unpool
                self.c = EqualizedConv2d(ch0, ch1, ksize, 1, pad, equalised=equalised, nobias=nobias, separable=separable,senet=senet)
            self.norm = norm_layer[norm](ch1)
            if '_res' in sample:
                self.normr = norm_layer[norm](ch1)
                self.cr = EqualizedConv2d(ch1, ch1, ksize, 1, pad, equalised=equalised, nobias=nobias, separable=separable,senet=senet)
                self.cskip = EqualizedConv2d(ch0, ch1, 1, 1, 0, equalised=equalised, separable=True, nobias=False)

    def __call__(self, x):
        if self.sample in ['maxpool_res','avgpool_res']:
            h = self.activation(self.norm(self.c(x)))
            h = self.normr(self.cr(h))
            if self.sample == 'maxpool_res':
                h = F.max_pooling_2d(h, 2, 2, 0)
                h = h + F.max_pooling_2d(self.cskip(x), 2, 2, 0)
            elif self.sample == 'avgpool_res':
                h = F.average_pooling_2d(h, 2, 2, 0)
                h = h + F.average_pooling_2d(self.cskip(x), 2, 2, 0)                
        elif 'resize' in self.sample:
            H,W = x.data.shape[2:]
            h0 = F.resize_images(x, (2*H,2*W))
            h = self.norm(self.c(h0))
            if self.sample == 'resize_res':
                h = self.activation(h)
                h = self.cskip(h0) + self.normr(self.cr(h))
        elif 'pixsh' in self.sample:
            h0 = F.depth2space(x, 2)
            h = self.norm(self.c(h0))
            if self.sample == 'pixsh_res':
                h = self.activation(h)
                h = self.cskip(h0) + self.normr(self.cr(h))
        elif 'unpool' in self.sample:
            h0 = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = self.norm(self.c(h0))
            if self.sample == 'unpool_res':
                h = self.activation(h)
                h = self.cskip(h0) + self.normr(self.cr(h))
        else:
            if self.sample == 'maxpool':
                h = self.c(x)
                h = F.max_pooling_2d(h, 2, 2, 0)
            elif self.sample == 'avgpool':
                h = self.c(x)
                h = F.average_pooling_2d(h, 2, 2, 0)
            else:
                h = self.c(x)
            h = self.norm(h)
        if self.dropout:
            h = F.dropout(h, ratio=self.dropout)
        if self.activation is not None:
            h = self.activation(h)
        return h

class LBR(chainer.Chain):
    def __init__(self, height, width, ch, norm='none', activation='tanh', dropout=False):
        super(LBR, self).__init__()
        self.activation = activation_func[activation]
        nobias = True if 'batch' in norm or 'instance' in norm else False
        self.dropout = dropout
        self.ch = ch
        self.width = width
        self.height = height
        with self.init_scope():
            self.l0 = L.Linear(ch*height*width,ch*height*width, nobias=nobias)
            self.norm = norm_layer[norm](ch*height*width)

    def __call__(self, x):
        h = self.l0(x)
        h = self.norm(h)
        if self.dropout:
            h = F.dropout(h, ratio=self.dropout)
        if self.activation is not None:
            h = self.activation(h)
        return F.reshape(h,x.shape)

class Encoder(chainer.Chain):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_resblock = (args.gen_nblock+1) // 2 # half for Enc and half for Dec
        self.chs = args.gen_chs
        if hasattr(args,'unet'):
            self.unet = args.unet
        else:
            self.unet = 'none'
        self.nfc = args.gen_fc
        with self.init_scope():
            for i in range(args.gen_fc):
                setattr(self, 'l' + str(i), LBR(args.crop_height,args.crop_width,args.ch, activation=args.gen_fc_activation))
            # nn.ReflectionPad2d in original
            self.c0 = CBR(args.ch, self.chs[0], norm=args.gen_norm, sample=args.gen_sample, activation=args.gen_activation, equalised=args.eqconv)
            for i in range(1,len(self.chs)):
                setattr(self, 'd' + str(i), CBR(self.chs[i-1], self.chs[i], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_down, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            if self.unet=='conv':
                for i in range(len(self.chs)):
                    setattr(self, 's' + str(i), CBR(self.chs[i], 4, ksize=3, norm=args.gen_norm, sample='none', activation='lrelu', dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            for i in range(self.n_resblock):
                setattr(self, 'r' + str(i), ResBlock(self.chs[-1], norm=args.gen_norm, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))

    def __call__(self, x):
        h = x
        for i in range(self.nfc):
            h=getattr(self, 'l' + str(i))(h)
        e = self.c0(x)
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
#            print(h[-1].data.shape)
        for i in range(self.n_resblock):
            e = getattr(self, 'r' + str(i))(e)
        h.append(e)
        return h

class Decoder(chainer.Chain):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_resblock = args.gen_nblock // 2 # half for Enc and half for Dec
        self.chs = args.gen_chs
        if hasattr(args,'unet'):
            self.unet = args.unet
        else:
            self.unet = 'none'
        if self.unet=='concat':
            up_chs = [2*self.chs[i] for i in range(len(self.chs))]
        elif self.unet in ['add','none']:
            up_chs = self.chs
        elif self.unet=='conv':
            up_chs = [self.chs[i]+4 for i in range(len(self.chs))]                
        if hasattr(args,'noise_z'):
            self.noise_z = args.noise_z
        else:
            self.noise_z = 0
        with self.init_scope():
            for i in range(self.n_resblock):
                setattr(self, 'r' + str(i), ResBlock(self.chs[-1], norm=args.gen_norm, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            for i in range(1,len(self.chs)):
                setattr(self, 'ua' + str(i), CBR(up_chs[-i], self.chs[-i-1], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_up, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            setattr(self, 'ua'+str(len(self.chs)),CBR(up_chs[0], args.ch, norm='none', sample=args.gen_sample, activation=args.gen_out_activation, equalised=args.eqconv, separable=args.spconv))

    def __call__(self, h):
        e = h[-1]
        if chainer.config.train and self.noise_z>0:
            e.data += self.noise_z * e.xp.random.randn(*e.data.shape, dtype=e.dtype)
        for i in range(self.n_resblock):
            e = getattr(self, 'r' + str(i))(e)
        for i in range(1,len(self.chs)+1):
            if self.unet in ['conv','concat']:
                e = getattr(self, 'ua' + str(i))(F.concat([e,h[-i-1]]))
            elif self.unet=='add':
                e = getattr(self, 'ua' + str(i))(e+h[-i-1])
            else:
                e = getattr(self, 'ua' + str(i))(e)
        return e

class Generator(chainer.Chain):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.n_resblock = args.gen_nblock
        self.chs = args.gen_chs
        if hasattr(args,'unet'):
            self.unet = args.unet
        else:
            self.unet = 'none'
        if self.unet=='concat':
            up_chs = [2*self.chs[i] for i in range(len(self.chs))]
        elif self.unet in ['add','none']:
            up_chs = self.chs
        elif self.unet=='conv':
            up_chs = [self.chs[i]+4 for i in range(len(self.chs))]                
        if hasattr(args,'noise_z'):
            self.noise_z = args.noise_z
        else:
            self.noise_z = 0
        self.nfc = args.gen_fc
        with self.init_scope():
            for i in range(args.gen_fc):
                setattr(self, 'l' + str(i), LBR(args.crop_height,args.crop_width,args.ch, activation=args.gen_fc_activation))
            self.c0 = CBR(args.ch, self.chs[0], norm=args.gen_norm, sample=args.gen_sample, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv)
            for i in range(1,len(self.chs)):
                setattr(self, 'd' + str(i), CBR(self.chs[i-1], self.chs[i], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_down, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            if self.unet=='conv':
                for i in range(len(self.chs)):
                    setattr(self, 's' + str(i), CBR(self.chs[i], 4, ksize=3, norm=args.gen_norm, sample='none', activation='lrelu', dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            for i in range(self.n_resblock):
                setattr(self, 'r' + str(i), ResBlock(self.chs[-1], norm=args.gen_norm, activation=args.gen_activation, equalised=args.eqconv, separable=args.spconv))
            for i in range(1,len(self.chs)):
                setattr(self, 'ua' + str(i), CBR(up_chs[-i], self.chs[-i-1], ksize=args.gen_ksize, norm=args.gen_norm, sample=args.gen_up, activation=args.gen_activation, dropout=args.gen_dropout, equalised=args.eqconv, separable=args.spconv))
            setattr(self, 'ua'+str(len(self.chs)),CBR(up_chs[0], args.ch, norm='none', sample=args.gen_sample, activation=args.gen_out_activation, equalised=args.eqconv, separable=args.spconv))

    def __call__(self, x):
        h = x
        for i in range(self.nfc):
            h=getattr(self, 'l' + str(i))(h)
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
#            print(h[-1].data.shape)
        for i in range(self.n_resblock):
            e = getattr(self, 'r' + str(i))(e)
            ## add noise
            if chainer.config.train and self.noise_z>0 and i == self.n_resblock//2:
                e.data += self.noise_z * e.xp.random.randn(*e.data.shape, dtype=e.dtype)
        for i in range(1,len(self.chs)+1):
            if self.unet in ['conv','concat']:
                e = getattr(self, 'ua' + str(i))(F.concat([e,h[-i]]))
            elif self.unet=='add':
                e = getattr(self, 'ua' + str(i))(e+h[-i])
            else:
                e = getattr(self, 'ua' + str(i))(e)
        return e

class Discriminator(chainer.Chain):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_down_layers = args.dis_ndown
        self.activation = args.dis_activation
        self.wgan = args.dis_wgan
        self.chs = args.dis_chs
        dis_out = 2 if args.dis_reg_weighting>0 else 1  ## weighted discriminator
        with self.init_scope():
            self.c0 = CBR(None, self.chs[0], ksize=args.dis_ksize, norm='none', 
                          sample=args.dis_sample, activation=args.dis_activation,dropout=args.dis_dropout, equalised=args.eqconv,senet=args.senet) #separable=args.spconv)
            for i in range(1, len(self.chs)):
                setattr(self, 'c' + str(i),
                        CBR(self.chs[i-1], self.chs[i], ksize=args.dis_ksize, norm=args.dis_norm,
                            sample=args.dis_down, activation=args.dis_activation, dropout=args.dis_dropout, equalised=args.eqconv, separable=args.spconv, senet=args.senet))
            self.csl = CBR(self.chs[-1], 2*self.chs[-1], ksize=args.dis_ksize, norm=args.dis_norm, sample='none', activation=args.dis_activation, dropout=args.dis_dropout, equalised=args.eqconv, separable=args.spconv, senet=args.senet)
            if self.wgan:
                self.fc1 = L.Linear(None, 1024)
                self.fc2 = L.Linear(None, 1)
            else:
                self.cl = CBR(2*self.chs[-1], dis_out, ksize=args.dis_ksize, norm='none', sample='none', activation='none', dropout=False, equalised=args.eqconv, separable=args.spconv, senet=args.senet)

    def __call__(self, x):
        h = self.c0(x)
        for i in range(1, len(self.chs)):
            h = getattr(self, 'c' + str(i))(h)
        h = self.csl(h)
        if self.wgan:
            h = F.average(h, axis=(2, 3))  # Global pooling
            h = activation_func[self.activation](self.fc1(h))
            h = self.fc2(h)
        else:
            h = self.cl(h)
        return h
