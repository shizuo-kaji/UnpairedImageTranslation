#!/usr/bin/env python

import chainer.functions as F
import numpy as np
from chainer import optimizers
import functools
import chainer.links as L

optim = {
    'SGD': optimizers.MomentumSGD,
    'Momentum': optimizers.MomentumSGD,
    'AdaDelta': optimizers.AdaDelta,
    'AdaGrad': optimizers.AdaGrad,
    'Adam': functools.partial(optimizers.Adam, beta1=0.1),
    'AdaBound': functools.partial(optimizers.Adam, beta1=0.1, adabound=True),
    'RMSprop': optimizers.RMSprop,
    'NesterovAG': optimizers.NesterovAG,
}
try:
    from eve import Eve
    optim['Eve'] = functools.partial(Eve, beta1=0.1)
except:
    pass
try:
    from lbfgs import LBFGS
    optim['LBFGS'] = functools.partial(LBFGS, stack_size=10)
except:
    pass
    
dtypes = {
    'fp16': np.float16,
    'fp32': np.float32
}

activation_func = {
    'relu': F.relu,
    'lrelu': lambda x: F.leaky_relu(x, slope=0.2),
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'none': None,
}

unettype = ['none','concat','add','conv']

def feature_vector_normalization(x, eps=1e-8):
    alpha = 1.0 / F.sqrt(F.mean(x*x, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x

norm_layer = {
    'none': lambda x: F.identity,
    'batch': functools.partial(L.BatchNormalization, use_gamma=False, use_beta=False),
    'batch_aff': functools.partial(L.BatchNormalization, use_gamma=True, use_beta=True),
    'layer': L.LayerNormalization,
    'rbatch': functools.partial(L.BatchRenormalization, use_gamma=False, use_beta=True),
    'group': functools.partial(L.GroupNormalization, 1),   ## currently very slow
    'fnorm': lambda x: feature_vector_normalization
}
try:
    from instance_normalization import InstanceNormalization
    norm_layer['instance'] = functools.partial(InstanceNormalization, use_gamma=False, use_beta=False)
except:
    pass
