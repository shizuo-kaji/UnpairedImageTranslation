#!/usr/bin/env python

import chainer.functions as F
import numpy as np

dtypes = {
    'fp16': np.float16,
    'fp32': np.float32
}
activation = {
    'relu': F.relu,
    'lrelu': lambda x: F.leaky_relu(x, slope=0.2),
    'tanh': F.tanh,
    'none': None,
}
