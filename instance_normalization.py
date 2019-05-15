# Instance Normalization in Chainer on top of Batch Normalization
# By M. Kozuki
# https://gist.github.com/crcrpar/6f1bc0937a02001f14d963ca2b86427a
#

from chainer import cuda
from chainer import functions
from chainer import links
from chainer.utils import argument
from chainer import variable
import numpy


class InstanceNormalization(links.BatchNormalization):

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=False, use_beta=False,
                 initial_gamma=None, initial_beta=None):
        # instance normalization is usually done without gamma and beta
        super(InstanceNormalization, self).__init__(
            size=size,
            decay=decay,
            eps=eps,
            dtype=dtype,
            use_gamma=use_gamma,
            use_beta=use_beta,
            initial_gamma=initial_gamma,
            initial_beta=initial_beta)

    def __call__(self, x, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        # reshape input x for instance normalization
        shape_org = x.shape
        B, C = shape_org[:2]
        shape_ins = (1, B * C) + shape_org[2:]
        x_reshaped = functions.reshape(x, shape_ins)

        with cuda.get_device_from_id(self._device_id):
            gamma = variable.Variable(self.xp.ones(
                self.avg_mean.shape, dtype=x.dtype))

        with cuda.get_device_from_id(self._device_id):
            beta = variable.Variable(self.xp.zeros(
                self.avg_mean.shape, dtype=x.dtype))

        gamma = functions.tile(gamma, (B,))
        beta = functions.tile(beta, (B,))
        mean = self.xp.tile(self.avg_mean, (B,))
        var = self.xp.tile(self.avg_var, (B,))

        # instance normalization is always done in training mode
        if finetune:
            self.N += 1
            decay = 1. - 1. / self.N
        else:
            decay = self.decay

        ret = functions.batch_normalization(
            x_reshaped, gamma, beta, eps=self.eps, running_mean=mean,
            running_var=var, decay=decay)

        self.avg_mean = mean.reshape(B, C).mean(axis=0)
        self.avg_var = var.reshape(B, C).mean(axis=0)

        # ret is normalized input x
        return functions.reshape(ret, shape_org)
