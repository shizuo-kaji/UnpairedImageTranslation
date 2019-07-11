import os

import chainer
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from chainer import Variable,cuda
import numpy as np
import cupy as cp
import functools
import chainer.functions as F
import losses

# assume [0,1] input
def postprocess(var):
    img = var.data.get()
    img = (img + 1.0) / 2.0  # [0, 1)
    img = img.transpose(0, 2, 3, 1)
    return img

from chainer.training import extensions
from chainer import reporter as reporter_module

class VisEvaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        params = kwargs.pop('params')
        super(VisEvaluator, self).__init__(*args, **kwargs)
        self.vis_out = params['vis_out']
        self.slice = params['slice']
        if self.slice:
            self.num_s = len(self.slice)
        else:
            self.num_s = 1
        self.count = 0

    def evaluate(self):
        batch_x =  self._iterators['testA'].next()
        batch_y =  self._iterators['testB'].next()
        models = self._targets
        if self.eval_hook:
            self.eval_hook(self)

        fig = plt.figure(figsize=(9, 3 * self.num_s*(len(batch_x)+ len(batch_y))))
        gs = gridspec.GridSpec( self.num_s*(len(batch_x)+ len(batch_y)), 3, wspace=0.1, hspace=0.1)

        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if len(models)>2:
                    x_y = models['dec_y'](models['enc_x'](x))        
                    #x_y_x = models['dec_x'](models['enc_x'](x))    ## X => Z => X
                    x_y_x = models['dec_x'](models['enc_y'](x_y))       ## X => Y => X
                else:
                    x_y = models['gen_g'](x)
                    x_y_x = models['gen_f'](x_y)

#        for i, var in enumerate([x, x_y]):
        for i, var in enumerate([x, x_y, x_y_x]):
            imgs = postprocess(var).astype(np.float32)
            for j in range(len(imgs)):
                if self.slice != None:
                    for k in self.slice:
                        ax = fig.add_subplot(gs[j*len(self.slice)+k,i])
                        ax.imshow(imgs[j,:,:,k], interpolation='none',cmap='gray',vmin=0,vmax=1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                else:
                    ax = fig.add_subplot(gs[j,i])
                    ax.imshow(imgs[j], interpolation='none',vmin=0,vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])

        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if len(models)>2:
                    y_x = models['dec_x'](models['enc_y'](y))
                    #y_x_y = models['dec_y'](models['enc_y'](y))   ## Y => Z => Y
                    y_x_y = models['dec_y'](models['enc_x'](y_x)) ## Y => X => Y
                else:   # (gen_g, gen_f)
                    y_x = models['gen_f'](y)
                    y_x_y = models['gen_g'](y_x)

#        for i, var in enumerate([y, y_y]):
        for i, var in enumerate([y, y_x, y_x_y]):
            imgs = postprocess(var).astype(np.float32)
            for j in range(len(imgs)):
                if self.slice != None:
                    for k in self.slice:
                        ax = fig.add_subplot(gs[(j+len(batch_x))*len(self.slice)+k,i])
                        ax.imshow(imgs[j,:,:,k], interpolation='none',cmap='gray',vmin=0,vmax=1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                else:
                    ax = fig.add_subplot(gs[j+len(batch_x),i])
                    ax.imshow(imgs[j], interpolation='none',vmin=0,vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])

        gs.tight_layout(fig)
        plt.savefig(os.path.join(self.vis_out,'count{:0>4}.jpg'.format(self.count)), dpi=200)
        self.count += 1
        plt.close()

        cycle_y_l1 = F.mean_absolute_error(y,y_x_y)
#        cycle_y_l2 = F.mean_squared_error(y,y_x_y)
        cycle_x_l1 = F.mean_absolute_error(x,x_y_x)
#        id_xy_grad = losses.loss_grad(x,x_y)

        result = {"myval/cycle_y_l1":cycle_y_l1, "myval/cycle_x_l1":cycle_x_l1}
        return result

## obsolete
def visualize(models,test_image_folder, test_A_iter, test_B_iter):
    @chainer.training.make_extension()
    def visualization(trainer):
        updater = trainer.updater
    return visualization
