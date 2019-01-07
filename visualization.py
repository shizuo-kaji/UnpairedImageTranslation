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
    xp = chainer.cuda.get_array_module(var.data)
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
        self.single_encoder = params['single_encoder']
        self.count = 0

    def evaluate(self):
        batch_x =  self._iterators['main'].next()
        batch_y =  self._iterators['testB'].next()
        models = self._targets
        if self.eval_hook:
            self.eval_hook(self)

        fig = plt.figure(figsize=(9, 3 * (len(batch_x)+ len(batch_y))))
        gs = gridspec.GridSpec(len(batch_x)+ len(batch_y), 3, wspace=0.1, hspace=0.1)

        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if len(models)>2:
                    x_y = models['dec_y'](models['enc_x'](x))        
                    if self.single_encoder:
                        x_y_x = models['dec_x'](models['enc_x'](x_y))
                    else:
                        x_y_x = models['dec_x'](models['enc_x'](x))    ## autoencoder
                        #x_y_x = models['dec_x'](models['enc_y'](x_y))       
                else:
                    x_y = models['gen_g'](x)
                    x_y_x = models['gen_f'](x_y)

#        for i, var in enumerate([x, x_y]):
        for i, var in enumerate([x, x_y, x_y_x]):
            imgs = postprocess(var).astype(np.float32)
            for j in range(len(imgs)):
                ax = fig.add_subplot(gs[j,i])
                if imgs[j].shape[2]==1:
                    ax.imshow(imgs[j,:,:,0], interpolation='none',cmap='gray',vmin=0,vmax=1)
                else:
                    ax.imshow(imgs[j], interpolation='none',vmin=0,vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if len(models)>2:
                    if self.single_encoder:
                        y_x = models['dec_x'](models['enc_x'](y))
                    else:
                        y_x = models['dec_x'](models['enc_y'](y))
#                    y_x_y = models['dec_y'](models['enc_y'](y))   ## autoencoder
                    y_x_y = models['dec_y'](models['enc_x'](y_x))
                else:   # (gen_g, gen_f)
                    y_x = models['gen_f'](y)
                    y_x_y = models['gen_g'](y_x)

#        for i, var in enumerate([y, y_y]):
        for i, var in enumerate([y, y_x, y_x_y]):
            imgs = postprocess(var).astype(np.float32)
            for j in range(len(imgs)):
                ax = fig.add_subplot(gs[j+len(batch_x),i])
                if imgs[j].shape[2]==1:
                    ax.imshow(imgs[j,:,:,0], interpolation='none',cmap='gray',vmin=0,vmax=1)
                else:
                    ax.imshow(imgs[j], interpolation='none',vmin=0,vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

        gs.tight_layout(fig)
        plt.savefig(os.path.join(self.vis_out,'count{:0>4}.jpg'.format(self.count)), dpi=200)
        self.count += 1
        plt.close()

        cycle_y_l1 = F.mean_absolute_error(y,y_x_y)
        cycle_y_l2 = F.mean_squared_error(y,y_x_y)
        cycle_x_l1 = F.mean_absolute_error(x,x_y_x)
        id_xy_grad = losses.loss_grad(x,x_y)
        id_xy_l1 = F.mean_absolute_error(x,x_y)

        result = {"myval/cycle_y_l1":cycle_y_l1, "myval/cycle_y_l2":cycle_y_l2, "myval/cycle_x_l1":cycle_x_l1, 
            "myval/id_xy_grad":id_xy_grad,
            "myval/id_xy_l1":id_xy_l1}
        return result

## obsolete
def visualize(models,test_image_folder, test_A_iter, test_B_iter):
    @chainer.training.make_extension()
    def visualization(trainer):
        updater = trainer.updater
#        batch_x = updater.get_iterator('main').next()
#        batch_y = updater.get_iterator('train_B').next()
        batch_x = test_A_iter.next()
        batch_y = test_B_iter.next()
        batchsize = len(batch_x)

        fig = plt.figure(figsize=(9, 6 * batchsize))
        gs = gridspec.GridSpec(2* batchsize, 2, wspace=0.1, hspace=0.1)

        x = Variable(updater.converter(batch_x, updater.device))
        y = Variable(updater.converter(batch_y, updater.device))

        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if len(models)==3:
                    # models = (enc_x, enc_y, dec_y)
                    x_y = models[2](models[0](x))          #  
                else:   # (gen_g, gen_f)
                    x_y = models[0](x)
    #                x_y_x = models[1](x_y)

        for i, var in enumerate([x, x_y]):
#        for i, var in enumerate([x, x_y, x_y_x]):
            imgs = postprocess(var)
            for j in range(batchsize):
                ax = fig.add_subplot(gs[j * 2,i])
                ax.imshow(imgs[j], interpolation='none',cmap='gray',vmin=0,vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

        with chainer.using_config('train', False):
            with chainer.function.no_backprop_mode():
                if len(models)==3:
                    # models = (enc_x, enc_y, dec_y)
                    y_z = models[1](y)
                    y_y = models[2](y_z)
                else:   # (gen_g, gen_f)
                    y_x = models[1](y)
                    y_y = models[0](y_x)

        for i, var in enumerate([y, y_y]):
#        for i, var in enumerate([y, y_x, y_x_y]):
            imgs = postprocess(var)
            for j in range(batchsize):
                ax = fig.add_subplot(gs[j * 2 + 1,i])
                ax.imshow(imgs[j], interpolation='none',cmap='gray',vmin=0,vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

        gs.tight_layout(fig)
        plt.savefig(os.path.join(test_image_folder,'epoch{:d}.jpg'.format(updater.epoch)), dpi=200)
        plt.close()

    return visualization
