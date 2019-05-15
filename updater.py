import random
import time
import gc

import chainer
import chainer.functions as F
from chainer import Variable,cuda
from chainer.links import VGG16Layers
import losses

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self._iter = 0
        self.xp = self.gen_g.xp
        self._buffer_x = losses.ImagePool(50 * self.args.batch_size)
        self._buffer_y = losses.ImagePool(50 * self.args.batch_size)
        self.init_alpha = self.get_optimizer('opt_g').alpha
        self.report_start = self.args.warmup*10 ## start reporting
        if self.args.lambda_identity_x > 0 or self.args.lambda_identity_y > 0:
            self.vgg = VGG16Layers()  # for perceptual loss
            self.vgg.to_gpu()

    def update_core(self):
        opt_g = self.get_optimizer('opt_g')
        opt_f = self.get_optimizer('opt_f')
        opt_x = self.get_optimizer('opt_x')
        opt_y = self.get_optimizer('opt_y')

        self._iter += 1
        # learning rate decay: TODO: weight_decay_rate of AdamW
        if self.is_new_epoch and self.epoch >= self.args.lrdecay_start:
            decay_step = self.init_alpha / self.args.lrdecay_period
            if opt_g.alpha > decay_step:
                opt_g.alpha -= decay_step
            if opt_f.alpha > decay_step:
                opt_f.alpha -= decay_step
            if opt_x.alpha > decay_step:
                opt_x.alpha -= decay_step
            if opt_y.alpha > decay_step:
                opt_y.alpha -= decay_step

        # get mini-batch
        batch_x = self.get_iterator('main').next()
        batch_y = self.get_iterator('train_B').next()
        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        ### generator
        # X => Y => X
        x_y = self.gen_g(losses.add_noise(x, sigma=self.args.noise))
        if self.args.conditional_discriminator:
            x_y_copy = Variable(self._buffer_y.query(F.concat([x,x_y]).data))
        else:
            x_y_copy = Variable(self._buffer_y.query(x_y.data))
        x_y_x = self.gen_f(x_y)
        loss_cycle_x = losses.loss_avg(x_y_x, x, ksize=self.args.cycle_ksize, norm='l1')

        loss_gen_g_adv = 0
        if self.args.gen_start<self._iter:
            if self.args.conditional_discriminator:
                if self.args.wgan:
                    loss_gen_g_adv = -F.average(self.dis_y(F.concat([x,x_y])))
                else:
                    loss_gen_g_adv = losses.loss_func_comp(self.dis_y(F.concat([x,x_y])),1.0)
            else:
                if self.args.wgan:
                    loss_gen_g_adv = -F.average(self.dis_y(x_y))
                else:
                    loss_gen_g_adv = losses.loss_func_comp(self.dis_y(x_y),1.0)                

        # Y => X => Y
        loss_gen_f_adv = 0
        y_x = self.gen_f(losses.add_noise(y, sigma=self.args.noise))  # noise injection
        if self.args.conditional_discriminator:
            y_x_copy = Variable(self._buffer_x.query(F.concat([y,y_x]).data))
        else:
            y_x_copy = Variable(self._buffer_x.query(y_x.data))
        y_x_y = self.gen_g(y_x)
        loss_cycle_y = losses.loss_avg(y_x_y, y, ksize=self.args.cycle_ksize, norm='l1')
        if self.args.gen_start<self._iter:
            if self.args.conditional_discriminator:
                if self.args.wgan:
                    loss_gen_f_adv = -F.average(self.dis_x(F.concat([y,y_x])))
                else:
                    loss_gen_f_adv = losses.loss_func_comp(self.dis_x(F.concat([y,y_x])),1.0)
            else:
                if self.args.wgan:
                    loss_gen_f_adv = -F.average(self.dis_x(y_x))
                else:
                    loss_gen_f_adv = losses.loss_func_comp(self.dis_x(y_x),1.0)

        ## total loss for generators
        loss_gen = (self.args.lambda_dis_y * loss_gen_g_adv + self.args.lambda_dis_x * loss_gen_f_adv) + (self.args.lambda_A * loss_cycle_x + self.args.lambda_B * loss_cycle_y)

        ## idempotence: f shouldn't change x
        if self.args.lambda_idempotence > 0:             
            loss_idem_x = F.mean_absolute_error(y_x,self.gen_f(y_x))
            loss_idem_y = F.mean_absolute_error(x_y,self.gen_g(x_y))
            loss_gen = loss_gen + self.args.lambda_idempotence * (loss_idem_x + loss_idem_y)
            if self.report_start<self._iter:
                chainer.report({'loss_idem': loss_idem_x}, self.gen_f) 
                chainer.report({'loss_idem': loss_idem_y}, self.gen_g)            
        if self.args.lambda_domain > 0:             
            loss_dom_x = F.mean_absolute_error(x,self.gen_f(x))
            loss_dom_y = F.mean_absolute_error(y,self.gen_g(y))
            if self._iter < self.args.warmup:
                loss_gen = loss_gen + max(self.args.lambda_domain,1.0) * (loss_dom_x + loss_dom_y)
            else:
                loss_gen = loss_gen + self.args.lambda_domain * (loss_dom_x + loss_dom_y)
            if self.report_start<self._iter:
                chainer.report({'loss_dom': loss_dom_x}, self.gen_f) 
                chainer.report({'loss_dom': loss_dom_y}, self.gen_g)

        ## images before/after conversion should look similar in terms of perceptual loss
        if self.args.lambda_identity_x > 0:
            loss_id_x = losses.loss_perceptual(x,x_y,self.vgg)
            loss_gen = loss_gen + self.args.lambda_identity_x * loss_id_x
            if self.report_start<self._iter:
                chainer.report({'loss_id': 1e-3*loss_id_x}, self.gen_g)
        if self.args.lambda_identity_y > 0:
            loss_id_y = losses.loss_perceptual(y,y_x,self.vgg)
            loss_gen = loss_gen + self.args.lambda_identity_y * loss_id_y
            if self.report_start<self._iter:
                chainer.report({'loss_id': 1e-3*loss_id_y}, self.gen_f)

        ## warm-up
        if self._iter < self.args.warmup:
            loss_gen = loss_gen + F.mean_squared_error(x,self.gen_f(x))
            loss_gen = loss_gen + F.mean_squared_error(y,self.gen_g(y))
#            loss_gen = loss_gen + losses.loss_avg(y,y_x, ksize=self.args.id_ksize, norm='l2')
#            loss_gen = loss_gen + losses.loss_avg(x,x_y, ksize=self.args.id_ksize, norm='l2')

        ## background should be preserved 
        if self.args.lambda_air > 0:
            loss_air_x = losses.loss_range_comp(x,x_y,0.9,norm='l2')
            loss_air_y = losses.loss_range_comp(y,y_x,0.9,norm='l2')
            loss_gen = loss_gen + self.args.lambda_air * (loss_air_x+loss_air_y)
            if self.report_start<self._iter:
                chainer.report({'loss_air': 0.1*loss_air_x}, self.gen_g)
                chainer.report({'loss_air': 0.1*loss_air_y}, self.gen_f)

        ## comparison of images before/after conversion in the gradient domain
        if self.args.lambda_grad > 0:
            loss_grad_x = losses.loss_grad(x,x_y,self.args.grad_norm)
            loss_grad_y = losses.loss_grad(y,y_x,self.args.grad_norm)
            loss_gen = loss_gen + self.args.lambda_grad * (loss_grad_x + loss_grad_y)
            if self.report_start<self._iter:
                chainer.report({'loss_grad': loss_grad_x}, self.gen_g)
                chainer.report({'loss_grad': loss_grad_y}, self.gen_f)
        ## total variation
        if self.args.lambda_tv > 0:
            loss_tv = losses.total_variation(x_y, self.args.tv_tau)
            loss_gen = loss_gen + self.args.lambda_tv * loss_tv
            if self.report_start<self._iter:
                chainer.report({'loss_tv': loss_tv}, self.gen_g)

        if self.report_start<self._iter:
            chainer.report({'loss_cycle_x': loss_cycle_x}, self.gen_f)
            chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
            chainer.report({'loss_cycle_y': loss_cycle_y}, self.gen_g)
            chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)

        self.gen_f.cleargrads()
        self.gen_g.cleargrads()
        loss_gen.backward()
        opt_f.update()
        opt_g.update()

        ### discriminator
        for t in range(self.args.n_critics):
            if self.args.wgan: ## synthesised -, real +
                loss_dis_y_fake = F.average(self.dis_y(x_y_copy))
                eps = self.xp.random.uniform(0, 1, size=len(batch_y)).astype(self.xp.float32)[:, None, None, None]
                if self.args.conditional_discriminator:
                    loss_dis_y_real = -F.average(self.dis_y(F.concat([x,y])))
                    y_mid = eps * F.concat([y,x]) + (1.0 - eps) * x_y_copy
                else:
                    loss_dis_y_real = -F.average(self.dis_y(y))
                    y_mid = eps * y + (1.0 - eps) * x_y_copy
                # gradient penalty
                gd_y, = chainer.grad([self.dis_y(y_mid)], [y_mid], enable_double_backprop=True)
                gd_y = F.sqrt(F.batch_l2_norm_squared(gd_y) + 1e-6)
                loss_dis_y_gp = F.mean_squared_error(gd_y, self.xp.ones_like(gd_y.data))
                
                if self.report_start<self._iter:
                    chainer.report({'loss_real': loss_dis_y_real}, self.dis_y)
                    chainer.report({'loss_fake': loss_dis_y_fake}, self.dis_y)
                    chainer.report({'loss_gp': self.args.lambda_wgan_gp * loss_dis_y_gp}, self.dis_y)
                loss_dis_y = (loss_dis_y_real + loss_dis_y_fake + self.args.lambda_wgan_gp * loss_dis_y_gp)
                self.dis_y.cleargrads()
                loss_dis_y.backward()
                opt_y.update()

                ## discriminator for B=>A
                loss_dis_x_fake = F.average(self.dis_x(y_x_copy))
                if self.args.conditional_discriminator:
                    loss_dis_x_real = -F.average(self.dis_x(losses.add_noise(F.concat([y,x]), sigma=self.args.noise)))
                    x_mid = eps * F.concat([x,y]) + (1.0 - eps) * y_x_copy
                else:
                    loss_dis_x_real = -F.average(self.dis_x(x))
                    x_mid = eps * x + (1.0 - eps) * y_x_copy
                # gradient penalty
                gd_x, = chainer.grad([self.dis_x(x_mid)], [x_mid], enable_double_backprop=True)
                gd_x = F.sqrt(F.batch_l2_norm_squared(gd_x) + 1e-6)
                loss_dis_x_gp = F.mean_squared_error(gd_x, self.xp.ones_like(gd_x.data))

                if self.report_start<self._iter:
                    chainer.report({'loss_real': loss_dis_x_real}, self.dis_x)
                    chainer.report({'loss_fake': loss_dis_x_fake}, self.dis_x)
                    chainer.report({'loss_gp': self.args.lambda_wgan_gp *loss_dis_x_gp}, self.dis_x)
                loss_dis_x = (loss_dis_x_real + loss_dis_x_fake + self.args.lambda_wgan_gp * loss_dis_x_gp)
                self.dis_x.cleargrads()
                loss_dis_x.backward()
                opt_x.update()
                
            else:
                ## discriminator for A=>B (real:1, fake:0)
                loss_dis_y_fake = losses.loss_func_comp(self.dis_y(x_y_copy),0.0, self.args.dis_jitter)
                if self.args.conditional_discriminator:
                    loss_dis_y_real = losses.loss_func_comp(self.dis_y(F.concat([x,y])),1.0, self.args.dis_jitter)
                else:
                    loss_dis_y_real = losses.loss_func_comp(self.dis_y(y),1.0, self.args.dis_jitter)
                loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
                self.dis_y.cleargrads()
                loss_dis_y.backward()
                opt_y.update()

                ## discriminator for B=>A
                loss_dis_x_fake = losses.loss_func_comp(self.dis_x(y_x_copy),0.0, self.args.dis_jitter)
                if self.args.conditional_discriminator:
                    loss_dis_x_real = losses.loss_func_comp(self.dis_x(F.concat([y,x])),1.0, self.args.dis_jitter)
                else:
                    loss_dis_x_real = losses.loss_func_comp(self.dis_x(x),1.0, self.args.dis_jitter)
                loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
                self.dis_x.cleargrads()
                loss_dis_x.backward()
                opt_x.update()

                if self.report_start<self._iter:
                    chainer.report({'loss_real': loss_dis_x_real}, self.dis_x)
                    chainer.report({'loss_fake': loss_dis_x_fake}, self.dis_x)
                    chainer.report({'loss_real': loss_dis_y_real}, self.dis_y)
                    chainer.report({'loss_fake': loss_dis_y_fake}, self.dis_y)

            # prepare next images
            if(t<self.args.n_critics-1):
                x_y_copy = Variable(self.xp.concatenate(random.sample(self._buffer_y.images,self.args.batch_size)))
                y_x_copy = Variable(self.xp.concatenate(random.sample(self._buffer_x.images,self.args.batch_size)))
                batch_x = self.get_iterator('main').next()
                batch_y = self.get_iterator('train_B').next()
                x = Variable(self.converter(batch_x, self.device))
                y = Variable(self.converter(batch_y, self.device))

        

