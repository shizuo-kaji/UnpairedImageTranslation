import random
import chainer
import chainer.functions as F
from chainer import Variable,cuda
import losses

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.enc_x, self.dec_x, self.enc_y, self.dec_y, self.dis_x, self.dis_y, self.dis_z = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.noise_decay = self.args.noise/self.args.lrdecay_period
        self.noise_z_decay = self.args.noise_z/self.args.lrdecay_period
        self._iter = 0
        self.xp = self.enc_x.xp
        self._buffer_xz = losses.ImagePool(50 * self.args.batch_size)
        self._buffer_y = losses.ImagePool(50 * self.args.batch_size)
        self._buffer_x = losses.ImagePool(50 * self.args.batch_size)
        self.init_alpha = self.get_optimizer('opt_enc_x').alpha
        self.report_start = self.args.warmup*10 ## start reporting
        if self.args.single_encoder:
            self.enc_y = self.enc_x

    def update_core(self):
        opt_enc_x = self.get_optimizer('opt_enc_x')
        opt_dec_x = self.get_optimizer('opt_dec_x')
        opt_enc_y = self.get_optimizer('opt_enc_y')
        opt_dec_y = self.get_optimizer('opt_dec_y')
        opt_x = self.get_optimizer('opt_x')
        opt_y = self.get_optimizer('opt_y')
        opt_z = self.get_optimizer('opt_z')
        self._iter += 1
        if self.is_new_epoch and self.epoch >= self.args.lrdecay_start:
            decay_step = self.init_alpha / self.args.lrdecay_period
            print('lr decay', decay_step)
            if opt_enc_x.alpha > decay_step:
                opt_enc_x.alpha -= decay_step
            if opt_dec_x.alpha > decay_step:
                opt_dec_x.alpha -= decay_step
            if opt_enc_y.alpha > decay_step:
                opt_enc_y.alpha -= decay_step
            if opt_dec_y.alpha > decay_step:
                opt_dec_y.alpha -= decay_step
            if opt_y.alpha > decay_step:
                opt_y.alpha -= decay_step
            if opt_x.alpha > decay_step:
                opt_x.alpha -= decay_step
            if opt_z.alpha > decay_step:
                opt_z.alpha -= decay_step
            self.args.noise -= self.noise_decay
            self.args.noise_z -= self.noise_z_decay

        # get mini-batch
        batch_x = self.get_iterator('main').next()
        batch_y = self.get_iterator('train_B').next()
        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        # to latent
        x_z = self.enc_x(losses.add_noise(x, sigma=self.args.noise))  # noise injection
        y_z = self.enc_y(losses.add_noise(y, sigma=self.args.noise))

        loss_gen = 0
        ## regularisation on the latent space
        if self.args.lambda_reg>0:
            loss_reg_enc_y = losses.loss_func_reg(y_z[-1],'l2')
            loss_reg_enc_x = losses.loss_func_reg(x_z[-1],'l2') 
            loss_gen = loss_gen + self.args.lambda_reg * (loss_reg_enc_x + loss_reg_enc_y)
            if self.report_start<self._iter:
                chainer.report({'loss_reg': loss_reg_enc_x}, self.enc_x)
                chainer.report({'loss_reg': loss_reg_enc_y}, self.enc_y)

        ## discriminator for the latent space: distribution of image of enc_x should look same as that of enc_y
        if self.args.lambda_dis_z>0 and self._iter>self.args.dis_z_start:
            x_z_copy = Variable(self._buffer_xz.query(x_z[-1].data))
            loss_enc_x_adv = losses.loss_func_comp(self.dis_z(x_z[-1]),1.0)
            loss_gen = loss_gen +  self.args.lambda_dis_z * loss_enc_x_adv
            if self.report_start<self._iter:
                chainer.report({'loss_adv': loss_enc_x_adv}, self.enc_x)

        # cycle for X=>Z=>X
        x_z[-1] = losses.add_noise(x_z[-1], sigma=self.args.noise_z)
        x_x = self.dec_x(x_z)
        loss_cycle_xzx = F.mean_absolute_error(x_x, x)
        if self.report_start<self._iter:
            chainer.report({'loss_cycle': loss_cycle_xzx}, self.enc_x)

        # cycle for Y=>Z=>Y
        y_z[-1] = losses.add_noise(y_z[-1], sigma=self.args.noise_z)
        y_y = self.dec_y(y_z)
        loss_cycle_yzy = F.mean_absolute_error(y_y, y)
        if self.report_start<self._iter:
            chainer.report({'loss_cycle': loss_cycle_yzy}, self.enc_y)

        loss_gen = loss_gen + self.args.lambda_A * loss_cycle_xzx + self.args.lambda_B * loss_cycle_yzy

        ## conversion
        x_y = self.dec_y(x_z)
        y_x = self.dec_x(y_z)

        # cycle for (X=>)Z=>Y=>Z
        x_y_z = self.enc_y(x_y)
        loss_cycle_zyz = F.mean_absolute_error(x_y_z[-1], x_z[-1])
        if self.report_start<self._iter:
            chainer.report({'loss_cycle': loss_cycle_zyz}, self.dec_y)
        # cycle for (Y=>)Z=>X=>Z
        y_x_z = self.enc_x(y_x)
        loss_cycle_zxz = F.mean_absolute_error(y_x_z[-1], y_z[-1])
        if self.report_start<self._iter:
            chainer.report({'loss_cycle': loss_cycle_zxz}, self.dec_x)

        loss_gen = loss_gen + self.args.lambda_A * loss_cycle_zxz + self.args.lambda_B * loss_cycle_zyz

        ## adversarial for Y
        if self.args.lambda_dis_y>0:
            if self.args.conditional_discriminator:
                x_y_copy = Variable(self._buffer_y.query(F.concat([x_y,x]).data))
                loss_dec_y_adv = losses.loss_func_comp(self.dis_y(F.concat([x_y,x])),1.0)
            else:
                x_y_copy = Variable(self._buffer_y.query(x_y.data))
                loss_dec_y_adv = losses.loss_func_comp(self.dis_y(x_y),1.0)
            loss_gen =  loss_gen + self.args.lambda_dis_y * loss_dec_y_adv            
            if self.report_start<self._iter:
                chainer.report({'loss_adv': loss_dec_y_adv}, self.dec_y)
        ## adversarial for X
        if self.args.lambda_dis_x>0:
            if self.args.conditional_discriminator:
                y_x_copy = Variable(self._buffer_x.query(F.concat([y_x,y]).data))
                loss_dec_x_adv = losses.loss_func_comp(self.dis_x(F.concat([y_x,y])),1.0)
            else:
                y_x_copy = Variable(self._buffer_x.query(y_x.data))
                loss_dec_x_adv = losses.loss_func_comp(self.dis_x(y_x),1.0)
            loss_gen =  loss_gen + self.args.lambda_dis_x * loss_dec_x_adv            
            if self.report_start<self._iter:
                chainer.report({'loss_adv': loss_dec_x_adv}, self.dec_x)

        ## X -> Y shouldn't change y
        if self.args.lambda_domain > 0 or self._iter < self.args.warmup:
            loss_dom_y = F.mean_absolute_error(y,self.dec_y(self.enc_x(y)))
            loss_dom_x = F.mean_absolute_error(x,self.dec_x(self.enc_y(x)))
            if self._iter < self.args.warmup:
                loss_gen = loss_gen + max(self.args.lambda_domain,1.0) * (loss_dom_x + loss_dom_y)
            else:
                loss_gen = loss_gen + self.args.lambda_domain * (loss_dom_x + loss_dom_y)
            if self.report_start<self._iter:
                chainer.report({'loss_dom': loss_dom_x}, self.enc_y) 
                chainer.report({'loss_dom': loss_dom_y}, self.enc_x) 
        ## images before/after conversion should look pixel-wise similar
        if self.args.lambda_identity_x > 0 or self._iter < self.args.warmup:
            loss_id_x  = losses.loss_avg(x,x_y, ksize=self.args.id_ksize, norm='l2')
            if self._iter < self.args.warmup:
                loss_gen = loss_gen + max(self.args.lambda_identity_x,2.0) * loss_id_x
            else:
                loss_gen = loss_gen + self.args.lambda_identity_x * loss_id_x
            if self.args.lambda_identity_x > 0 and self.report_start<self._iter:
                chainer.report({'loss_id': loss_id_x}, self.enc_x)
        if self.args.lambda_identity_y > 0 or self._iter < self.args.warmup:
            loss_id_y  = losses.loss_avg(y,y_x, ksize=self.args.id_ksize, norm='l2')
            if self._iter < self.args.warmup:
                loss_gen = loss_gen + max(self.args.lambda_identity_y,2.0) * loss_id_y
            else:
                loss_gen = loss_gen + self.args.lambda_identity_y * loss_id_y
            if self.args.lambda_identity_y > 0 and self.report_start<self._iter:
                chainer.report({'loss_id': loss_id_y}, self.enc_y)
        ## air should be -1
        if self.args.lambda_air > 0:
            loss_air_x = self.args.lambda_air * losses.loss_range_comp(x,x_y,0.9,norm='l2')
            loss_air_y = self.args.lambda_air * losses.loss_range_comp(y,y_x,0.9,norm='l2')
            loss_gen = loss_gen +(loss_air_x+loss_air_y)
            if self.report_start<self._iter:
                chainer.report({'loss_air': 0.1*loss_air_x}, self.dec_y)
                chainer.report({'loss_air': 0.1*loss_air_y}, self.dec_x)
        ## images before/after conversion should look similar in the gradient domain
        if self.args.lambda_grad > 0 or self._iter < self.args.warmup:
            loss_grad_x = losses.loss_grad(x,x_y)
            if self._iter < self.args.warmup:
                loss_gen = loss_gen + max(self.args.lambda_grad,1.0) * loss_grad_x
            else:
                loss_gen = loss_gen + self.args.lambda_grad * loss_grad_x
            if self.report_start<self._iter:
                chainer.report({'loss_grad': loss_grad_x}, self.dec_y)
        if self.args.lambda_tv > 0:
            loss_tv = losses.total_variation(x_y, self.args.tv_tau)
            loss_gen = loss_gen + self.args.lambda_tv * loss_tv
            if self.report_start<self._iter:
                chainer.report({'loss_tv': loss_tv}, self.dec_y)

        ## back propagate 
        self.enc_x.cleargrads()
        self.dec_x.cleargrads()
        self.enc_y.cleargrads()
        self.dec_y.cleargrads()
        loss_gen.backward()
        opt_enc_x.update()
        opt_dec_x.update()
        if not self.args.single_encoder:
            opt_enc_y.update()
        opt_dec_y.update()

        ## discriminator for Y
        if self.args.lambda_dis_y>0:
            loss_dis_y_fake = losses.loss_func_comp(self.dis_y(x_y_copy),0.0,self.args.dis_jitter)
            if self.args.conditional_discriminator:
                loss_dis_y_real = losses.loss_func_comp(self.dis_y(F.concat([y,x])),1.0,self.args.dis_jitter)
            else:
                loss_dis_y_real = losses.loss_func_comp(self.dis_y(y),1.0,self.args.dis_jitter)
            loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
            if self.report_start<self._iter:
                chainer.report({'loss_fake': loss_dis_y_fake}, self.dis_y)
                chainer.report({'loss_real': loss_dis_y_real}, self.dis_y)
            self.dis_y.cleargrads()
            loss_dis_y.backward()
            opt_y.update()

        ## discriminator for X
        if self.args.lambda_dis_x>0:
            loss_dis_x_fake = losses.loss_func_comp(self.dis_x(y_x_copy),0.0, self.args.dis_jitter)
            if self.args.conditional_discriminator:
                loss_dis_x_real = losses.loss_func_comp(self.dis_x(F.concat([x,y])),1.0,self.args.dis_jitter)
            else:
                loss_dis_x_real = losses.loss_func_comp(self.dis_x(x),1.0,self.args.dis_jitter)
            loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
            if self.report_start<self._iter:
                chainer.report({'loss_fake': loss_dis_x_fake}, self.dis_x)
                chainer.report({'loss_real': loss_dis_x_real}, self.dis_x)
            self.dis_x.cleargrads()
            loss_dis_x.backward()
            opt_x.update()

        ## discriminator for latent
        if self.args.lambda_dis_z>0 and self._iter>self.args.dis_z_start:
            loss_dis_z_x = losses.loss_func_comp(self.dis_z(x_z_copy),0.0,self.args.dis_jitter)
            loss_dis_z_y = losses.loss_func_comp(self.dis_z(y_z[-1]),1.0,self.args.dis_jitter)
            loss_dis_z = (loss_dis_z_x + loss_dis_z_y) * 0.5
            if self.report_start<self._iter:
                chainer.report({'loss_x': loss_dis_z_x}, self.dis_z)
                chainer.report({'loss_y': loss_dis_z_y}, self.dis_z)
            self.dis_z.cleargrads()
            loss_dis_z.backward()
            opt_z.update()

