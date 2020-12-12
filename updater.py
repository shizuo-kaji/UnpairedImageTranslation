import random
import chainer
import chainer.functions as F
from chainer import Variable,cuda
import losses

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.enc_x, self.dec_x, self.enc_y, self.dec_y, self.dis_x, self.dis_y, self.dis_z = kwargs.pop('models')
        params = kwargs.pop('params')
#        self.device_id = kwargs.pop('device')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.xp = self.enc_x.xp
        self._buffer_y = losses.ImagePool(50 * self.args.batch_size)
        self._buffer_x = losses.ImagePool(50 * self.args.batch_size)
        self.perceptual_model = params['perceptual_model']

    def update_core(self):
        opt_enc_x = self.get_optimizer('enc_x')
        opt_dec_x = self.get_optimizer('dec_x')
        opt_enc_y = self.get_optimizer('enc_y')
        opt_dec_y = self.get_optimizer('dec_y')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')
        opt_z = self.get_optimizer('dis_z')

        # get mini-batch
        batch_x = self.get_iterator('main').next()
        batch_y = self.get_iterator('train_B').next()
        x = Variable(self.converter(batch_x, self.args.gpu[0]))
        y = Variable(self.converter(batch_y, self.args.gpu[0]))

        # encode to latent (X,Y => Z)
        x_z = self.enc_x(losses.add_noise(x, sigma=self.args.noise))
        y_z = self.enc_y(losses.add_noise(y, sigma=self.args.noise))

        loss_gen = 0
        ## regularisation on the latent space
        if self.args.lambda_reg>0:
            loss_reg_enc_y = losses.loss_func_reg(y_z[-1],'l2')
            loss_reg_enc_x = losses.loss_func_reg(x_z[-1],'l2') 
            loss_gen = loss_gen + self.args.lambda_reg * (loss_reg_enc_x + loss_reg_enc_y)
            chainer.report({'loss_reg': loss_reg_enc_x}, self.enc_x)
            chainer.report({'loss_reg': loss_reg_enc_y}, self.enc_y)

        ## discriminator for the latent space: distribution of image of enc_x should look same as that of enc_y
        # since z is a list (for u-net), we use only the output of the last layer
        if self.args.lambda_dis_z>0:
            if self.args.dis_wgan:
                loss_enc_x_adv = -F.average(self.dis_z(x_z[-1]))
                loss_enc_y_adv = F.average(self.dis_z(y_z[-1]))
            else:
                loss_enc_x_adv = losses.loss_func_comp(self.dis_z(x_z[-1]),1.0)
                loss_enc_y_adv = losses.loss_func_comp(self.dis_z(y_z[-1]),0.0)
            loss_gen = loss_gen +  self.args.lambda_dis_z * (loss_enc_x_adv+loss_enc_y_adv)
            chainer.report({'loss_adv': loss_enc_x_adv}, self.enc_x)
            chainer.report({'loss_adv': loss_enc_y_adv}, self.enc_y)

        # cycle for X=>Z=>X (Autoencoder)
        x_x = self.dec_x(x_z)
        loss_cycle_xzx = F.mean_absolute_error(x_x, x)
        chainer.report({'loss_cycle': loss_cycle_xzx}, self.enc_x)
        # cycle for Y=>Z=>Y (Autoencoder)
        y_y = self.dec_y(y_z)
        loss_cycle_yzy = F.mean_absolute_error(y_y, y)
        chainer.report({'loss_cycle': loss_cycle_yzy}, self.enc_y)
        loss_gen = loss_gen + self.args.lambda_Az * loss_cycle_xzx + self.args.lambda_Bz * loss_cycle_yzy

        ## decode from latent Z => Y,X
        x_y = self.dec_y(x_z)
        y_x = self.dec_x(y_z)

        # cycle for X=>Z=>Y=>Z=>X  (Z=>Y=>Z does not work well)
        x_y_x = self.dec_x(self.enc_y(x_y))
        loss_cycle_x = F.mean_absolute_error(x_y_x,x)
        chainer.report({'loss_cycle': loss_cycle_x}, self.dec_x)
        # cycle for Y=>Z=>X=>Z=>Y
        y_x_y = self.dec_y(self.enc_x(y_x))
        loss_cycle_y = F.mean_absolute_error(y_x_y,y)
        chainer.report({'loss_cycle': loss_cycle_y}, self.dec_y)
        loss_gen = loss_gen + self.args.lambda_A * loss_cycle_x + self.args.lambda_B * loss_cycle_y

        ## adversarial for Y
        if self.args.lambda_dis_y>0:
            x_y_copy = Variable(self._buffer_y.query(x_y.data))
            if self.args.dis_wgan:
                loss_dec_y_adv = -F.average(self.dis_y(x_y))
            else:
                loss_dec_y_adv = losses.loss_func_comp(self.dis_y(x_y),1.0)
            loss_gen =  loss_gen + self.args.lambda_dis_y * loss_dec_y_adv            
            chainer.report({'loss_adv': loss_dec_y_adv}, self.dec_y)
        ## adversarial for X
        if self.args.lambda_dis_x>0:
            y_x_copy = Variable(self._buffer_x.query(y_x.data))
            if self.args.dis_wgan:
                loss_dec_x_adv = -F.average(self.dis_x(y_x))
            else:
                loss_dec_x_adv = losses.loss_func_comp(self.dis_x(y_x),1.0)
            loss_gen =  loss_gen + self.args.lambda_dis_x * loss_dec_x_adv            
            chainer.report({'loss_adv': loss_dec_x_adv}, self.dec_x)

        ## idempotence
        if self.args.lambda_idempotence > 0:             
            loss_idem_x = F.mean_absolute_error(y_x,self.dec_x(self.enc_y(y_x)))
            loss_idem_y = F.mean_absolute_error(x_y,self.dec_y(self.enc_x(x_y)))
            loss_gen = loss_gen + self.args.lambda_idempotence * (loss_idem_x + loss_idem_y)
            chainer.report({'loss_idem': loss_idem_x}, self.dec_x) 
            chainer.report({'loss_idem': loss_idem_y}, self.dec_y)
        # Y => X shouldn't change X            
        if self.args.lambda_domain > 0:             
            loss_dom_x = F.mean_absolute_error(x,self.dec_x(self.enc_y(x)))
            loss_dom_y = F.mean_absolute_error(y,self.dec_y(self.enc_x(y)))
            loss_gen = loss_gen + self.args.lambda_domain * (loss_dom_x + loss_dom_y)
            chainer.report({'loss_dom': loss_dom_x}, self.dec_x) 
            chainer.report({'loss_dom': loss_dom_y}, self.dec_y)

        ## images before/after conversion should look similar in terms of perceptual loss
        if self.args.lambda_identity_x > 0:
            loss_id_x = losses.loss_perceptual(x,x_y,self.perceptual_model,layer=self.args.perceptual_layer,grey=self.args.grey)
            loss_gen = loss_gen + self.args.lambda_identity_x * loss_id_x
            chainer.report({'loss_id': 1e-3*loss_id_x}, self.enc_x)
        if self.args.lambda_identity_y > 0:
            loss_id_y = losses.loss_perceptual(y,y_x,self.perceptual_model,layer=self.args.perceptual_layer,grey=self.args.grey)
            loss_gen = loss_gen + self.args.lambda_identity_y * loss_id_y
            chainer.report({'loss_id': 1e-3*loss_id_y}, self.enc_y)
        ## background (pixels with value -1) should be preserved
        if self.args.lambda_air > 0:
            loss_air_x = losses.loss_comp_low(x,x_y,self.args.air_threshold,norm='l1')
            loss_air_y = losses.loss_comp_low(y,y_x,self.args.air_threshold,norm='l1')
            loss_gen = loss_gen + self.args.lambda_air * (loss_air_x+loss_air_y)
            chainer.report({'loss_air': loss_air_x}, self.dec_y)
            chainer.report({'loss_air': loss_air_y}, self.dec_x)
        ## images before/after conversion should look similar in the gradient domain
        if self.args.lambda_grad > 0:
            loss_grad_x = losses.loss_grad(x,x_y,norm='l1')
            loss_grad_y = losses.loss_grad(y,y_x,norm='l1')
            loss_gen = loss_gen + self.args.lambda_grad * (loss_grad_x + loss_grad_y)
            chainer.report({'loss_grad': loss_grad_x}, self.dec_y)
            chainer.report({'loss_grad': loss_grad_y}, self.dec_x)
        ## total variation (only for X -> Y)
        if self.args.lambda_tv > 0:
            loss_tv = losses.total_variation(x_y, tau=self.args.tv_tau, method=self.args.tv_method)
            if self.args.imgtype=="dcm" and self.args.num_slices>1:
                loss_tv += losses.total_variation_ch(x_y)
            loss_gen = loss_gen + self.args.lambda_tv * loss_tv
            chainer.report({'loss_tv': loss_tv}, self.dec_y)

        ## back propagate 
        self.enc_x.cleargrads()
        self.dec_x.cleargrads()
        self.enc_y.cleargrads()
        self.dec_y.cleargrads()
        loss_gen.backward()
        opt_enc_x.update(loss=loss_gen)
        opt_dec_x.update(loss=loss_gen)
        if not self.args.single_encoder:
            opt_enc_y.update(loss=loss_gen)
        opt_dec_y.update(loss=loss_gen)

        ##########################################
        ## discriminator for Y
        if self.args.dis_wgan: ## synthesised -, real +
            eps = self.xp.random.uniform(0, 1, size=len(batch_y)).astype(self.xp.float32)[:, None, None, None]
            if self.args.lambda_dis_y>0:
                ## discriminator for X=>Y
                loss_dis_y = F.average(self.dis_y(x_y_copy)-self.dis_y(y))
                y_mid = eps * y + (1.0 - eps) * x_y_copy
                # gradient penalty
                gd_y, = chainer.grad([self.dis_y(y_mid)], [y_mid], enable_double_backprop=True)
                gd_y = F.sqrt(F.batch_l2_norm_squared(gd_y) + 1e-6)
                loss_dis_y_gp = F.mean_squared_error(gd_y, self.xp.ones_like(gd_y.data))                
                chainer.report({'loss_dis': loss_dis_y}, self.dis_y)
                chainer.report({'loss_gp': self.args.lambda_wgan_gp * loss_dis_y_gp}, self.dis_y)
                loss_dis_y = loss_dis_y + self.args.lambda_wgan_gp * loss_dis_y_gp
                self.dis_y.cleargrads()
                loss_dis_y.backward()
                opt_y.update(loss=loss_dis_y)

            if self.args.lambda_dis_x>0:
                ## discriminator for B=>A
                loss_dis_x = F.average(self.dis_x(y_x_copy)-self.dis_x(x))
                x_mid = eps * x + (1.0 - eps) * y_x_copy
                # gradient penalty
                gd_x, = chainer.grad([self.dis_x(x_mid)], [x_mid], enable_double_backprop=True)
                gd_x = F.sqrt(F.batch_l2_norm_squared(gd_x) + 1e-6)
                loss_dis_x_gp = F.mean_squared_error(gd_x, self.xp.ones_like(gd_x.data))
                chainer.report({'loss_dis': loss_dis_x}, self.dis_x)
                chainer.report({'loss_gp': self.args.lambda_wgan_gp *loss_dis_x_gp}, self.dis_x)
                loss_dis_x = loss_dis_x + self.args.lambda_wgan_gp * loss_dis_x_gp
                self.dis_x.cleargrads()
                loss_dis_x.backward()
                opt_x.update(loss=loss_dis_x)

            ## discriminator for latent: X -> Z is - while Y -> Z is +
            if self.args.lambda_dis_z>0:
                loss_dis_z = F.average(self.dis_z(x_z[-1])-self.dis_z(y_z[-1]))
                z_mid = eps * x_z[-1] + (1.0 - eps) * y_z[-1]
                # gradient penalty
                gd_z, = chainer.grad([self.dis_z(z_mid)], [z_mid], enable_double_backprop=True)
                gd_z = F.sqrt(F.batch_l2_norm_squared(gd_z) + 1e-6)
                loss_dis_z_gp = F.mean_squared_error(gd_z, self.xp.ones_like(gd_z.data))                
                chainer.report({'loss_dis': loss_dis_z}, self.dis_z)
                chainer.report({'loss_gp': self.args.lambda_wgan_gp * loss_dis_y_gp}, self.dis_y)
                loss_dis_z = loss_dis_z + self.args.lambda_wgan_gp * loss_dis_z_gp
                self.dis_z.cleargrads()
                loss_dis_z.backward()
                opt_z.update(loss=loss_dis_z)

        elif(self.iteration % self.args.learning_freq_d == 0):  ## LSGAN
            if self.args.lambda_dis_y>0:
                ## discriminator for A=>B (real:1, fake:0)
                disy_fake = self.dis_y(x_y_copy)
                loss_dis_y_fake = losses.loss_func_comp(disy_fake,0.0, self.args.dis_jitter)
                disy_real = self.dis_y(y)
                loss_dis_y_real = losses.loss_func_comp(disy_real,1.0, self.args.dis_jitter)
                if self.args.dis_reg_weighting>0:  ## regularization
                    loss_dis_y_reg = (F.average(F.absolute(disy_real[:,1,:,:])) + F.average(F.absolute(disy_fake[:,1,:,:])))
                else:
                    loss_dis_y_reg = 0
                chainer.report({'loss_reg': loss_dis_y_reg}, self.dis_y)
                chainer.report({'loss_fake': loss_dis_y_fake}, self.dis_y)
                chainer.report({'loss_real': loss_dis_y_real}, self.dis_y)
                loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5 + self.args.dis_reg_weighting * loss_dis_y_reg
                self.dis_y.cleargrads()
                loss_dis_y.backward()
                opt_y.update(loss=loss_dis_y)

            if self.args.lambda_dis_x>0:
                ## discriminator for B=>A
                disx_fake = self.dis_x(y_x_copy)
                loss_dis_x_fake = losses.loss_func_comp(disx_fake,0.0, self.args.dis_jitter)
                disx_real = self.dis_x(x)
                loss_dis_x_real = losses.loss_func_comp(disx_real,1.0, self.args.dis_jitter)
                if self.args.dis_reg_weighting>0: ## regularization
                    loss_dis_x_reg = (F.average(F.absolute(disx_fake[:,1,:,:]))+ F.average(F.absolute(disx_real[:,1,:,:])))
                else:
                    loss_dis_x_reg = 0
                chainer.report({'loss_reg': loss_dis_x_reg}, self.dis_x)
                chainer.report({'loss_fake': loss_dis_x_fake}, self.dis_x)
                chainer.report({'loss_real': loss_dis_x_real}, self.dis_x)
                loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5 + self.args.dis_reg_weighting * loss_dis_x_reg
                self.dis_x.cleargrads()
                loss_dis_x.backward()
                opt_x.update(loss=loss_dis_x)

            ## discriminator for latent: X -> Z is 0.0 while Y -> Z is 1.0
            if self.args.lambda_dis_z>0:
                disz_xz = self.dis_z(x_z[-1])
                loss_dis_z_x = losses.loss_func_comp(disz_xz,0.0,self.args.dis_jitter)
                disz_yz = self.dis_z(y_z[-1])
                loss_dis_z_y = losses.loss_func_comp(disz_yz,1.0,self.args.dis_jitter)
                if self.args.dis_reg_weighting>0: ## regularization
                    loss_dis_z_reg = (F.average(F.absolute(disz_xz[:,1,:,:]))+ F.average(F.absolute(disz_yz[:,1,:,:])))
                else:
                    loss_dis_z_reg = 0
                chainer.report({'loss_x': loss_dis_z_x}, self.dis_z)
                chainer.report({'loss_y': loss_dis_z_y}, self.dis_z)
                chainer.report({'loss_reg': loss_dis_z_reg}, self.dis_z)                
                loss_dis_z = (loss_dis_z_x + loss_dis_z_y) * 0.5 + self.args.dis_reg_weighting * loss_dis_z_reg
                self.dis_z.cleargrads()
                loss_dis_z.backward()
                opt_z.update(loss=loss_dis_z)

        # prepare next images
        # if(t<self.args.n_critics-1):
        #     x_y_copy = Variable(self.xp.concatenate(random.sample(self._buffer_y.images,self.args.batch_size)))
        #     y_x_copy = Variable(self.xp.concatenate(random.sample(self._buffer_x.images,self.args.batch_size)))
        #     batch_x = self.get_iterator('main').next()
        #     batch_y = self.get_iterator('train_B').next()
        #     x = Variable(self.converter(batch_x, self.device))
        #     y = Variable(self.converter(batch_y, self.device))

