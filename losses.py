import random
import chainer
import chainer.functions as F
from chainer import Variable,cuda

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        xp = cuda.get_array_module(images)
        for image in images:
            image = xp.expand_dims(image, axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = xp.copy(self.images[random_id])
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = xp.concatenate(return_images)
        return return_images

def add_noise(h, sigma): 
    xp = cuda.get_array_module(h.data)
    if chainer.config.train and sigma>0:
        return h + sigma * xp.random.randn(*h.data.shape, dtype=h.dtype)
    else:
        return h

def loss_avg(x,y, ksize=3, norm='l2'):
    if ksize>1:
        ax = F.average_pooling_2d(x,ksize,1,0)
        ay = F.average_pooling_2d(y,ksize,1,0)
    else:
        ax = x
        ay = y
    if norm=='l1':
        return F.mean_absolute_error(ax,ay)
    else:
        return F.mean_squared_error(ax,ay)

def loss_avg_d(diff, ksize=3):
    a = F.average_pooling_2d(diff,ksize,1,0)
    return(F.average(a**2))

def loss_perceptual(x,y,model):
    with chainer.using_config('train', False) and chainer.function.no_backprop_mode():
        if x.shape[1]==1:
            xp = cuda.get_array_module(x.data)
            vx = model(F.concat([x,x,x]), layers=['pool3'])['pool3']
            vy = model(F.concat([y,y,y]), layers=['pool3'])['pool3']
        else:
            vx = model(x, layers=['pool3'])['pool3']
            vy = model(y, layers=['pool3'])['pool3']
    return(F.mean_squared_error(vx,vy))

def loss_grad(x, y, norm='l1'):
    xp = cuda.get_array_module(x.data)
    grad = xp.tile(xp.asarray([[[[1,0,-1],[2,0,-2],[1,0,-1]]]],dtype=x.dtype),(x.data.shape[1],1,1))
    dxx = F.convolution_2d(x,grad)
    dyx = F.convolution_2d(y,grad)
    dxy = F.convolution_2d(x,xp.transpose(grad,(0,1,3,2)))
    dyy = F.convolution_2d(y,xp.transpose(grad,(0,1,3,2)))
    if norm=='l1':
        return F.mean_absolute_error(dxx,dyx)+F.mean_squared_error(dxy,dyy)
    else:
        return F.mean_squared_error(dxx,dyx)+F.mean_squared_error(dxy,dyy)

def loss_grad_d(diff):
    xp = cuda.get_array_module(diff.data)
    grad = xp.tile(xp.asarray([[[[1,0,-1],[2,0,-2],[1,0,-1]]]],dtype=diff.dtype),(diff.data.shape[1],1,1))
    dx = F.convolution_2d(diff,grad)
    dy = F.convolution_2d(diff,xp.transpose(grad,(0,1,3,2)))
#        target = self.xp.zeros_like(dx.data)
#        return 0.5*(F.mean_squared_error(dx,target)+F.mean_squared_error(dy,target))
    return F.average(dx**2) + F.average(dy**2)

## align air
def loss_range_comp(x,y,cutoff,norm='l1'):
    # compare only pixels with -x > cutoff
    if norm=='l1':
        return(F.sum(F.absolute(F.relu(-x-cutoff)-F.relu(-y-cutoff))))
    else:
        return(F.sum((F.relu(-x-cutoff)-F.relu(-y-cutoff))**2))

def loss_func_comp(y, val, noise=0):
    xp = cuda.get_array_module(y.data)
    if noise>0:
        val += random.normalvariate(0,noise)   ## jitter for the target value
#        val += random.uniform(-noise, noise)   ## jitter for the target value
    target = xp.full(y.data.shape, val, dtype=y.dtype)
    return F.mean_squared_error(y, target)

def loss_func_reg(y,norm='l1'):
    if norm=='l1':
        return(F.average(F.absolute(y)))
    else:
        return(F.average(y**2))

def total_variation(x,tau=1e-6):
    xp = cuda.get_array_module(x.data)
    wh = xp.tile(xp.asarray([[[[1,0],[-1,0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
    ww = xp.tile(xp.asarray([[[[1, -1],[0, 0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
    dx = F.convolution_2d(x, W=wh)
    dy = F.convolution_2d(x, W=ww)
    d = F.sqrt(dx**2 + dy**2 + xp.full(dx.data.shape, tau**2, dtype=dx.dtype))
    return(F.average(d))

def total_variation2(x):
    xp = cuda.get_array_module(x.data)
    wh = xp.asarray([[[[1],[-1]]]], dtype=x.dtype)
    ww = xp.asarray([[[[1, -1]]]], dtype=x.dtype)
    dx = F.convolution_2d(x, W=wh)
    dy = F.convolution_2d(x, W=ww)
#    dx = x[:, 1:, :, :] - x[:, :-1, :, :]
#    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return F.average(F.absolute(dx))+F.average(F.absolute(dy))
