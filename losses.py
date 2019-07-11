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
                if random.choice([True, False]):
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

def loss_perceptual(x,y,model,layer='conv4_2',grey=False):
    with chainer.using_config('train', False):
        if grey:
            loss = 0
            for i in range(x.shape[1]):
                xp = cuda.get_array_module(x.data)
                xx=x[:,i:(i+1),:,:]
                vx = model(F.concat([xx,xx,xx]), layers=[layer])[layer]
                yy=y[:,i:(i+1),:,:]
                vy = model(F.concat([yy,yy,yy]), layers=[layer])[layer]
                loss += F.mean_squared_error(vx,vy)
            loss /= x.shape[1]
        else:
            vx = model(x, layers=[layer])[layer]
            vy = model(y, layers=[layer])[layer]
            loss = F.mean_squared_error(vx,vy)
    return(loss)

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
    return F.average(dx**2) + F.average(dy**2)

# compare only pixels with x < threshold. Note x is fixed and y is variable
def loss_comp_low(x,y,threshold,norm='l1'):
    if norm=='l1':
        return(F.average( ( (x.array<threshold)+(y.array<threshold) ) * F.absolute(x-y)))
    else:
        return(F.average( ( (x.array<threshold)+(y.array<threshold) ) * ((x-y)**2) ))

def loss_func_comp(y, val, noise=0):
    xp = cuda.get_array_module(y.data)
    if noise>0:
        val += random.normalvariate(0,noise)   ## jitter for the target value
#        val += random.uniform(-noise, noise)   ## jitter for the target value
    shape = y.data.shape
    if y.shape[1] == 2:  ## weighted discriminator
        shape = (shape[0],1,shape[2],shape[3])
        target = xp.full(shape, val, dtype=y.dtype)
        W = F.tanh(y[:,1,:,:])+1
        return F.average( ((y[:,0,:,:]-target)**2) * W )  ## weighted loss
    else:
        target = xp.full(shape, val, dtype=y.dtype)
        return F.mean_squared_error(y, target)

def loss_func_reg(y,norm='l1'):
    if norm=='l1':
        return(F.average(F.absolute(y)))
    else:
        return(F.average(y**2))

def total_variation(x,tau=1e-6, method="abs"):
    xp = cuda.get_array_module(x.data)
    if method=="abs":
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return F.average(F.absolute(dx))+F.average(F.absolute(dy))
    elif method=="sobel":
        wh = xp.tile(xp.asarray([[[[1,0],[-1,0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
        ww = xp.tile(xp.asarray([[[[1, -1],[0, 0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
        dx = F.convolution_2d(x, W=wh)
        dy = F.convolution_2d(x, W=ww)
        d = F.sqrt(dx**2 + dy**2 + xp.full(dx.data.shape, tau**2, dtype=dx.dtype))
        return(F.average(d))
    elif method=="usual":
        xp = cuda.get_array_module(x.data)
        dx = x[:, :, 1:, :-1] - x[:, :, :-1, :-1]
        dy = x[:, :, :-1, 1:] - x[:, :, :-1, :-1]
        d = F.sqrt(dx**2 + dy**2 + xp.full(dx.data.shape, tau**2, dtype=dx.dtype))
        return(F.average(d))

def total_variation_ch(x):
    xp = cuda.get_array_module(x.data)
    dx = x[:, 1:, :, :] - x[:, :-1, :, :]
    return F.average(F.absolute(dx))
