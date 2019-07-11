# Image translation by CNNs trained on unpaired data
Written by Shizuo KAJI

This is an implementation of CycleGAN

- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, in IEEE International Conference on Computer Vision (ICCV), 2017. 

with several enhancements.

This code is based on 
- https://github.com/naoto0804/chainer-cyclegan
- https://gist.github.com/crcrpar/6f1bc0937a02001f14d963ca2b86427a

## Licence
MIT Licence

### Requirements
- a modern GPU
- python 3: [Anaconda](https://anaconda.org) is recommended
- chainer >= 6.1.0, cupy, chainerui, chainercv: install them by
```
pip install cupy,chainer,chainerui,chainercv
```
- a pretrained VGG16 model (it will be downloaded automatically when used for the first time. Thus, it may take a while.)

### Training
- Some demo datasets are available at https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
- Training data preparation: Under the directory named images, place four directories
"trainA","trainB","testA","testB"
- We train four neural networks enc_x, enc_y, dec_x, and dec_y (generators) together with two or three networks (discriminators).
- enc_x takes an image X in domain A (placed under "trainA") and converts it to a latent representation Z.
Then, dec_y takes Z and converts it to an image in domain B. enc_y and dec_x go in the opposite way.
- Images under "testA" and "testB" are used for validation (visualisation produced during training).
- A typical training is done by
```
python train.py -R images -it jpg -cw 256 -ch 256 -o results -gc 64 128 256 -gd maxpool -gu resize -lix 1.0 -liy 1.0 -ltv 1e-3 -lreg 0.1 -lz 1 -n 0.03 -nz 0.03 -e 50
```
The jpg (-it jpg) files under "images/trainA/" and "images/trainB/" are cropped to 256 x 256 (-cw 256 -ch 256)
and fed to the neural networks.
Crop size may have to be divisible by a large power of two (such as 8,16), if you encounter any error regarding the "shape of array".

The generators downsampling layers consists of 64,128,256 channels (-gc 64 128 256) with convolution and maxpooling (-gd maxpool)
and upsampling layers use bilinear interpolation (-gu resize) followed by a convolution.
The generator's loss consists of the perceptual loss comparing X and dec_y(enc_x(x)) (-lix 1.0) and that comparing Y and dec_x(enc_y(y)) (-liy 1.0),
and total variation (-ltv 1e-3).
The latent representations Z are regularised by a third discriminator (-lz 1) and by the Euclidean norm (-lreg 0.1).
Gaussian noise is injected before conversion (-n 0.03) and also in the latent bottleneck layer (-nz 0.03).

The training lasts for 50 epochs (-e 50).
Learned model files "gen_g??.npz" and "gen_f??.npz" will appear under the directory "results" (-o results).
During training, it occasionally produces image files under "results/vis" containing original, converted, cyclically converted images in each row. 
- A brief description of other command-line arguments is given by
```
python train.py -h
```
Note that adding a lot of different losses may cause memory shortage.

### Conversion
```
python convert.py -a results/args -it jpg -R input_dir -o output_dir -b 10 -m enc_x50.npz
```
searches for jpg files recursively under input_dir and outputs converted images by the generator dec_y(enc_x(X)) to output_dir.
If you specify -m enc_y50.npz instead, you get converted images in the opposite way.
A larger batch size (-b 10) increases the conversion speed but may consume too much GPU memory.
