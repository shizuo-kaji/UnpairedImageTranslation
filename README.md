# Image translation by CNNs trained on unpaired data
Written by Shizuo KAJI

Based on https://github.com/naoto0804/chainer-cyclegan

This is an implementation of CycleGAN

- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, in IEEE International Conference on Computer Vision (ICCV), 2017. 

with several enhancements.


### Requirements
- a modern GPU
- python 3: [Anaconda](https://anaconda.org) is recommended
- chainer >= 5.3.0, cupy, chainerui, chainercv: install them by
```
pip install cupy,chainer,chainerui,chainercv
```
- a pretrained VGG16 model (it will be downloaded automatically when used for the first time. Thus, it may take a while.)

### Training
- Some demo datasets are available at https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
- Training data preparation:
Under the directory named images, place four directories
"trainA","trainB","testA","testB"
- We train two neural networks G and F (generators) together with two auxiliary networks (discriminators).
- G takes images in domain A (placed under "trainA") and converts them to images in domain B. F goes in the opposite way.
- Images under "testA" and "testB" are used for validation
- A typical training is done by
```
python train.py -R images -t jpg -cw 256 -ch 256 -o results -gc 64 128 256 -gd maxpool -gu resize -lix 1.0 -liy 1.0 -ltv 1e-3 -ld 1.0 -n 0.03 -nz 0.03 -e 50
```
The jpg (-t jpg) files under "images/trainA/" and "images/trainB/" are cropped to 256 x 256 (-cw 256 -ch 256)
and fed in to neural networks.
The generators downsampling layers consists of 64,128,256 channels (-gc 64 128 256) with convolution and maxpooling (-gd maxpool)
and upsampling layers use binilear interpolation (-gu resize) followed by a convolution.
The generator's loss consists of the perceptual loss comparing x and G(x) (-lix 1.0) and that comparing y and F(y) (-liy 1.0),
and total variation (-ltv 1e-3) and the domain preservation which is the L2 error comparing x and F(x) and y and G(y) (-ld 1.0).
Gaussian noise is injected before conversion (-n 0.03) and also in the latent bottleneck layer (-nz 0.03).

The training lasts for 50 epochs (-e 50).
You will obtain learnt model files "gen_g??.npz" and "gen_f??.npz" under the directory "results" (-o results).
Also, it occasionally produces image files under "results/vis" containing
rows with A G(A) F(G(A)) and B F(B) G(F(B)).
- A brief description of other command-line arguments is given by
```
python train.py -h
```
Note that adding a lot of different losses may cause memory shortage.

### Conversion
```
python convert.py -a results/args -t jpg -R input_dir -o output_dir -b 10 -m gen_g50.npz
```
searches for jpg files recursively under input_dir and outputs converted images by the generator G to output_dir.
If you specify -m gen_f50.npz instead, you get converted images by the generator F.
A larger batch size (-b 10) increases the conversion speed but may consume too much memory.

### Another version
A version based on a shared latent space model is also included.
A typical training goes similarly with
```
python trainAE.py -cw 256 -ch 256 -R images -t jpg -o results -lreg 0.03 -n 0.03 -nz 0.03 -lz 1 -lix 1 -liy 1
```
