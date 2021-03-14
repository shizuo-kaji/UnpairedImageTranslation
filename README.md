# Image translation by CNNs trained on unpaired data
Written by Shizuo KAJI

This is an implementation of CycleGAN

- Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, in IEEE International Conference on Computer Vision (ICCV), 2017. 

with several enhancements, most notably for dealing with CT images.
The codes have been used in 

1. S. Kida, S. Kaji, K. Nawa, T. Imae, T. Nakamoto, S. Ozaki, T. Ota, Y. Nozawa, and K. Nakagawa, Visual enhancement of Cone-beam CT by use of CycleGAN, Medical Physics, 47-3 (2020), 998--1010, https://doi.org/10.1002/mp.13963
2. S. Kaji and S. Kida, Overview of image-to-image translation by use of deep neural networks: denoising, super-resolution, modality conversion, and reconstruction in  medical imaging, Radiological Physics and Technology,  Volume 12, Issue 3 (2019), pp 235--248. https://doi.org/10.1007/s12194-019-00520-y
3. Toshikazu Imae, Shizuo Kaji, Satoshi Kida, Kanako Matsuda, Shigeharu Takenaka, Atsushi Aoki, Takahiro Nakamoto, Sho Ozaki, Kanabu Nawa, Hideomi Yamashita, Keiichi Nakagawa, and Osamu Abe, Improvement in Image Quality of CBCT during Treatment by Cycle Generative Adversarial Network, Japanese Journal of Radiological Technology, Vol. 76(11), 1173-1184, 2020, DOI: 10.6009/jjrt.2020_JSRT_76.11.1173

Please cite 1. if you use this code in your work.

This code is based on 
- https://github.com/naoto0804/chainer-cyclegan
- https://gist.github.com/crcrpar/6f1bc0937a02001f14d963ca2b86427a

## Licence
MIT Licence

## Requirements
- GPU
- python 3: [Anaconda](https://anaconda.org) is recommended
- chainer >= 7.1.0, cupy, chainerui, chainercv: install them by
```
pip install cupy,chainer,chainerui,chainercv
```

Note that with GeForce 30 RTX series, 
the installation of chainer and cupy can be a little tricky for now.
You need CUDA >= 11.1 for these GPUs, and it is supported by CuPy >= v8.
The latest version of Chainer v7.7.0 available on pip is not compatible with the latest version of CuPy.
See [here](https://github.com/chainer/chainer/pull/8583).
You can install the latest Chainer directly from the github repository, which is compatible with the latest version of CuPy.
For example, follow the following procedure:
- Install CUDA 11.1
- pip install cupy-cuda111
- pip install -U git+https://github.com/chainer/chainer.git

You will see some warning messages, but you can ignore them.


## Training
- Some demo datasets are available at https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
- Training data preparation: Under the directory named `images`, create four directories `trainA`,`trainB`,`testA`,`testB`.
Place images or directories containing images from domain A under `trainA`, and so on.
- We train four neural networks enc_x, enc_y, dec_x, and dec_y (generators) together with two or three networks (discriminators).
- enc_x takes an image X in domain A (placed under "trainA") and converts it to a latent representation Z.
Then, dec_y takes Z and converts it to an image in domain B. enc_y and dec_x go in the opposite way.
- Images under "testA" and "testB" are used for validation (visualisation produced during training).
- A brief description of other command-line arguments is given by
```
python train.py -h
```

### Example: JPEG images
- A typical training is done by
```
python train.py -R images -it jpg -cw 256 -ch 256 -o results -gc 64 128 256 -gd maxpool -gu resize -lix 1.0 -liy 1.0 -ltv 1e-3 -lreg 0.1 -lz 1 -n 0.03 -nz 0.03 -e 50
```
- The jpg (-it jpg) files under "images/trainA/" and "images/trainB/" are cropped to 256 x 256 (-cw 256 -ch 256)
and fed to the neural networks.
Crop size may have to be divisible by a large power of two (such as 8,16), if you encounter any error regarding the "shape of array".
- The generators downsampling layers consists of 64,128,256 channels (-gc 64 128 256) with convolution and maxpooling (-gd maxpool)
and upsampling layers use bilinear interpolation (-gu resize) followed by a convolution.
- The generator's loss consists of the perceptual loss comparing X and dec_y(enc_x(x)) (-lix 1.0) and that comparing Y and dec_x(enc_y(y)) (-liy 1.0),
and total variation (-ltv 1e-3).
- The latent representations Z are regularised by a third discriminator (-lz 1) and by the Euclidean norm (-lreg 0.1).
- Gaussian noise is injected before conversion (-n 0.03) and also in the latent bottleneck layer (-nz 0.03).
- The training lasts for 50 epochs (-e 50).

Learned model files "enc_???.npz" and "dec_???.npz" will appear under the directory "results" (-o results).
During training, it occasionally produces image files under "results/vis" containing (from left to right):
```
    (3 rows of) original A, converted A=>B, cyclically converted A=>B=>A, cyclically converted A=>Z=>A
    (3 rows of) original B, converted B=>A, cyclically converted B=>A=>B, cyclically converted B=>Z=>B
``` 
where Z denotes the latent representation. The number of rows (3 times 2 by default) can be changed by specifying (-nvis_A 5 -nvis_B 4).

- You can also specify various other parameters. For example,
    - (-lcA 10 -lcB 10) tells the strength of cycle consistency (A=>B=>A, B=>A=>B) in the loss function
    - (-lcAz 10 -lcBz 10) tells the strength of cycle consistency (A=>Z=>A, B=>Z=>B) in the loss function
    - `python train.py -h` provides a full list.

### Example: DICOM files
```
train.py -R dicom/ -it dcm -e 50 -u none -huba -600 -hura 1000 -hubb -600 -hurb 1000 -la 1 -lix 1 -liy 1 -rr 20 -rt 20
```
- It reads dcm files (-it dcm) placed under the directory (and its subdirectories) `dicom/trainA` and so on (-R dicom/).
- The type of skip-connections is specified by (-u none). The choices are `none` (no skip), `concat` (concatenation along channel),
`conv` (concat after convolution with channels specified by (--skipdim 4)), `add` (addition).
- HU values in domain A are internally scaled to [-1,1], by taking -600 to -1 (-huba -600) and 400 to 1 (-hura 1000; 400 is -600+1000).
HU values lower than -600 are clipped to -1 and higher than 400 are clipped to 1.
- Same scaling and clipping of HU values applies to domain B specified by (-hubb -600 -hurb 1000).
- the ratio of the pixel size can be specified by, e.g., (-fs 0.7634). The performance gets better if the pixel sizes are consistent for domains A and B.
- (-la 1) specifies the weight of loss function to preserve air region which are pixels with HU lower than the threshold.
The threshold is specified by (--air_threshold -0.997): the value -0.997 indicates the scaled HU as above.
- Data augmentation is performed according to (-rr 20, random rotation of -20 to 20 degrees) and (-rt 20, random translation of up to 20 pixels along each axis).


### Conversion
```
python convert.py -a results/args -it jpg -R input_dir -o output_dir -b 10 -m enc_x50.npz
```
searches for jpg files recursively under `input_dir` and outputs converted images by the generator dec_y(enc_x(X)) to output_dir.
If you specify -m enc_y50.npz instead, you get converted images in the opposite way.
A larger batch size (-b 10) increases the conversion speed but may consume too much GPU memory.
