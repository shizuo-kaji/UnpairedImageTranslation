# Image translation by CNNs trained on unpaired data
- Based on https://github.com/naoto0804/chainer-cyclegan
- Demo data https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

### Requirements
- chainer >= 5.0.0
- pip install cupy,chainer,chainerui,chainercv

### Training
- Training data preparation:
Under the directory named images, place four directories
"trainA","trainB","testA","testB"
- We train two neural networks G and F (generators) together with two auxiliary networks (discriminators).
- G takes images in domain A (placed under "trainA") and converts them to images in domain B. F goes in the opposite way.
- Images under "testA" and "testB" are used for validation
- A typical training is done by
```
python train.py -cw 256 -ch 256 -R images -t jpg -o results -gc 64 128 256
```
The jpg files under "images/trainA" and "images/trainB" are cropped in the middle to 256x256
and fed in to neural networks.
You will obtain learnt model files "gen_g??.npz" and "gen_f??.npz"
under the directory "results".
Also, it occasionally produces image files under "results/vis" containing
rows with A G(A) F(G(A)) and B F(B) G(F(B)).
- A brief description of other command-line arguments is give by
```
python train.py -h
```

### Conversion
```
python convert.py -a results/args -t jpg -R input_dir -o output_dir -b 10 -m gen_g50.npz
```
searches for jpg files recursively under input_dir and outputs converted images by the generator G to output_dir.
If you specify -m gen_f50.npz instead, you get converted images by the generator F.

### Another version
A version based on a shared latent space model is also included.
A typical training goes similarly with
```
python trainAE.py -lreg 0.1 --noise 0.01 --noise_z 0.01 -lz 1
```