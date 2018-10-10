# Autoencoder

This repository is to do convolutional autoencoder with SetNet based on Cars Dataset from Stanford.


## Dependencies

- Python 3.5
- PyTorch 0.4

## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

 ![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/random.jpg)

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Autoencoder/data
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

## Architecture

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/segnet.jpg)

## Usage

### Data Pre-processing
Extract 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation):
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

### Demo
Download pre-trained [model](https://github.com/foamliu/Autoencoder/releases/download/v1.0/BEST_checkpoint.tar) weights into "models" folder then run:

```bash
$ python demo.py
```

Then check results in images folder, something like:

Input | Output |
|---|---|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/0_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/0_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/1_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/2_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/2_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/3_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/3_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/4_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/4_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/5_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/6_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/6_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/7_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/7_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/8_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/8_out.png)|
|![image](https://github.com/foamliu/Autoencoder/raw/master/images/9_image.png) | ![image](https://github.com/foamliu/Autoencoder/raw/master/images/9_out.png)|
