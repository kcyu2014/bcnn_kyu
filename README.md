# BCNN for fine-grained recognition #

This repository contains the code reproducing the results published in ICCV 2015 paper "Bilinear CNN Models for Fine-grained Visual Recognition." The model is evaluated on cub-200-2011, FGVC-aircraft and cars dataset. We tested our code on Ubuntu 14.04 using Nvidia K40 GPU.

##Dependencies##

###The repository contains code using VLFeat and MatConvNet to:###
* Extract convolutional feature
* fine-tuning BCNN models
* train linear svm classifiers on features

Please visit MatConvNet and Vlfeat git repositories to download the packages. Our code is built based on MatConvNet version 1.0-beta8. Version 1.0-beta9 also works fine for the speedup by cudnn.
Edit setup.m to link to the packages.

To retrieve older version of MatConvNet:

      git fetch --tags
      git checkout tags/v1.0-beta8


##Pre-train models##
ImageNet pre-trained models: Since we don't support the latest MatConvNet implementation, the pre-trained models download from MatConvNet page don't work properly here. We provide the links to download [vgg-m](http://vis-www.cs.umass.edu/bcnn/download/imagenet-vgg-m.mat) and [vgg-verydeep-16](http://vis-www.cs.umass.edu/bcnn/download/imagenet-vgg-verydeep-16.mat) in old format.

We provide three BCNN pre-trained models([M,M], [D,M], and [D,D}) for each of cub-200-2011, FGVC-aircraft and cars dataset. Note that for [M,M] and [D,D], we run the symmetric model, where you can simply use the same network for both two streams during encoding.

### bcnn pre-trained on cub ###

* [bcnn-cub-mm-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cub-mm.mat)
* [bcnn-cub-dm-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cub-dm.zip)
* [bcnn-cub-dd-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cub-dd.mat)

### bcnn pre-trained on FGVC-aircraft ###

* [bcnn-aircrafts-mm-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-aircrafts-mm.mat)
* [bcnn-aircrafts-dm-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-aircrafts-dm.zip)
* [bcnn-aircrafts-dd-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-aircrafts-dd.mat)

### bcnn pre-trained on cars ###

* [bcnn-car-mm-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cars-mm.mat)
* [bcnn-car-dm-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cars-dm.zip)
* [bcnn-car-dd-net](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cars-dd.mat)

###Get dataset###

Please download the datasets from their webpage and edit the function model_setup for your dataset location.

* [cub-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [FGVC-aircraft](http://www.robots.ox.ac.uk/~vgg/data/oid/)
* [cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

###Run toy demo###
The script demo_test.m takes an input image and runs our pre-trained fine-grained bird classifier to predict the top five species and shows some examples of the class with highest score. Please download our pre-trained [B-CNN(D,M)](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cub-dm.zip) and [svm](http://vis-www.cs.umass.edu/bcnn/download/svm_cub_vdm.mat) models for this demo. Choose your favorite bird images, edit the line 4 to 13 for your local setting and run the demo.

###Fine-tune BCNN models###
See run_experiments_bcnn_train.m for fine-tuning B-CNN model. Note that this code catching all the intermediate results during fine-tuning takes about 200GB disk storage.

Quick Start to fine-tune B-CNN(M,M) model

1. download CUB-200-2011 dataset
1. edit opts.cubDir=CUBROOT in model_setup.m, CUBROOT is the location of cub dataset
1. download imagenet-vgg-m model
1. set the path of the model in run_experiments_bcnn_train.m

        bcnnmm.opts = {..
           'type', 'bcnn', ...
           'modela', PRETRAINMODEL, ...
    'layera', 14,...
    'modelb', PRETRAINMODEL, ...
    'layerb', 14,...
    'shareWeight', true,...
    } ;

###Evaluation on dataset###
See run_experiments.m for training svm and testing.