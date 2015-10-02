## B-CNN: Bilinear CNNs for fine-grained visual recognition

Created by Tsung-Yu Lin, Aruni RoyChowdhury and Subhransu Maji at UMass Amherst

### Introduction

This repository contains the code for reproducing the results in ICCV 2015 paper:

	@inproceedings{lin2015bilinear,
        Author = {Tsung-Yu Lin, Aruni RoyChoudhury, and Subhransu Maji},
        Title = {Blinear CNNs for Fine-grained Visual Recognition},
        Booktitle = {International Conference on Computer Vision (ICCV)},
        Year = {2015}
    }
	
The code is tested on Ubuntu 14.04 using NVIDIA K40 GPU and MATLAB R2014b.

### Fine-grained classification results


Method         | Birds 	   | Birds + box | Aircrafts | Cars
-------------- |:--------:|:----------:|:--------:|:------------:
B-CNN [M,M]    | 78.1%        | 77.5%        | 77.9%   | 86.5%
B-CNN [D,M]    | 84.1%        | 85.1%        | 83.9%   | 91.3%
B-CNN [D,D]    | 84.0%        | 84.8%        | 84.1%   | 90.6%

* Dataset details:
	* Birds: [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The birds + box is when object bounding-boxes are used both at training and test time.
	* Aircrafts: [FGVC aircraft dataset](http://www.robots.ox.ac.uk/~vgg/data/oid/)
	* Cars: [Stanford cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* These results are with domain specific fine-tuning. For more details see the updated [B-CNN tech report](http://arxiv.org/abs/1504.07889).
* The pre-trained models are available (see below).

### Installation

This code depends on VLFEAT and MatConvNet. Please follow instructions on their project pages to install them first. Our code is built on MatConvNet version 1.0-beta8. Version 1.0-beta9 which include speedup by cudnn also works fine. To retrieve a particular version of MatConvNet using git type:

      git fetch --tags
      git checkout tags/v1.0-beta8
      
Once done, edit the file `setup.m` to link the packages.
      
The code implements the bilinear combination layer in symmetic and assymetic CNNs and contains scripts to fine-tune models and run experiments on several fine-grained recognition datasets. We also provide pre-trained models.


### Pre-trained models

ImageNet pre-trained models: Since we don't support the latest MatConvNet implementation, the pre-trained models download from MatConvNet page don't work properly here. We provide the links to download [vgg-m](http://vis-www.cs.umass.edu/bcnn/download/imagenet-vgg-m.mat) and [vgg-verydeep-16](http://vis-www.cs.umass.edu/bcnn/download/imagenet-vgg-verydeep-16.mat) in old format.

We provide three BCNN pre-trained models ([M,M], [D,M], and [D,D]) for each of CUB-200-2011, FGVC Aircraft and Cars dataset. Note that for [M,M] and [D,D], we run the symmetric model, where you can simply use the same network for both two streams during encoding.

The pretrained models can be downloaded [here](http://vis-www.cs.umass.edu/bcnn/download).


### Datasets

To run experiments you have to download the datasets from various places and edit the `model_setup.m` file to point it to the location of each dataset.

### Classification demo

The script `demo_test.m` takes an input image and runs our pre-trained fine-grained bird classifier to predict the top five species and shows some examples of the class with highest score. Please download our pre-trained [B-CNN [D,M]](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cub-dm.zip) and [SVM](http://vis-www.cs.umass.edu/bcnn/download/svm_cub_vdm.mat) models for this demo. Choose your favorite bird images, edit lines 4 to 13 for your local setting and run the demo.

### Fine-tuning B-CNN models

See `run_experiments_bcnn_train.m` for fine-tuning a B-CNN model. Note that this code caches all the intermediate results during fine-tuning which takes about 200GB disk space.

Here are the steps to fine-tuning a B-CNN(M,M) model:

1. Download CUB-200-2011 dataset
1. Edit opts.cubDir=CUBROOT in `model_setup.m`, CUBROOT is the location of cub dataset
1. Download imagenet-vgg-m model
1. Set the path of the model in `run_experiments_bcnn_train.m`. For example, set PRETRAINMODEL='data/model/imagenet-vgg-m.mat', to use the Oxford's VGG-M model trained on ImageNet LSVRC 2012 dataset. You also have to set the `bcnnmm.opts` to:

        bcnnmm.opts = {..
           'type', 'bcnn', ...
           'modela', PRETRAINMODEL, ...
           'layera', 14,...
           'modelb', PRETRAINMODEL, ...
           'layerb', 14,...
           'shareWeight', true,...
        } ;
        
The `shareWeight=true` means the resulting model is a symmetric one. For assymetric models set `shareWeight=false`. Note that this rouughly doubles the memory requirements. 

Once the fine-tuning is complete, you can train a linear SVM on the extracted features to evaluate the model. See `run_experiments.m` for training/testing using SVMs. You can simply set the MODELPATH to the location of the fine-tuned model by setting MODELPATH='data/ft-models/bcnn-cub-mm.mat' and the `bcnnmm.opts` to:

        bcnnmm.opts = {..
           'type', 'bcnn', ...
           'modela', MODELPATH, ...
           'layera', 14,...
           'modelb', MODELPATH, ...
           'layerb', 14,...
        } ;
        
And type ``>> run_experiments`` on the MATLAB command line.

### Acknowldgements

We thank MatConvNet and VLFEAT teams for creating and maintaining these excellent packages.
        