# B-CNN: Bilinear CNNs for fine-grained visual recognition

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

Link to the [project page](http://vis-www.cs.umass.edu/bcnn).

### Fine-grained classification results


Method         | Birds 	   | Birds + box | Aircrafts | Cars
-------------- |:--------:|:----------:|:--------:|:------------:
B-CNN [M,M]    | 78.1%        | 77.5%        | 77.9%   | 86.5%
B-CNN [D,M]    | 84.1%        | 85.1%        | 83.9%   | 91.3%
B-CNN [D,D]    | 84.0%        | 84.8%        | 84.1%   | 90.6%

* Dataset details:
	* Birds: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Birds + box uses bounding-boxes at training and test time.
	* Aircrafts: [FGVC aircraft dataset](http://www.robots.ox.ac.uk/~vgg/data/oid/)
	* Cars: [Stanford cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* These results are with domain specific fine-tuning. For more details see the updated [B-CNN tech report](http://arxiv.org/abs/1504.07889).
* The pre-trained models are available (see below).

### Installation

This code depends on [VLFEAT](http://www.vlfeat.org) and [MatConvNet](http://www.vlfeat.org/matconvnet). Follow instructions on their project pages to install them first. Our code is built on MatConvNet version `1.0-beta8`. Version `1.0-beta9` which includes speedup by `cudnn` also works. To retrieve a particular version of MatConvNet using git type:

	>> git fetch --tags
	>> git checkout tags/v1.0-beta8
      
Once these are installed edit the `setup.m` to run the corresponding `setup` scripts.

The code implements the bilinear combination layer in symmetic and assymetic CNNs and contains scripts to fine-tune models and run experiments on several fine-grained recognition datasets. We also provide pre-trained models.


### Pre-trained models

**ImageNet LSVRC 2012 pre-trained models:** Since we don't support the latest MatConvNet implementation, the pre-trained models download from MatConvNet page don't work properly here. We provide the links to download [vgg-m](http://vis-www.cs.umass.edu/bcnn/download/imagenet-vgg-m.mat) and [vgg-verydeep-16](http://vis-www.cs.umass.edu/bcnn/download/imagenet-vgg-verydeep-16.mat) in old format.

**Fine-tuned models:** We provide three B-CNN fine-trained models ([M,M], [D,M], and [D,D]) for each of CUB-200-2011, FGVC Aircraft and Cars dataset. Note that for [M,M] and [D,D], we run the symmetric model, where you can simply use the same network for both two streams. These can be downloaded [here](http://vis-www.cs.umass.edu/bcnn/download). You can download all the models by running `fetch_models.m`. The models will be stored in `data/models` directory.

### Fine-grained datasets

To run experiments download the datasets from various places and edit the `model_setup.m` file to point it to the location of each dataset. For instance, you can point to the birds dataset directory by setting `opts.cubDir = 'data/cub'`.

### Classification demo

The script `demo_test.m` takes an image and runs our pre-trained fine-grained bird classifier to predict the top five species and shows some examples of the class with highest score. Download our pre-trained [B-CNN [D,M]](http://vis-www.cs.umass.edu/bcnn/download/bcnn-cub-dm.zip) and [SVM](http://vis-www.cs.umass.edu/bcnn/download/svm_cub_vdm.mat) models for this demo and put them in the `data/models` directory. Choose your favorite bird images, edit lines 4 to 13 for your local setting and run the demo.

### Fine-tuning B-CNN models

See `run_experiments_bcnn_train.m` for fine-tuning a B-CNN model. Note that this code caches all the intermediate results during fine-tuning which takes about 200GB disk space.

Here are the steps to fine-tuning a B-CNN(M,M) model on the Birds dataset:

1. Download `CUB-200-2011` dataset (see link above)
1. Edit `opts.cubDir=CUBROOT` in `model_setup.m`, CUBROOT is the location of CUB dataset.
1. Download `imagenet-vgg-m` model (see link above)
1. Set the path of the model in `run_experiments_bcnn_train.m`. For example, set PRETRAINMODEL='data/model/imagenet-vgg-m.mat', to use the Oxford's VGG-M model trained on ImageNet LSVRC 2012 dataset. You also have to set the `bcnnmm.opts` to:

        bcnnmm.opts = {..
           'type', 'bcnn', ...
           'modela', PRETRAINMODEL, ...
           'layera', 14,...
           'modelb', PRETRAINMODEL, ...
           'layerb', 14,...
           'shareWeight', true,...
        } ;
        
	The option `shareWeight=true` implies that the blinear model uses the same CNN to extract both features resulting in a symmetric model. For assymetric models set `shareWeight=false`. Note that this roughly doubles the GPU memory requirement.

1. Once the fine-tuning is complete, you can train a linear SVM on the extracted features to evaluate the model. See `run_experiments.m` for training/testing using SVMs. You can simply set the MODELPATH to the location of the fine-tuned model by setting MODELPATH='data/ft-models/bcnn-cub-mm.mat' and the `bcnnmm.opts` to:

        bcnnmm.opts = {..
           'type', 'bcnn', ...
           'modela', MODELPATH, ...
           'layera', 14,...
           'modelb', MODELPATH, ...
           'layerb', 14,...
        } ;
        
1. And type ``>> run_experiments`` on the MATLAB command line. The results with be saved in the `opts.resultPath`.

### Running B-CNN on other datasets

The code can be used for other classification datasets as well. You have to implement the corresponding `>> imdb = <dataset-name>_get_database()` function that returns the `imdb` structure in the right format. Take a look at the `cub_get_database.m` file as an example.

### Acknowldgements

We thank MatConvNet and VLFEAT teams for creating and maintaining these excellent packages.