# B-CNN: Bilinear CNNs for fine-grained visual recognition

Created by Tsung-Yu Lin, Aruni RoyChowdhury and Subhransu Maji at UMass Amherst
### Introduction

This repository contains the code for reproducing the results in ICCV 2015 paper:

	@inproceedings{lin2015bilinear,
        Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
        Title = {Bilinear CNNs for Fine-grained Visual Recognition},
        Booktitle = {International Conference on Computer Vision (ICCV)},
        Year = {2015}
    }
	
The code is tested on Ubuntu 14.04 using NVIDIA  Titan X GPU and MATLAB R2014b. Recently, we have upgraded the code to support **dag** implementation. Meanwhile, the implementation of bilinear pooling layers and our customized layers are wrapped into a separate [bcnn-package](https://bitbucket.org/tsungyu/bcnn-package).

Link to the [project page](http://vis-www.cs.umass.edu/bcnn).

### Fine-grained classification results


Method         | Birds 	    | Birds + box  | Aircrafts | Cars
-------------- |:---------:|:------------:|:---------:|:-------:
B-CNN [M,M]    | 78.1%     | 80.4%        | 77.9%     | 86.5%
B-CNN [D,M]    | 84.1%     | 85.1%        | 83.9%     | 91.3%
B-CNN [D,D]    | 84.0%     | 84.8%        | 84.1%     | 90.6%

* Dataset details:
	* Birds: [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Birds + box uses bounding-boxes at training and test time.
	* Aircrafts: [FGVC aircraft dataset](http://www.robots.ox.ac.uk/~vgg/data/oid/)
	* Cars: [Stanford cars dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* These results are with domain specific fine-tuning. For more details see the updated [B-CNN tech report](http://arxiv.org/abs/1504.07889).
* The pre-trained models are available (see below).

### Installation


This code depends on [VLFEAT](http://www.vlfeat.org) and [MatConvNet](http://www.vlfeat.org/matconvnet) and [bcnn-package](https://bitbucket.org/tsungyu/bcnn-package). They are pre-defined as submodules for this project. To download the code, type:

	>> git submodule init
	>> git submodule update

Follow instructions on [VLFEAT](http://www.vlfeat.org) and [MatConvNet](http://www.vlfeat.org/matconvnet) project pages to install them first. Our code is built on MatConvNet version `1.0-beta18`. To retrieve a particular version of MatConvNet using git, cd to MatConvNet folder and type:

	>> git fetch --tags
	>> git checkout tags/v1.0-beta18
      
Once these are installed edit the `setup.m` to run the corresponding `setup` scripts.

The implementation of the bilinear combination layer in symmetic and assymetic CNNs is included in the [bcnn-package](https://bitbucket.org/tsungyu/bcnn-package). This code contains scripts to fine-tune models and run experiments on several fine-grained recognition datasets. We also provide pre-trained models.


### Pre-trained models

**ImageNet LSVRC 2012 pre-trained models:** We use vgg-m and vgg-verydeep-16 as our basic models. Please download the models from matconvnet [pre-trained models](http://www.vlfeat.org/matconvnet/pretrained/) page.

**Fine-tuned models:** We provide three B-CNN fine-trained models ([M,M], [D,M], and [D,D]) and SVM models trained on respective bcnn features for each of CUB-200-2011, FGVC Aircraft and Cars dataset. Note that for [M,M] and [D,D], we run the symmetric model, where you can simply use the same network for both two streams. These can be downloaded individually [here](http://maxwell.cs.umass.edu/bcnn/models2). 


You can also download all the model files as a tar.gz [here](http://maxwell.cs.umass.edu/bcnn/models2.tar.gz).

### Fine-grained datasets

To run experiments download the datasets from various places and edit the `model_setup.m` file to point it to the location of each dataset. For instance, you can point to the birds dataset directory by setting `opts.cubDir = 'data/cub'`.

### Classification demo

The script `bird_demo` takes an image and runs our pre-trained fine-grained bird classifier to predict the top five species and shows some examples images of the class with the highest score. If you haven't already done so, download our pre-trained [B-CNN [D,M]](http://maxwell.cs.umass.edu/bcnn/models/bcnn-cub-dm) and [SVM](http://maxwell.cs.umass.edu/bcnn/models/svm_cub_vdm.mat) models for this demo and locate them in `data/models`. In addition, download the [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset to `data/cub` as well. You can follow our default setting or edit `opts` in the script to point it to the models and dataset. If you have GPU installed on your machine, set `opts.useGpu=true` to speedup the computation. You should see the following output when you run `bird_demo()`:

	>> bird_demo();
	0.09s to load imdb.
	1.63s to load models into memory.
	Top 5 prediction for test_image.jpg:
	064.Ring_billed_Gull
	059.California_Gull
	147.Least_Tern
	062.Herring_Gull
	060.Glaucous_winged_Gull
	3.80s to make predictions [GPU=0]

To run it on your own images run `bird_demo('imgPath', 'favorite-bird.jpg');`. Classification roughlly takes 4s per image on my laptop on a CPU. On an NVIDIA K40 GPU with bigger batch sizes you should roughly get a throughput of 8 images/second with the `B-CNN [D,M]` model.

### Using B-CNN models
`run_experments.m` extracts B-CNN features and trains a svm classifier on fine-grained categories. Following shows how to setup B-CNN models:

1. Symmetric B-CNN: extracts the self outer-product of features at *'layera'*.
	
        bcnn.opts = {..
           'type', 'bcnn', ...
           'modela', PRETRAINMODEL, ...
           'layera', 14,...
           'modelb', [], ...
           'layerb', [],...
        } ;

2. Cross layer B-CNN: extracts the outer-product between features at *'layera'* and *'layerb'* using the same CNN.
	
        bcnn.opts = {..
           'type', 'bcnn', ...
           'modela', PRETRAINMODEL, ...
           'layera', 14,...
           'modelb', [], ...
           'layerb', 12,...
        } ;
        

3. Asymmetric B-CNN: extracts the outer-product between features from CNN *'modela'* at *'layera'* and CNN *'modelb'* at *'layerb'*.

        bcnn.opts = {..
           'type', 'bcnn', ...
           'modela', PRETRAINMODEL_A, ...
           'layera', 30,...
           'modelb', PRETRAINMODEL_B, ...
           'layerb', 14,...
        } ;
        
4. Fine-tuned B-CNN: If you fine-tune a B-CNN network (see next section), you can evaluate the model using:

        bcnn.opts = {..
           'type', 'bcnn', ...
           'modela', FINE-TUNED_MODEL, ...
           'layera', [],...
           'modelb', [], ...
           'layerb', [],...
        } ;

### Fine-tuning B-CNN models

See `run_experiments_bcnn_train.m` for fine-tuning a B-CNN model. Note that this code caches all the intermediate results during fine-tuning which takes about 200GB disk space.

Here are the steps to fine-tuning a B-CNN [M,M] model on the CUB dataset:

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
        
	The option `shareWeight=true` implies that the blinear model uses the same CNN to extract both features resulting in a symmetric model. For assymetric models set `shareWeight=false`. Note that this roughly doubles the GPU memory requirement. The `cnn_train()` provided from MatConvNet requires the setup of validation set. You need to prepare a validation set for the datasets without pre-defined validation set.

1. Once the fine-tuning is complete, you can train a linear SVM on the extracted features to evaluate the model. See `run_experiments.m` for training/testing using SVMs. You can simply set the MODELPATH to the location of the fine-tuned model by setting MODELPATH='data/ft-models/bcnn-cub-mm.mat' and the `bcnnmm.opts` to:

        bcnnmm.opts = {..
           'type', 'bcnn', ...
           'modela', MODELPATH, ...
           'layera', [],...
           'modelb', [], ...
           'layerb', [],...
        } ;
        
1. And type ``>> run_experiments()`` on the MATLAB command line. The results with be saved in the `opts.resultPath`.

### Implementation details

The asymmetric B-CNN model is implemented using two networks whose feature outputs are bilinearly combined followed by normalization and softmax loss layers. The network is constructed using DagNN structure. You can find the details in `initializeNetworksTwoStreams()` and `bcnn_train_dag()`.

When the same network is used to extract both features, the symmetric B-CNN model is implemented as a single network architecture consisting of `bilinearpool`, `sqrt`, and `l2norm` layers on the top of `convolutional` layers. This implementation is about twice as fast and memory efficient than asymmetric implementaion. You can find the details in `initializeNetworkSharedWeights()` and `bcnn_train_simplenn()`.


The code for B-CNN is implemented in the following MATLAB functions:

1. `vl_bilinearnn()	`: This extends `vl_simplenn()` of the MatConvNet library to include the bilinear layers.
1. `vl_nnbilinearpool()`: Bilinear feature pooling with outer product with itself.
1. `vl_nnbilinearclpool()`: Bilinear feature pooling with outer product of two different features. Current version only supports the same resolution of two feature outputs.
1. `vl_nnsqrt()`: Signed square-root normalization.
1. `vl_nnl2norm()`: L2 normalization.

### Running B-CNN on other datasets

The code can be used for other classification datasets as well. You have to implement the corresponding `>> imdb = <dataset-name>_get_database()` function that returns the `imdb` structure in the right format. Take a look at the `cub_get_database.m` file as an example.

### Acknowldgements

We thank MatConvNet and VLFEAT teams for creating and maintaining these excellent packages.