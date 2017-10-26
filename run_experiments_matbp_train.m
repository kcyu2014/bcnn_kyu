function run_experiments_matbp_train()

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% This code is used for fine-tuning bilinear model

if(~exist('data', 'dir'))
    mkdir('data');
end
  % Use BCNN pipeline only switching bilinear to matbp
  matbpvd.name = 'matbpvd' ;
  matbpvd.opts = {...
    'type', 'matbp', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30, ...
    'modelb', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layerb', 30, ...
    'shareWeight', true,...
    };

  setupNameList = {'matbpvd'};
  encoderList = {{matbpvd}}; 
  % datasetList = {{'minc', 1}};  
  % datasetList = {{'aircraft-variant', 1}};  
  datasetList = {{'cub', 1}};
  for ii = 1 : numel(datasetList)
    dataset = datasetList{ii} ;
    if iscell(dataset)
      numSplits = dataset{2} ;
      dataset = dataset{1} ;
    else
      numSplits = 1 ;
    end
    for jj = 1 : numSplits
      for ee = 1: numel(encoderList)
        
          [opts, imdb] = model_setup('dataset', dataset, ...
			  'encoders', encoderList{ee}, ...
			  'prefix', sprintf('%s_exp', setupNameList{ee}), ...  % output folder name
			  'batchSize', 16, ...
			  'imgScale', 2, ...       % specify the scale of input images
			  'bcnnLRinit', true, ...   % do logistic regression to initilize softmax layer
			  'dataAugmentation', {'f2','none','none'},...      % do data augmentation [train, val, test]. Only support flipping for train set on current release.
			  'gpus', [], ...          %specify the GPU to use. 0 for using CPU
        'learningRate', reshape(repmat([1e-4, 1e-5, 1e-6], 20, 1), 3*20, 1), ...
			  'numEpochs', 70, ...
			  'momentum', 0.9, ...
			  'keepAspect', true, ...
			  'printDatasetInfo', true, ...
			  'fromScratch', false, ...
			  'rgbJitter', false, ...  
			  'useVal', false,...
        'numSubBatches', 1, ...
        'nonftbcnnDir', 'nonftmatbpcnn');
          imdb.images.set(imdb.images.set==3) = 2;
          imdb_bcnn_train_dag(imdb, opts);
      end
    end
  end
end

%{
The following are the setting we run in which fine-tuning works stable without GPU memory issues on Nvidia K40.
m-m model: batchSize 64, momentum 0.9
d-m model: batchSize 1, momentum 0.3
%}

