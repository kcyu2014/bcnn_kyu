function run_experiments()

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


% This code is used for testing different encoding schemes with svm


%% fully connected pooling 
  rcnn.name = 'rcnn' ;
  rcnn.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-m.mat', ...
    'layer', 19} ;

  rcnnvd.name = 'rcnnvd' ;
  rcnnvd.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layer', 35} ;

%% fisher vector CNN
  dcnn.name = 'dcnn' ;
  dcnn.opts = {...
    'type', 'dcnn', ...
    'model', 'data/models/imagenet-vgg-m.mat', ...
    'layer', 14, ...
    'numWords', 64} ;

  dcnnvd.name = 'dcnnvd' ;
  dcnnvd.opts = {...
    'type', 'dcnn', ...
    'model', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layer', 30, ...
    'numWords', 64} ;

%% fisher vector SIFT
  dsift.name = 'dsift' ;
  dsift.opts = {...
    'type', 'dsift', ...
    'numWords', 256, ...
    'numPcaDimensions', 80} ;

%% bilinear pooling CNN
  bcnnmm.name = 'bcnnmm' ;
  bcnnmm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', [], ...   % set to empty when use two identical networks
    'layerb', 14
    } ;

  bcnnvdm.name = 'bcnnvdm' ;
  bcnnvdm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/models/imagenet-vgg-m.mat', ...
    'layerb', 14
    } ;

  bcnnvdvd.name = 'bcnnvdvd' ;
  bcnnvdvd.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', [], ...
    'layerb', 30,...
    };

 %% fine-tuned bilinear pooling CNN
 
 % set other than modela to empty if your network has bilinear pooling
 % layers
 
  bcnnmmft.name = 'bcnnmmft' ;
  bcnnmmft.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft_models/bcnn-cub-mm-net.mat', ...
    'layera', [],...
    'modelb', [], ...
    'layerb', []
    } ;

  bcnnvdmft.name = 'bcnnvdmft' ;
  bcnnvdmft.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft_models/bcnn-cub-dm.mat', ...
    'layera', [],...
    'modelb', [], ...
    'layerb', []
    } ;

  bcnnvdvdft.name = 'bcnnvdvdft' ;
  bcnnvdvdft.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft_models/bcnn-cub-dd.mat', ...
    'layera', [],...
    'modelb', [], ...
    'layerb', [],...
    };


%   setupNameList = {'bcnnmmft', 'bcnnvdmft'};   % list of models to train and test
%   encoderList = {{bcnnmmft}, {bcnnvdmft}};
  setupNameList = {'bcnnvdvd'};
  encoderList = {{bcnnvdvd}};
  datasetList = {{'minc', 5}};
  
  scales = [2, 2];


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
        % train and test the model
        model_train(...
          'dataset', dataset, ...
          'seed', jj, ...
          'encoders', encoderList{ee}, ...
          'prefix', 'exp', ...              % name of the output folder
          'suffix', setupNameList{ee}, ...
          'printDatasetInfo', ee == 1, ...
          'gpus', [1], ...
		  'imgScale', scales(ee), ...  
          'dataAugmentation', 'f2') ;       %flipping for data augmentation. "none" for no augmentation
      end
    end
  end
end
