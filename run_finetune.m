function run_finetune()

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


% This code is used for testing different encoding schemes with svm




%% fisher vector CNN

  dcnnvdft.name = 'dcnnvdft';
  dcnnvdft.opts = { ...
      'type', 'dcnn', ...
      'model',  'data/ft-cnn/minc-seed-01/final-model.mat', ...
      'layer', 30, ...
      'numWords', 64};



 %% fine-tuned bilinear pooling CNN
 
 % set other than modela to empty if your network has bilinear pooling
 % layers
 

  bcnnvdvdft.name = 'bcnnvdvdft' ;
  bcnnvdvdft.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/ft-cnn/minc-seed-01/final-model.mat', ...
    'layera', 30,...
    'modelb', [], ...
    'layerb', 30,...
    };


%   setupNameList = {'bcnnmmft', 'bcnnvdmft'};   % list of models to train and test
%   encoderList = {{bcnnmmft}, {bcnnvdmft}};
  setupNameList = {'bcnnvdvdft'};
  encoderList = {{bcnnvdvdft}};
  datasetList = {{'minc', 1}};
  
  scales = [1, 1];

  sprintf('Experiment with model %s', setupNameList{1});
  for ii = 1 : numel(datasetList)
    dataset = datasetList{ii} ;
    if iscell(dataset)
      numSplits = dataset{2} ;
      dataset = dataset{1} ;
    else
      numSplits = 1 ;
    end
%     for jj = 1 : numSplits
    for ee = 1: numel(encoderList)
      % train and test the model
      model_train(...
        'dataset', dataset, ...
        'seed', numSplits, ...
        'encoders', encoderList{ee}, ...
        'prefix', 'exp', ...              % name of the output folder
        'suffix', setupNameList{ee}, ...
        'printDatasetInfo', ee == 1, ...
        'gpus', 1, ...
        'imgScale', scales(ee), ...  
        'dataAugmentation', 'f2') ;       %flipping for data augmentation. "none" for no augmentation
    end
%     end
  end
end
