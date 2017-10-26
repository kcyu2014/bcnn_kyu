function run_pv_experiments()

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


% This code is used for testing different encoding schemes with svm


 %% fine-tuned bilinear pooling CNN
 
 % set other than modela to empty if your network has bilinear pooling
 % layers
 
  
  gspvdft.name = 'gspvdft' ;
  gspvdft.opts = {...
    'type', 'pvcnn', ...
    'modela', 'data/pvvd_exp/cub-seed-01/fine-tuned-model/final-model.mat', ...
    'layera', [],...
    'modelb', [], ...
    'layerb', [],...
    };


%   setupNameList = {'bcnnmmft', 'bcnnvdmft'};   % list of models to train and test
%   encoderList = {{bcnnmmft}, {bcnnvdmft}};
%   setupNameList = {'dcnnvdft'};
%   encoderList = {{dcnnvdft}};
%   datasetList = {{'minc', 1}};
  setupNameList = {'gspvdft'};
  encoderList = {{gspvdft}};
  datasetList = {{'cub', 1}};
  
  scales = [2, 2];

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
