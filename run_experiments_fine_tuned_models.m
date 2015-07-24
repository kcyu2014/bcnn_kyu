function run_experiments_fine_tuned_models()

  rcnn.name = 'rcnn' ;
  rcnn.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-m.mat', ...
    'layer', 19} ;

  rcnnvd.name = 'rcnnvd' ;
  rcnnvd.opts = {...
    'type', 'rcnn', ...
    'model', 'data/models/imagenet-vgg-verydeep-19.mat', ...
    'layer', 41} ;

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

  dsift.name = 'dsift' ;
  dsift.opts = {...
    'type', 'dsift', ...
    'numWords', 256, ...
    'numPcaDimensions', 80} ;

  bcnnmm.name = 'bcnnmm' ;
  bcnnmm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_mm/aircraft-variant-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/bcnn-train_mm/aircraft-variant-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 14,...
    } ;

  bcnnmmpca.name = 'bcnnmmpca' ;
  bcnnmmpca.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_mm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/bcnn-train_mm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 15,...
    } ;
    
  bcnnvdmpca.name = 'bcnnvdmpca' ;
  bcnnvdmpca.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_vdm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/bcnn-train_vdm_one_pca_64_norelu/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 15,...
    } ;

  bcnnvdm.name = 'bcnnvdm' ;
  bcnnvdm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_vdm/cars-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/bcnn-train_vdm/cars-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 14,...
    } ;

  bcnnvdvd.name = 'bcnnvdvd' ;
  bcnnvdvd.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/models/imagenet-vgg-verydeep-16.mat', ...
    'layerb', 30,...
    };
 
  setupNameList = {'bcnnvdm'};
  encoderList = {{bcnnvdm}};    
%   setupNameList = {'bcnnvdvd', 'bcnnvdm', 'bcnnmm'};
%   encoderList = {{bcnnvdvd}, {bcnnvdm}, {bcnnmm}};
%   setupNameList = {'dcnn', 'dcnnvd', 'dsift'};
%   encoderList = {{dcnn}, {dcnnvd}, {dsift}};
  datasetList = {{'cars', 1} };

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
        model_train(...
          'dataset', dataset, ...
          'seed', jj, ...
          'encoders', encoderList{ee}, ...
          'prefix', 'bcnn-train-fine-tuned_vdm', ...
          'suffix', setupNameList{ee}, ...
          'printDatasetInfo', ee == 1, ...
          'useGpu', true) ;
      end
    end
  end
end
