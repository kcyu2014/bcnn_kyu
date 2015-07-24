% function run_predict_logreg()

clear

bcnnNetcPath = 'test_logreg';

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
    'modela', 'data/bcnn-train_mm/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/bcnn-train_mm/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 14,...
    } ;

  bcnnmmpca.name = 'bcnnmmpca' ;
  bcnnmmpca.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_mm_one_pca_64/cubcrop-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/bcnn-train_mm_one_pca_64/cubcrop-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
    'layerb', 16,...
    } ;

  bcnnvdm.name = 'bcnnvdm' ;
  bcnnvdm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/bcnn-train_vdm/cub-seed-01/fine-tuned-model/fine-tuned-neta-imagenet-vgg-verydeep-16.mat', ...
    'layera', 30,...
    'modelb', 'data/bcnn-train_vdm/cub-seed-01/fine-tuned-model/fine-tuned-netb-imagenet-vgg-m.mat', ...
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
 
  setupNameList = {'bcnnmmpca'};
  encoderList = {{bcnnmmpca}};    
%   setupNameList = {'bcnnvdvd', 'bcnnvdm', 'bcnnmm'};
%   encoderList = {{bcnnvdvd}, {bcnnvdm}, {bcnnmm}};
%   setupNameList = {'dcnn', 'dcnnvd', 'dsift'};
%   encoderList = {{dcnn}, {dcnnvd}, {dsift}};
  datasetList = {{'cubcrop', 1} };
  
 
 bcnnNetcPath = fullfile('data', bcnnNetcPath, [datasetList{1}{1}, '-seed-01'], 'cpunetc');
 load(bcnnNetcPath, 'netc')
 
    netc.layers{end} = struct('type', 'softmax') ;

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
			  'prefix', 'test_logreg');
          
%           
%         model_train(...
%           'dataset', dataset, ...
%           'seed', jj, ...
%           'encoders', encoderList{ee}, ...
%           'prefix', 'bcnn-train-fine-tuned_one_pca_64', ...
%           'suffix', setupNameList{ee}, ...
%           'printDatasetInfo', ee == 1, ...
%           'useGpu', true) ;
      end
    end
  end
  
  
  opts.modela = opts.encoders{1}.opts{4};
  opts.modelb = opts.encoders{1}.opts{8};
  opts.layera = opts.encoders{1}.opts{6};
  opts.layerb = opts.encoders{1}.opts{10};
  
       encoder.normalization = 'sqrt';
       encoder.neta = load(opts.modela);
       encoder.neta.layers = encoder.neta.layers(1:opts.layera);
       encoder.netb = load(opts.modelb);
       encoder.netb.layers = encoder.netb.layers(1:opts.layerb);
       if opts.useGpu,
           encoder.neta = vl_simplenn_move(encoder.neta, 'gpu');
           encoder.netb = vl_simplenn_move(encoder.netb, 'gpu');
           encoder.neta.useGpu = true;
           encoder.netb.useGpu = true;
       else
           encoder.neta = vl_simplenn_move(encoder.neta, 'cpu');
           encoder.netb = vl_simplenn_move(encoder.netb, 'cpu');
           encoder.neta.useGpu = false;
           encoder.netb.useGpu = false;

      end
%       code = encoder_extract_for_images(encoder, imdb, imdb.images.id) ;
%       savefast(opts.encoders{i}.codePath, 'code') ;
%     end
%     psi{i} = code ;
%     clear code ;
%   end
%   psi = cat(1, psi{:}) ;

test = ismember(imdb.images.set, 3) ;
batch = find(test);
m = numel(batch) ;
% m=100;
im = cell(1, m) ;
for i = 1:m
  disp(['encoding testing images ', num2str(i), '/', num2str(m)]);
  im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{imdb.images.id == batch(i)}));
  if size(im{i}, 3) == 1, im{i} = repmat(im{i}, [1 1 3]);, end; %grayscale image
end

  disp('computing bcnn codes')
  
  psi = get_bcnn_features(encoder.neta, encoder.netb,...
         im, ...
        'regionBorder', '0.05', ...
        'normalization', 'sqrt');
    
    psi = cat(2, psi{:});
    
    
    psi_sqrt = sign(psi).*sqrt(abs(psi));
    psi_sqrt_norm = arrayfun(@(x) norm(psi_sqrt(:,x)), 1:size(psi_sqrt,2));
    
    % derivatives of taking sqrt root and L2 normalization
    
    psi_n = bsxfun(@rdivide, psi_sqrt, psi_sqrt_norm);
    
%     d_psi = size(psi_n, 1);
    psi_n = reshape(psi_n, [1,1,size(psi_n,1),size(psi_n,2)]);
    
    
    
    res = [];
    
    pred = zeros(numel(netc.layers{1}.biases), m);
    for i=1:m
       res = vl_simplenn(netc, psi_n(:,:,:,i), [], res, ...
      'conserveMemory', true) ;
        pred(:,i) = res(end).x;
    end
    
    [~, y] = max(pred, [], 1);
    
    acc = sum(y == imdb.images.label(batch(1:m)))/m
    
    
    