function [opts, imdb] = model_setup(varargin)
setup ;

opts.seed = 1 ;
opts.batchSize = 128 ;
opts.numEpochs = 100;
opts.momentum = 0.9;
opts.keepAspect = false;
opts.useVal = false;
opts.useGpu = 1 ;
opts.regionBorder = 0.05 ;
opts.numDCNNWords = 64 ;
opts.numDSIFTWords = 256 ;
opts.numSamplesPerWord = 1000 ;
opts.printDatasetInfo = true ;
opts.excludeDifficult = true ;
opts.datasetSize = inf;
opts.encoders = {struct('type', 'rcnn', 'opts', {})} ;
opts.dataset = 'lfw' ;
opts.janusDir = '/scratch2/data/CS2-protocal';
opts.janusFaceDir = '/scratch2/data/CS2';
opts.facescrubDir = 'data/facescrub' ;
opts.mitDir = 'data/mit_indoor';
opts.cubDir = 'data/cub';
opts.dogDir = 'data/stanford_dogs';
opts.aircraftDir = 'data/fgvc-aircraft-2013b';
opts.modelnetDir = 'data/modelnet40toon';
opts.suffix = 'baseline' ;
opts.prefix = 'v1' ;
opts.model  = 'imagenet-vgg-m.mat';
opts.modela = 'imagenet-vgg-m.mat';
opts.modelb = 'imagenet-vgg-s.mat';
opts.layer  = 14;
opts.layera = 14;
opts.layerb = 14;
%opts.bcnn = false;
opts.bcnnScale = 1;
opts.bcnnLRinit = false;
opts.bcnnLayer = 14;
opts.dataAugmentation = {'none', 'none', 'none'};
[opts, varargin] = vl_argparse(opts,varargin) ;

opts.expDir = sprintf('data/%s/%s-seed-%02d', opts.prefix, opts.dataset, opts.seed) ;
opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
opts.resultPath = fullfile(opts.expDir, sprintf('result-%s.mat', opts.suffix)) ;

opts = vl_argparse(opts,varargin) ;

if nargout <= 1, return ; end

% Setup GPU if needed
if opts.useGpu
  gpuDevice(opts.useGpu) ;
end

% -------------------------------------------------------------------------
%                                                            Setup encoders
% -------------------------------------------------------------------------

models = {} ;
modelPath = {};
for i = 1:numel(opts.encoders)
  if isstruct(opts.encoders{i})
    name = opts.encoders{i}.name ;
    opts.encoders{i}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
    opts.encoders{i}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
    [md, mdpath] = get_cnn_model_from_encoder_opts(opts.encoders{i});
    models = horzcat(models, md) ;
    modelPath = horzcat(modelPath, mdpath);
%     models = horzcat(models, get_cnn_model_from_encoder_opts(opts.encoders{i})) ;
  else
    for j = 1:numel(opts.encoders{i})
      name = opts.encoders{i}{j}.name ;
      opts.encoders{i}{j}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
      opts.encoders{i}{j}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
      [md, mdpath] = get_cnn_model_from_encoder_opts(opts.encoders{i}{j});      
      models = horzcat(models, md) ;
      modelPath = horzcat(modelPath, mdpath);
%       models = horzcat(models, get_cnn_model_from_encoder_opts(opts.encoders{i}{j})) ;
    end
  end
end

% -------------------------------------------------------------------------
%                                                       Download CNN models
% -------------------------------------------------------------------------

for i = 1:numel(models)
    if ~exist(modelPath{i})
%   if ~exist(fullfile('data/models', models{i}))
        fprintf('downloading model %s\n', models{i}) ;
        vl_xmkdir('data/models') ;
        urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', models{i}),...
            modelPath{i}) ;
%     urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', models{i}),...
%       fullfile('data/models', models{i})) ;
    end
end

% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------

vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;

imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
if exist(imdbPath)
  imdb = load(imdbPath) ;
  return ;
end

switch opts.dataset
    case 'cubcrop'
        imdb = cub_get_database(opts.cubDir, true, false);
    case 'cub'
        imdb = cub_get_database(opts.cubDir, false, opts.useVal);
    case 'dogcrop'
        imdb = stanford_dogs_get_database(opts.dogDir, true);
    case 'dog'
        imdb = stanford_dogs_get_database(opts.dogDir, false);
    case 'mitindoor'
        imdb = mit_indoor_get_database(opts.mitDir);
    case 'facescrub'
        imdb = facescrub_get_database(opts.facescrubDir) ;
    case 'aircraft-variant'
        imdb = aircraft_get_database(opts.aircraftDir, 'variant');
    case 'aircraft-model'
        imdb = aircraft_get_database(opts.aircraftDir, 'model');
    case 'aircraft-family'
        imdb = aircraft_get_database(opts.aircraftDir, 'family');
    case 'modelnet'
        imdb = modelnet_get_database(opts.modelnetDir);
    case 'janus-train'
        imdb = janus_face_get_database_train(opts.janusDir, opts.janusFaceDir, ...
            opts.seed, 'classify' );
    case 'janus-classify'
        imdb = janus_face_get_database(opts.janusDir, opts.janusFaceDir, ...
                                    opts.seed, 'classify' );
    otherwise
        error('Unknown dataset %s', opts.dataset) ;
end

save(imdbPath, '-struct', 'imdb') ;

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end

% -------------------------------------------------------------------------
function [model, modelPath] = get_cnn_model_from_encoder_opts(encoder)
% -------------------------------------------------------------------------
p = find(strcmp('model', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = {[m e]} ;
  modelPath = encoder.opts{p+1};
else
  model = {} ;
  modelPath = {};
end

% bilinear cnn models
p = find(strcmp('modela', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
  modelPath = horzcat(modelPath, encoder.opts{p+1});
end
p = find(strcmp('modelb', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = horzcat(model,{[m e]}) ;
  modelPath = horzcat(modelPath, encoder.opts{p+1});
end


