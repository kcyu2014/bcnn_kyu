function imdb_bcnn_train(imdb, opts, varargin)
% Train a bilinear CNN model on a dataset supplied by imdb

opts.lite = false ;
opts.numFetchThreads = 0 ;
opts.train.batchSize = 256 ;
opts.train.numEpochs = 45 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.prefetch = false ;
opts.train.learningRate = [0.001*ones(1, 10) 0.001*ones(1, 10) 0.001*ones(1,10)] ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;


% opts.lite = false ;
% opts.numFetchThreads = 0 ;
opts.inittrain.weightDecay = 0 ;
opts.inittrain.batchSize = 256 ;
opts.inittrain.numEpochs = 300 ;
opts.inittrain.continue = true ;
opts.inittrain.useGpu = false ;
opts.inittrain.prefetch = false ;
opts.inittrain.learningRate = 0.001 ;
opts.inittrain.expDir = fullfile(opts.expDir, 'init') ;
opts.inittrain.nonftbcnnDir = fullfile(opts.expDir, 'nonftbcnn');


if(opts.useGpu)
    opts.train.useGpu = opts.useGpu;
    opts.inittrain.useGpu = opts.useGpu;
end



bcnn_net = initializeNetwork(imdb, opts) ;
fn = getBatchWrapper(bcnn_net.neta.normalization, opts.numFetchThreads) ;

[bcnn_net,info] = bcnn_train(bcnn_net, fn, imdb, opts.train, 'batchSize', opts.batchSize, 'conserveMemory', true) ;


if(~exist(fullfile(opts.expDir, 'fine-tuned-model'), 'dir'))
    mkdir(fullfile(opts.expDir, 'fine-tuned-model'))
end
% [~, namea, ~] = fileparts(encoderOpts.modela);
% [~, nameb, ~] = fileparts(encoderOpts.modelb);
% saveNetwork(fullfile(opts.expDir, 'fine-tuned-model', ['fine-tuned-neta-', namea, '.mat']), bcnn_net.neta);
% saveNetwork(fullfile(opts.expDir, 'fine-tuned-model', ['fine-tuned-netb-', nameb, '.mat']), bcnn_net.netb);

saveNetwork(fullfile(opts.expDir, 'fine-tuned-model', 'final-model-neta.mat'), bcnn_net.neta);
saveNetwork(fullfile(opts.expDir, 'fine-tuned-model', 'final-model-netb.mat'), bcnn_net.netb);

% -------------------------------------------------------------------------
function saveNetwork(fileName, net)
% -------------------------------------------------------------------------
layers = net.layers;
% 
% % Replace the last layer with softmax
% layers{end}.type = 'softmax';
% layers{end}.name = 'prob';

% Remove fields corresponding to training parameters
ignoreFields = {'filtersMomentum', ...
                'biasesMomentum',...
                'filtersLearningRate',...
                'biasesLearningRate',...
                'filtersWeightDecay',...
                'biasesWeightDecay',...
                'class'};
for i = 1:length(layers),
    layers{i} = rmfield(layers{i}, ignoreFields(isfield(layers{i}, ignoreFields)));
end
classes = net.classes;
normalization = net.normalization;
save(fileName, 'layers', 'classes', 'normalization');

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts, numThreads)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,numThreads) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatch(imdb, batch, opts, numThreads)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir '/'], imdb.images.name(batch)) ;
im = imdb_get_batch_bcnn(images, opts, ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;



function [im,labels] = getBatch_bcnn_fromdisk(imdb, batch, opts, numThreads)
% -------------------------------------------------------------------------
% images = strcat([imdb.imageDir '/'], imdb.images.name(batch)) ;
% im = imdb_get_batch(images, opts, ...
%                             'numThreads', numThreads, ...
%                             'prefetch', nargout == 0);
im = cell(1, numel(batch));
for i=1:numel(batch)
    load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
    im{i} = code;
end
im = cat(2, im{:});
im = reshape(im, 1, 1, size(im,1), size(im, 2));
labels = imdb.images.label(batch) ;



function net = initializeNetwork(imdb, opts)
% -------------------------------------------------------------------------


if (~isempty(opts.modela)&& ~isempty(opts.modelb) && ~opts.shareparameters)

% set the two networks
% net.neta = encoder.neta;
% net.netb = encoder.netb;

encoderOpts.type = 'bcnn';
encoderOpts.modela = [];
encoderOpts.layera = 14;
encoderOpts.modelb = [];
encoderOpts.layerb = 14;

encoderOpts = vl_argparse(encoderOpts, opts.encoders{1}.opts);

% net.neta = load(encoderOpts.modela); % Load model if specified
% net.netb = load(encoderOpts.modelb); % Load model if specified
%  
% net.neta.layers = net.neta.layers(1:encoderOpts.layera);
% net.netb.layers = net.netb.layers(1:encoderOpts.layerb);


encoder.neta = load(encoderOpts.modela);
encoder.neta.layers = encoder.neta.layers(1:encoderOpts.layera);
encoder.netb = load(encoderOpts.modelb);
encoder.netb.layers = encoder.netb.layers(1:encoderOpts.layerb);
encoder.regionBorder = 0.05;
encoder.type = 'bcnn';
encoder.normalization = 'sqrt_L2';
if opts.useGpu
    encoder.neta = vl_simplenn_move(encoder.neta, 'gpu') ;
    encoder.netb = vl_simplenn_move(encoder.netb, 'gpu') ;
    encoder.neta.useGpu = true ;
    encoder.netb.useGpu = true ;
else
    encoder.neta = vl_simplenn_move(encoder.neta, 'cpu') ;
    encoder.netb = vl_simplenn_move(encoder.netb, 'cpu') ;
    encoder.neta.useGpu = false ;
    encoder.netb.useGpu = false ;
end

net.neta = encoder.neta;
net.netb = encoder.netb;


numClass = length(imdb.classes.name);

netc.layers = {};

% randomly initialize the layers on top
scal = 1 ;
init_bias = 0.1;


for i=numel(net.neta.layers):-1:1
    if strcmp(net.neta.layers{i}.type, 'conv')
        idx = i;
        break;
    end
end
ch1 = numel(net.neta.layers{idx}.biases);

for i=numel(net.netb.layers):-1:1
    if strcmp(net.netb.layers{i}.type, 'conv')
        idx = i;
        break;
    end
end
ch2 = numel(net.netb.layers{idx}.biases);

dim = ch1*ch2;







%% get bilinear cnn features of images to initialize fully connected layer

trainIdx = find(ismember(imdb.images.set, [1 2]));
testIdx = find(ismember(imdb.images.set, 3));

if ~exist(opts.inittrain.nonftbcnnDir)
    mkdir(opts.inittrain.nonftbcnnDir)
    batchSize = 10000;
    for b=1:ceil(numel(trainIdx)/batchSize)
        idxEnd = min(numel(trainIdx), b*batchSize);
        idx = trainIdx((b-1)*batchSize+1:idxEnd);
        codeCell = encoder_extract_for_images(encoder, imdb, imdb.images.id(idx), 'concatenateCode', false);
        for i=1:numel(codeCell)
            code = codeCell{i};
            savefast(fullfile(opts.inittrain.nonftbcnnDir, ['bcnn_nonft_', num2str(idx(i), '%05d')]), 'code');
        end
    end
end




% Else initial model randomly
netc.layers = {};
net.netc.layers = {};



net.netc.layers{end+1} = struct('type', 'sqrt');
net.netc.layers{end+1} = struct('type', 'l2norm');

  
initialW = 0.001/scal *randn(1,1,dim, numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');


if exist(fullfile(opts.expDir, 'initial_fc.mat'))
    load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
%     netc.layers{1}.filtersLearningRate = 10;
%     netc.layers{1}.biasesLearningRate = 10;
%     
%     netc.layers{4}.filtersLearningRate = 10;
%     netc.layers{4}.biasesLearningRate = 10;
else
    %{
    % fully connected layer
    fcdim = 1000;
    netc.layers{end+1} = struct('type', 'conv', ...
        'filters', 0.001/scal *randn(1,1,dim, fcdim,'single'), ...
        'biases', init_bias.*ones(1, fcdim, 'single'), ...
        'stride', 1, ...
        'pad', 0, ...
        'filtersLearningRate', 1000, ...
        'biasesLearningRate', 1000, ...
        'filtersWeightDecay', 0, ...
        'biasesWeightDecay', 0) ;
        
	netc.layers{end+1} = struct('type', 'relu') ;
	netc.layers{end+1} = struct('type', 'dropout', ...
                           'rate', 0.5) ;
        
    netc.layers{end+1} = struct('type', 'conv', ...
        'filters', 0.001/scal *randn(1,1,fcdim, numClass,'single'), ...
        'biases', init_bias.*ones(1, numClass, 'single'), ...
        'stride', 1, ...
        'pad', 0, ...
        'filtersLearningRate', 1000, ...
        'biasesLearningRate', 1000, ...
        'filtersWeightDecay', 0, ...
        'biasesWeightDecay', 0) ;

%}
    
    
    netc.layers{end+1} = struct('type', 'conv', ...
        'filters', initialW, ...
        'biases', initialBias, ...
        'stride', 1, ...
        'pad', 0, ...
        'filtersLearningRate', 1000, ...
        'biasesLearningRate', 1000, ...
        'filtersWeightDecay', 0, ...
        'biasesWeightDecay', 0) ;
  
    
    netc.layers{end+1} = struct('type', 'softmaxloss') ;
    
    
    bcnndb = imdb;
    tempStr = sprintf('%05d\t', trainIdx);
    tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
    bcnndb.images.name = strcat('bcnn_nonft_', tempStr{1}');
    bcnndb.images.id = bcnndb.images.id(trainIdx);
    bcnndb.images.label = bcnndb.images.label(trainIdx);
    bcnndb.images.set = bcnndb.images.set(trainIdx);
    bcnndb.imageDir = opts.inittrain.nonftbcnnDir;
    %bcnndb.codes = code;
    
    [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn_fromdisk, opts.inittrain, ...
        'batchSize', opts.inittrain.batchSize, 'weightDecay', opts.inittrain.weightDecay, ...
        'conserveMemory', true, 'expDir', opts.inittrain.expDir);

    save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc', '-v7.3') ;
end

% net.netc.layers{end+1} = netc.layers{1};
% net.netc.layers{end+1} = netc.layers{2};


for i=1:numel(netc.layers)
	net.netc.layers{end+1} = netc.layers{i};
end
    

end


function code = encoder_extract_for_images(encoder, imdb, imageIds, varargin)
% -------------------------------------------------------------------------
opts.batchSize = 128 ;
opts.maxNumLocalDescriptorsReturned = 500 ;
opts.concatenateCode = true;
opts = vl_argparse(opts, varargin) ;

[~,imageSel] = ismember(imageIds, imdb.images.id) ;
imageIds = unique(imdb.images.id(imageSel)) ;
n = numel(imageIds) ;

% prepare batches
n = ceil(numel(imageIds)/opts.batchSize) ;
% batches = mat2cell(1:numel(imageIds), 1, [opts.batchSize * ones(1, n-1), numel(imageIds) - opts.batchSize*(n-1)]) ;
batches = mat2cell(1:numel(imageIds), 1, [opts.batchSize * ones(1, n-1), numel(imageIds) - opts.batchSize*(n-1)]) ;
batchResults = cell(1, numel(batches)) ;

% just use as many workers as are already available
numWorkers = matlabpool('size') ;
%parfor (b = 1:numel(batches), numWorkers)
for b = numel(batches):-1:1
  batchResults{b} = get_batch_results(imdb, imageIds, batches{b}, ...
                        encoder, opts.maxNumLocalDescriptorsReturned) ;
end
%{
code = zeros(size(batchResults{b}.code{1},1), numel(imageIds), 'single');
for b=1:numel(batches)
    m = numel(batches{b});
    for j=1:m
        code(:,batches{b}(j)) = batcheResults{b}.code{j};
    end
end
%}

code = cell(size(imageIds)) ;
for b = 1:numel(batches)
  m = numel(batches{b});
  for j = 1:m
      k = batches{b}(j) ;
      code{k} = batchResults{b}.code{j};
  end
end

if opts.concatenateCode
    clear batchResults
   code = cat(2, code{:}) ;
   % code = cell2mat(code);
end


function result = get_batch_results(imdb, imageIds, batch, encoder, maxn)
% -------------------------------------------------------------------------
m = numel(batch) ;
im = cell(1, m) ;
task = getCurrentTask() ;
if ~isempty(task), tid = task.ID ; else tid = 1 ; end
for i = 1:m
  fprintf('Task: %03d: encoder: extract features: image %d of %d\n', tid, batch(i), numel(imageIds)) ;
  im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{imdb.images.id == imageIds(batch(i))}));
  if size(im{i}, 3) == 1, im{i} = repmat(im{i}, [1 1 3]);, end; %grayscale image
end

if ~isfield(encoder, 'numSpatialSubdivisions')
  encoder.numSpatialSubdivisions = 1 ;
end
       code_ = get_bcnn_features(encoder.neta, encoder.netb,...
         im, ...
        'regionBorder', encoder.regionBorder, ...
        'normalization', encoder.normalization);
    
result.code = code_ ;
