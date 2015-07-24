function imdb_bcnn_train_multilabel(imdb, opts, varargin)
% Train a bilinear CNN model on a dataset supplied by imdb

opts.lite = false ;
opts.numFetchThreads = 0 ;
opts.train.batchSize = 256 ;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.prefetch = false ;
%opts.train.learningRate = [0.001*ones(1, 10) 0.0001*ones(1, 10) 0.00001*ones(1,10)] ;
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


if(opts.useGpu)
    opts.train.useGpu = opts.useGpu;
    opts.inittrain.useGpu = opts.useGpu;
end

encoderOpts.type = 'bcnn';
encoderOpts.modela = [];
encoderOpts.layera = 14;
encoderOpts.modelb = [];
encoderOpts.layerb = 14;
encoderOpts = vl_argparse(encoderOpts, opts.encoders{1}.opts);

encoder.neta = load(encoderOpts.modela);
encoder.neta.layers = encoder.neta.layers(1:encoderOpts.layera);
encoder.netb = load(encoderOpts.modelb);
encoder.netb.layers = encoder.netb.layers(1:encoderOpts.layerb);
encoder.regionBorder = 0.05;
encoder.type = 'bcnn';
encoder.normalization = 'sqrt';


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


bcnn_net = initializeNetwork(imdb, encoder, opts) ;
fna = getBatchWrapper(bcnn_net.neta.normalization, opts.numFetchThreads) ;
fnb = getBatchWrapper(bcnn_net.netb.normalization, opts.numFetchThreads) ;
% [bcnn_net,info] = bcnn_train(bcnn_net, imdb, fna, fnb, opts.inittrain, 'batchSize', opts.batchSize, 'conserveMemory', true) ;

im = cell(numel(imdb.images.label),1);
for i=1:numel(im)
    fprintf('reading image %d.\n', i);
    im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{i}));
    if size(im{i}, 3) == 1, im{i} = repmat(im{i}, [1 1 3]); end; %grayscale image
end

[bcnn_net,info] = bcnn_train(bcnn_net, encoder, im, imdb, opts.train, 'batchSize', opts.batchSize, 'conserveMemory', true) ;

if(~exist(fullfile(opts.expDir, 'fine-tuned-model'), 'dir'))
    mkdir(fullfile(opts.expDir, 'fine-tuned-model'))
end
[~, namea, ~] = fileparts(encoderOpts.modela);
[~, nameb, ~] = fileparts(encoderOpts.modelb);
saveNetwork(fullfile(opts.expDir, 'fine-tuned-model', ['fine-tuned-neta-', namea, '.mat']), bcnn_net.neta);
saveNetwork(fullfile(opts.expDir, 'fine-tuned-model', ['fine-tuned-netb-', nameb, '.mat']), bcnn_net.netb);


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
im = imdb_get_batch(images, opts, ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;



function net = initializeNetwork(imdb, encoder, opts)
% -------------------------------------------------------------------------

net.neta = encoder.neta;
net.netb = encoder.netb;



scal = 1 ;
init_bias = 0.1;
numClass = length(imdb.classes.name);




%% get bilinear cnn features of images to initialize fully connected layer


if exist(fullfile(opts.expDir, 'ptBcnnCode.mat'))
    load(fullfile(opts.expDir, 'ptBcnnCode.mat'), 'code') ;
else
    encoderInit = encoder;
    encoderInit.normalization = 'none';
    code = encoder_extract_for_images(encoderInit, imdb, imdb.images.id) ;
    savefast(fullfile(opts.expDir, 'ptBcnnCode.mat'), 'code') ;
end

code = sqrt(code);
c_norm = arrayfun(@(x) norm(code(:,x)), 1:size(code,2));
code = bsxfun(@rdivide, code, c_norm);


% Else initial model randomly
netc.layers = {};
bi_d = size(code, 1);


imdbLabel = imdb.images.label;

trainIdx = ismember(imdb.images.set, [1 2]) ;
testIdx = ismember(imdb.images.set, 3) ;



initialW = 0.001/scal *randn(1,1,bi_d, numClass,'single');
% initialW = 0.001/scal *randn(bi_d, numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');


if exist(fullfile(opts.expDir, 'initial_fc.mat'))
%if false
    load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
else
    
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
    bcnndb.codes = code;
    
    [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn, opts.inittrain, ...
        'batchSize', opts.inittrain.batchSize, 'weightDecay', opts.inittrain.weightDecay, ...
        'conserveMemory', true, 'expDir', opts.inittrain.expDir);

    save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
end

net.netc = netc;


%{
% Other details
net.normalization.imageSize = [227, 227, 3] ;
net.normalization.interpolation = 'bicubic' ;
net.normalization.border = 256 - net.normalization.imageSize(1:2) ;
net.normalization.averageImage = [] ;
net.normalization.keepAspect = true ;
%}




function [im,labels] = getBatch_bcnn(imdb, batch)
% -------------------------------------------------------------------------
% images = strcat([imdb.imageDir '/'], imdb.images.name(batch)) ;
% im = imdb_get_batch(images, opts, ...
%                             'numThreads', numThreads, ...
%                             'prefetch', nargout == 0);
im = reshape(imdb.codes(:,batch), 1,1,size(imdb.codes,1), numel(batch));
labels = imdb.images.label(batch) ;


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
batches = mat2cell(1:numel(imageIds), 1, [opts.batchSize * ones(1, n-1), numel(imageIds) - opts.batchSize*(n-1)]) ;
batchResults = cell(1, numel(batches)) ;

% just use as many workers as are already available
numWorkers = matlabpool('size') ;
%parfor (b = 1:numel(batches), numWorkers)
for b = numel(batches):-1:1
  batchResults{b} = get_batch_results(imdb, imageIds, batches{b}, ...
                        encoder, opts.maxNumLocalDescriptorsReturned) ;
end

code = cell(size(imageIds)) ;
for b = 1:numel(batches)
  m = numel(batches{b});
  for j = 1:m
      k = batches{b}(j) ;
      code{k} = batchResults{b}.code{j};
  end
end
if opts.concatenateCode
   code = cat(2, code{:}) ;
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
       code_ = get_bcnn_features_nonsquare(encoder.neta, encoder.netb,...
         im, ...
        'regionBorder', encoder.regionBorder, ...
        'normalization', encoder.normalization);
    
result.code = code_ ;


function [w, bias] = logistic_reg(labels_, x_, imageSet, l_rate, maxIter, epsilon, scal)


w_decay = 0.0005;
momentum = 0.9;

trainIdx = find(ismember(imageSet, 1)) ;
validIdx = find(ismember(imageSet, 2)) ;
testIdx = find(ismember(imageSet, 3)) ;

train_N = numel(trainIdx);
valid_N = numel(validIdx);

batchSize = 128;
[d, N] = size(x_);
numClass = max(labels_);

%R = randperm(N);
%x_ = x_(:,R);
%labels_ = labels_(:,R);


w = 0.01/scal *randn(d,numClass,'single');
bias = zeros(numClass, 1, 'single');

w_momentum = zeros(size(w), 'single');
bias_momentum = zeros(size(bias), 'single');

iter = 1;
e = 1000;
ce_old = [];
while iter<=maxIter && e>epsilon
	%%
%     target_ = zeros(numClass, N, 'single');
%     ind = sub2ind([numClass, N], labels_, 1:N);
%     target_(ind) = 1;
    pred_ = zeros(1, N, 'single', 'gpuArray');
    logy_ = zeros(1, N, 'single', 'gpuArray');
    rn = randperm(numel(trainIdx));
    trainIdx = trainIdx(rn);
    aa = tic;
    for i=1:ceil(train_N/batchSize)
%         disp(num2str(i))
        batch = trainIdx((i-1)*batchSize+1:min(train_N,i*batchSize));
        x = gpuArray(x_(:, batch));
        labels = gpuArray(labels_(batch));
        
        target = zeros(numClass, numel(batch), 'single', 'gpuArray');
        ind = sub2ind([numClass, numel(batch)], labels, 1:numel(batch));
        target(ind) = 1;
        
        
        y = exp(w'*x + repmat(bias, 1, numel(batch)));
        sum_y = sum(y,1);
        y = bsxfun(@rdivide, y, sum_y);
        
        logy = log(y);
        logy(y<eps) = -100000;
        logy_(batch) = logy(target==1);
        
        [~,pred] = max(y, [], 1);
        
        pred_(batch) = pred;
        
        
        y_t = y-target;
        
        d_b = squeeze(mean(y_t, 2));
        
        
        d_w = arrayfun(@(z) mean(repmat(y_t(z,:), d, 1).*x, 2), 1:numClass, 'UniformOutput', false);
        d_w = cat(2, d_w{:});
        
        
        
        
        
        w_momentum = momentum*w_momentum - l_rate*w_decay*w - (l_rate.*d_w);
        bias_momentum = momentum*bias_momentum - l_rate*w_decay*bias - (l_rate.*d_b);
       
        w = w + w_momentum;
        bias = bias + bias_momentum;
        
        
%         w = w - l_rate*(d_w+0.01.*w);
%         bias = bias - l_rate*(d_b+0.01.*bias);

    end
    toc(aa)
    
    
    for i=1:ceil(valid_N/batchSize)
        batch = validIdx((i-1)*batchSize+1:min(valid_N,i*batchSize));
        x = gpuArray(x_(:, batch));
        labels = gpuArray(labels_(batch));
        
        target = zeros(numClass, numel(batch), 'single', 'gpuArray');
        ind = sub2ind([numClass, numel(batch)], labels, 1:numel(batch));
        target(ind) = 1;
        
        
        y = exp(w'*x + repmat(bias, 1, numel(batch)));
        sum_y = sum(y,1);
        y = bsxfun(@rdivide, y, sum_y);
        
        logy = log(y);
        logy(y<eps) = -100000;
        logy_(batch) = logy(target==1);
        
        [~,pred] = max(y, [], 1);
        
        pred_(batch) = pred;
        
    end
    
    pred_ = gather(pred_);
    logy_ = gather(logy_);
    accuracy = sum(pred_(trainIdx)==labels_(trainIdx))/train_N;
    
    
        ce = mean(-1*logy_(trainIdx));
        if isempty(ce_old)
            e = 1000;
        else
            e = abs(ce - ce_old);
        end
    
	ce_old = ce;
	
    valid_acc = sum(pred_(validIdx)==labels_(validIdx))/valid_N;
    
    
        valid_ce = mean(-1*logy_(validIdx));
        
        if valid_acc<eps
            dsljf = 1;
        end
    
	disp(['iteration: ', num2str(iter), ' training acc: ', num2str(accuracy), ' ce: ', ...
		num2str(ce), ' validation acc: ', num2str(valid_acc), ' ce: ', num2str(valid_ce)]);
	iter = iter + 1;
% 	pause;
end

w = gather(w);
bias = gather(bias);
