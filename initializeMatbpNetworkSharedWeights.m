function net = initializeMatbpNetworkSharedWeights(imdb, encoderOpts, opts)
% Modified by Kaicheng Yu
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% This code is used for initializing symmetric B-CNN network 

% -------------------------------------------------------------------------

scal = 1 ;
init_bias = 0.1;
numClass = length(imdb.classes.name);


assert(strcmp(encoderOpts.modela, encoderOpts.modelb), 'neta and netb are required to be the same');
assert(~isempty(encoderOpts.modela), 'network is not specified');

% Load the model
net = load(encoderOpts.modela);
net.meta.normalization.keepAspect = opts.keepAspect;

% truncate the network
maxLayer = max(encoderOpts.layera, encoderOpts.layerb);
net.layers = net.layers(1:maxLayer);

% get the feature dimension for both layers
netInfo = vl_simplenn_display(net);
mapSize1 = netInfo.dataSize(3, encoderOpts.layera+1);
mapSize2 = netInfo.dataSize(3, encoderOpts.layerb+1);

% network setting
net = vl_simplenn_tidy(net) ;
for l=numel(net.layers):-1:1
    if strcmp(net.layers{l}.type, 'conv')
        net.layers{l}.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    end
end

% add batch normalization
if opts.batchNormalization
    for l=numel(net.layers):-1:1
        if isfield(net.layers{l}, 'weights')
            ndim = size(net.layers{l}.weights{1}, 4);
            
            layer = struct('type', 'bnorm', 'name', sprintf('bn%s',l), ...
                'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), [zeros(ndim, 1, 'single'), ones(ndim, 1, 'single')]}}, ...
                'learningRate', [2 1 0.05], ...
                'weightDecay', [0 0]) ;
            
            
            net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
            
        end
    end
    net = simpleRemoveLayersOfType(net,'lrn');
end


% stack matbp layer
if(encoderOpts.layera==encoderOpts.layerb)
      net.layers{end+1} = struct('type','custom', ...
      'method', 'o2p_avg_log', ...
      'epsilon', 0.001, ...
      'forward', @nnfspool_forward, 'backward',@nnfspool_backward);

else
    error('MatBP Not supported bilinea no share weights');
end

% should not stack normalization
% net.layers{end+1} = struct('type', 'sqrt', 'name', 'sqrt_norm');
% net.layers{end+1} = struct('type', 'l2norm', 'name', 'l2_norm');


% build a linear classifier netc
initialW = 0.001/scal * randn(1,1,mapSize1*mapSize2,numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');
netc.layers = {};
netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
    'weights', {{initialW, initialBias}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1000 1000], ...
    'weightDecay', [0 0]) ;
netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
netc = vl_simplenn_tidy(netc) ;


% pretrain the linear classifier with logistic regression
if(opts.bcnnLRinit && ~opts.fromScratch)
    
    % get bcnn feature for train and val sets
    train = find(imdb.images.set==1|imdb.images.set==2);
    if ~exist(opts.nonftbcnnDir, 'dir')
        netInit = net;
        
        if ~isempty(opts.train.gpus)
            netInit = vl_simplenn_move(netInit, 'gpu') ;
        end
        
        batchSize = 64;
        
        bopts = netInit.meta.normalization ;
        bopts.numThreads = opts.numFetchThreads ;
        bopts.transformation = 'none' ;
        bopts.rgbVariance = [] ;
        bopts.scale = opts.imgScale;
        
        
        getBatchFn = getBatchSimpleNNWrapper(bopts);
        
        mkdir(opts.nonftbcnnDir)
        
        % compute and cache the bilinear cnn features
        for t=1:batchSize:numel(train)
            fprintf('Initialization: extracting cnn feature of batch %d/%d\n', ceil(t/batchSize), ceil(numel(train)/batchSize));
            batch = train(t:min(numel(train), t+batchSize-1));
            [im, labels] = getBatchFn(imdb, batch) ;
            if opts.train.prefetch
                nextBatch = train(t+batchSize:min(t+2*batchSize-1, numel(train))) ;
                getBatcFn(imdb, nextBatch) ;
            end
            im = im{1};
            if ~isempty(opts.train.gpus)
                im = gpuArray(im) ;
            end
            
            net.layers{end}.class = labels ;
            
            res = [] ;
            res = vl_bilinearnn(netInit, im, [], res, ...
                'accumulate', false, ...
                'mode', 'test', ...
                'conserveMemory', true, ...
                'sync', true, ...
                'cudnn', opts.cudnn) ;
            codeb = squeeze(gather(res(end).x));
            for i=1:numel(batch)
                code = codeb(:,i);
                savefast(fullfile(opts.nonftbcnnDir, ['bcnn_nonft_', num2str(batch(i), '%05d')]), 'code');
            end
        end
    end
    
    clear code res netInit
    
    % get the pretrain linear classifier
    if exist(fullfile(opts.expDir, 'initial_fc.mat'), 'file')
        load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
    else
        
        bcnndb = imdb;
        tempStr = sprintf('%05d\t', train);
        tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
        bcnndb.images.name = strcat('bcnn_nonft_', tempStr{1}');
        bcnndb.images.id = bcnndb.images.id(train);
        bcnndb.images.label = bcnndb.images.label(train);
        bcnndb.images.set = bcnndb.images.set(train);
        bcnndb.imageDir = opts.nonftbcnnDir;
        
        %train logistic regression
        [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn_fromdisk, opts.inittrain, ...
            'conserveMemory', true);
        
        save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc', '-v7.3') ;
    end
    
end

% set all parameters to random number if train the model from scratch
if(opts.fromScratch)
    for i=1:numel(net.layers)
        if ~strcmp(net.layers{i}.type, 'conv'), continue ; end
        net.layers{i}.learningRate = [1 2];
        net.layers{i}.weightDecay = [1 0];
        net.layers{i}.weights = {0.01/scal * randn(size(net.layers{i}.weights{1}), 'single'), init_bias*ones(size(net.layers{i}.weights{2}), 'single')};
    end
end

% stack netc on network
for i=1:numel(netc.layers)
    net.layers{end+1} = netc.layers{i};
end
clear netc

% Rename classes
net.meta.classes.name = imdb.classes.name;
net.meta.classes.description = imdb.classes.name;

% add border for translation data jittering
if(~strcmp(opts.dataAugmentation{1}, 'f2') && ~strcmp(opts.dataAugmentation{1}, 'none'))
    net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
end

 

function [im,labels] = getBatch_bcnn_fromdisk(imdb, batch)
% -------------------------------------------------------------------------

im = cell(1, numel(batch));
for i=1:numel(batch)
    load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
    im{i} = code;
end
im = cat(2, im{:});
im = reshape(im, 1, 1, size(im,1), size(im, 2));
labels = imdb.images.label(batch) ;



function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;

% -------------------------------------------------------------------------
function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = simpleFindLayersOfType(net, type) ;
net.layers(layers) = [] ;