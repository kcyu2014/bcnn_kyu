function imdb_bcnn_train(imdb, opts, varargin)
% Train a bilinear CNN model on a dataset supplied by imdb

opts.lite = false ;
opts.numFetchThreads = 0 ;
opts.train.batchSize = 256 ;
opts.train.numEpochs = 45 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.prefetch = false ;
%opts.train.learningRate = [0.001*ones(1, 10) 0.0001*ones(1, 10) 0.00001*ones(1,10)] ;
opts.train.learningRate = [0.001*ones(1, 10) 0.001*ones(1, 10) 0.001*ones(1,10)] ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;


if(opts.useGpu)
    opts.train.useGpu = opts.useGpu;
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


bcnn_net = initializeNetwork(imdb, encoder, opts) ;
fn = getBatchWrapper(bcnn_net.neta.normalization, opts.numFetchThreads) ;
% [bcnn_net,info] = bcnn_train(bcnn_net, imdb, fna, fnb, opts.inittrain, 'batchSize', opts.batchSize, 'conserveMemory', true) ;

%{
im = cell(numel(imdb.images.label),1);
for i=1:numel(im)
    fprintf('reading image %d.\n', i);
    im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{i}));
    if size(im{i}, 3) == 1, im{i} = repmat(im{i}, [1 1 3]); end; %grayscale image
end
%}

[bcnn_net,info] = bcnn_train(bcnn_net, encoder, fn, imdb, opts.train, 'batchSize', opts.batchSize, 'conserveMemory', true) ;


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
im = imdb_get_batch_bcnn(images, opts, ...
                            'numThreads', numThreads, ...
                            'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;



function net = initializeNetwork(imdb, encoder, opts)
% -------------------------------------------------------------------------

% set the two networks
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


netc.layers{end+1} = struct('type', 'sqrt');
netc.layers{end+1} = struct('type', 'l2norm');
    
initialW = 0.001/scal *randn(1,1,dim, numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');



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
net.netc = netc;


