function imdb_bcnn_train_pca(imdb, opts, varargin)
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


% opts.lite = false ;
% opts.numFetchThreads = 0 ;
opts.inittrain.weightDecay = 0 ;
opts.inittrain.batchSize = 256 ;
opts.inittrain.numEpochs = 100 ;
opts.inittrain.continue = true ;
opts.inittrain.useGpu = false ;
opts.inittrain.prefetch = false ;
opts.inittrain.learningRate = [0.001*ones(1, 300) 0.001*ones(1, 10)] ;
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


[bcnn_net, encoder] = initializeNetwork(imdb, encoder, opts) ;
% fna = getBatchWrapper(bcnn_net.neta.normalization, opts.numFetchThreads) ;
% fnb = getBatchWrapper(bcnn_net.netb.normalization, opts.numFetchThreads) ;
% [bcnn_net,info] = bcnn_train(bcnn_net, imdb, fna, fnb, opts.inittrain, 'batchSize', opts.batchSize, 'conserveMemory', true) ;


if(~exist(fullfile(opts.expDir, 'pca-initialize-model'), 'dir'))
    mkdir(fullfile(opts.expDir, 'pca-initialize-model'))
end
[~, namea, ~] = fileparts(encoderOpts.modela);
[~, nameb, ~] = fileparts(encoderOpts.modelb);
saveNetwork(fullfile(opts.expDir, 'pca-initialize-model', ['fine-tuned-neta-', namea, '.mat']), bcnn_net.neta);
saveNetwork(fullfile(opts.expDir, 'pca-initialize-model', ['fine-tuned-netb-', nameb, '.mat']), bcnn_net.netb);



im = cell(numel(imdb.images.label),1);
for i=1:numel(im)
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



function [net, varargout] = initializeNetwork(imdb, encoder, opts)
% -------------------------------------------------------------------------

net.neta = encoder.neta;
net.netb = encoder.netb;


dA = 256;
dB = 64;

resA = [];
resB = [];

imageIds = imdb.images.id;
[~,imageSel] = ismember(imageIds, imdb.images.id) ;
imageIds = unique(imdb.images.id(imageSel)) ;
n = numel(imageIds) ;
% n=200;
A = cell(n,1);
B = cell(n,1);


if exist(fullfile(opts.expDir, 'initial_pca.mat'))
    load(fullfile(opts.expDir, 'initial_pca.mat')) ;
else
    
    for i=1:n
        disp(['computing image ', num2str(i)])
        opts2.crop = true ;
        %opts.scales = 2.^(1.5:-.5:-3); % try a bunch of scales
        opts2.scales = 2;
        opts2.encoder = [] ;
        opts2.regionBorder = 0.05;
        opts2.normalization = 'false';
        
        
        im{i} = imread(fullfile(imdb.imageDir, imdb.images.name{i}));
        
        
        
        % get parameters of the network
        info = vl_simplenn_display(net.neta) ;
        borderA = round(info.receptiveField(end)/2+1) ;
        averageColourA = mean(mean(net.neta.normalization.averageImage,1),2) ;
        imageSizeA = net.neta.normalization.imageSize;
        
        info = vl_simplenn_display(net.netb) ;
        borderB = round(info.receptiveField(end)/2+1) ;
        averageColourB = mean(mean(net.netb.normalization.averageImage,1),2) ;
        imageSizeB = net.netb.normalization.imageSize;
        
        assert(all(imageSizeA == imageSizeB));
        
        
        
        im_cropped = imresize(single(im{i}), imageSizeA([2 1]), 'bilinear');
        crop_h = size(im_cropped,1) ;
        crop_w = size(im_cropped,2) ;
        % for each scale
        
        if min(crop_h,crop_w) * opts2.scales < min(borderA, borderB), continue ; end
        if sqrt(crop_h*crop_w) * opts2.scales > 1024, continue ; end
        
        % resize the cropped image and extract features everywhere
        im_resized = imresize(im_cropped, opts2.scales) ;
        im_resizedA = bsxfun(@minus, im_resized, averageColourA) ;
        im_resizedB = bsxfun(@minus, im_resized, averageColourB) ;
        if net.neta.useGpu
            im_resizedA = gpuArray(im_resizedA) ;
            im_resizedB = gpuArray(im_resizedB) ;
        end
        resA = vl_simplenn(net.neta, im_resizedA, [], resA, ...
            'conserveMemory', true, 'sync', true);
        resB = vl_simplenn(net.netb, im_resizedB, [], resB, ...
            'conserveMemory', true, 'sync', true);
        A{i} = gather(resA(end).x);
        B{i} = gather(resB(end).x);
        A{i} = reshape(A{i}, [size(A{i},1)*size(A{i},2), size(A{i},3)]);
        B{i} = reshape(B{i}, [size(B{i},1)*size(B{i},2), size(B{i},3)]);
        
        A{i} = A{i}(randperm(size(A{i},1), ceil(0.05*size(A{i},1))),:);
        B{i} = B{i}(randperm(size(B{i},1), ceil(0.05*size(B{i},1))),:);
        
        
    end
    
    
    % A = cat(1, A{:});
    % B = cat(1, B{:});
    A = cell2mat(A);
    B = cell2mat(B);
    mA = mean(A,1);
    mB = mean(B,1);
    A = bsxfun(@minus, A, mA);
    B = bsxfun(@minus, B, mB);
    A = A'*A;
    [Wa, Sa] = eig(A);
    B = B'*B;
    [Wb, Sb] = eig(B);
    Sa = diag(Sa);
    Sb = diag(Sb);
    
    save(fullfile(opts.expDir, 'initial_pca.mat'), 'Wa', 'Wb', 'Sa', 'Sb', 'mA', 'mB')
    
end

% [~, idx] = sort(Sa, 'descend');
% Wa = Wa(:,idx(1:dA));
[~, idx] = sort(Sb, 'descend');
Wb = Wb(:,idx(1:dB));


% biasA = -mA*Wa;
% biasB = -mB*Wb;

% biasA = -mA*Wa + 0.*sqrt(Sa(idx(1:dA))');
biasB = -mB*Wb + 0.*sqrt(Sb(idx(1:dB))');

%{


Sa = diag(Sa);
checkSa = cumsum(Sa(idx));
checkSa = checkSa./checkSa(end);
Sb = diag(Sb);
checkSb = cumsum(Sb(idx));
checkSb = checkSb./checkSb(end);

h1 = figure(10)
plot(1:numel(checkSa), checkSa, 'r-')
saveas(h1, 'checkSVDa.fig', 'fig')

h2 = figure(11)
plot(1:numel(checkSb), checkSb, 'r-')
saveas(h2, 'checkSVDb.fig', 'fig')

pause
close([10,11])
%}

%{
net.neta.layers{end+1} = struct('type', 'conv', ...
    'filters', reshape(Wa, 1, 1, size(Wa, 1), size(Wa, 2)), ...
    'biases', biasA, ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 1, ...
    'biasesLearningRate', 1, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 1) ;

net.neta.layers{end+1} = struct('type', 'relu') ;

if(net.neta.useGpu)
    net.neta.layers{end-1}.filters = gpuArray(net.neta.layers{end-1}.filters);
    net.neta.layers{end-1}.biases = gpuArray(net.neta.layers{end-1}.biases);
end

%}

net.netb.layers{end+1} = struct('type', 'conv', ...
    'filters', reshape(Wb, 1, 1, size(Wb, 1), size(Wb, 2)), ...
    'biases', biasB, ...
    'stride', 1, ...
    'pad', 0, ...
    'filtersLearningRate', 1, ...
    'biasesLearningRate', 1, ...
    'filtersWeightDecay', 1, ...
    'biasesWeightDecay', 1) ;



if(net.netb.useGpu)
    net.netb.layers{end}.filters = gpuArray(net.netb.layers{end}.filters);
    net.netb.layers{end}.biases = gpuArray(net.netb.layers{end}.biases);
end


% net.netb.layers{end+1} = struct('type', 'relu') ;
% 
% if(net.netb.useGpu)
%     net.netb.layers{end-1}.filters = gpuArray(net.netb.layers{end-1}.filters);
%     net.netb.layers{end-1}.biases = gpuArray(net.netb.layers{end-1}.biases);
% end

encoder.neta = net.neta;
encoder.netb = net.netb;

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

code = sign(code).*sqrt(abs(code));
c_norm = arrayfun(@(x) norm(code(:,x)), 1:size(code,2));
code = bsxfun(@rdivide, code, c_norm);


% Else initial model randomly
netc.layers = {};
bi_d = size(code, 1);


imdbLabel = imdb.images.label;

trainIdx = ismember(imdb.images.set, [1 2]) ;
testIdx = ismember(imdb.images.set, 3) ;



initialW = 0.01/scal *randn(1,1,bi_d, numClass,'single');
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
    bcnndb.sets = imdb.sets;
    bcnndb.classes = imdb.classes;
    bcnndb.images = imdb.images;
    bcnndb.meta = imdb.meta;
    bcnndb.codes = code;
    
    
    [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn, opts.inittrain, ...
        'batchSize', opts.inittrain.batchSize, 'weightDecay', opts.inittrain.weightDecay, ...
        'conserveMemory', true, 'expDir', opts.inittrain.expDir);

    save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
end

net.netc = netc;

if nargout==2
    varargout{1} = encoder;
end




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
       code_ = get_bcnn_features(encoder.neta, encoder.netb,...
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
