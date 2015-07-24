% function info = tryMultiClassLogistic(method, reweight)
function tryMultiClassLogistic
featPath = fullfile('data', 'bcnn-train_v1', 'cubcrop-seed-01-kepler');

load(fullfile(featPath, 'bcnnmm-codes.mat'));



maxNumCompThreads(10);
  bcnnmm.name = 'bcnnmm' ;
  bcnnmm.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/imagenet-vgg-m.mat', ...
    'layera', 14,...
    'modelb', 'data/models/imagenet-vgg-m.mat', ...
    'layerb', 14,...
    } ;


setupNameList = {'bcnnmm'};
encoderList = {{bcnnmm}};
datasetList = {{'cubcrop', 1}};

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
            
            [~, imdb] = model_setup('dataset', dataset, ...
                'encoders', encoderList{ee}, ...
                'prefix', 'bcnn-train_v2', ...
                'batchSize', 128, ...
                'useGpu', true);
        end
    end
end

% X = sparse(double(code));


imdb.meta.inUse(21:end) = false;

addpath('/scratch1/tsungyulin/matlabToolbox/liblinear-1.5-dense-float/matlab/')

    info.classes = find(imdb.meta.inUse) ;
    info.dataUsed = ismember(imdb.images.label, info.classes);
    imdb.images.set = imdb.images.set(info.dataUsed);
    imdb.images.label = imdb.images.label(info.dataUsed);
multiLabel = (size(imdb.images.label,1) > 1) ; % e.g. PASCAL VOC cls
    
    code = code(:,info.dataUsed);

    
    
    
    
opts.train.batchSize = 256 ;
opts.train.numEpochs = 100 ;
opts.train.continue = false ;
opts.train.useGpu = true ;
opts.train.prefetch = false ;
% opts.train.learningRate = 0.1.*[0.001*ones(1, 30) 0.0001*ones(1, 30) 0.00001*ones(1,40)] ;
opts.train.learningRate = [0.1*ones(1, 10), 0.01*ones(1, 20), 0.001*ones(1, 50)];


scal = 1 ;
init_bias = 0.1;
numClass = numel(find(imdb.meta.inUse));



%% CNN i

net.layers = {};
bi_d = size(code, 1);


initialW = 0.001/scal *randn(bi_d, numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');


net.layers{end+1} = struct('type', 'conv', ...
                           'filters', reshape(initialW, [1, 1, size(initialW, 1), size(initialW, 2)]), ...
                           'biases', initialBias, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'filtersWeightDecay', 1, ...
                           'biasesWeightDecay', 0) ;
                       
net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', 0.5) ;

net.layers{end+1} = struct('type', 'softmaxloss') ;
bcnndb.sets = imdb.sets;
bcnndb.classes = imdb.classes;
bcnndb.images = imdb.images;
bcnndb.meta = imdb.meta;
bcnndb.codes = code;

% bcnndb.images.set(ismember(bcnndb.images.set, [1,2])) = 1;
% bcnndb.images.set(ismember(bcnndb.images.set, 3)) = 2;

% [net, info] = cnn_train(net, bcnndb, @getBatch_bcnn, opts.train, ...
%     'batchSize', opts.train.batchSize, 'conserveMemory', true, ...
%     'momentum', 0.9, 'weightDecay', 0.5, 'expDir', fullfile(featPath, 'multi-LogReg-subset-small-weight'));

l_rate = [0.001.*ones(1, 100), 0.001.*ones(1,150)];
[initialW, initialBias] = logistic_reg(bcnndb.images.label, code, imdb.images.set, l_rate, 300, 10^-10, 10);

% 
% 	if(reweight)
% 		fname = fullfile('ovaSubsetResults', ['ova_', method, '_rew']);
% 	else
% 		fname = fullfile('ovaSubsetResults', ['ova_', method]);
% 	end    
% 	save(fname, 'info');
  
function [im,labels] = getBatch_bcnn(imdb, batch)
% -------------------------------------------------------------------------
% images = strcat([imdb.imageDir '/'], imdb.images.name(batch)) ;
% im = imdb_get_batch(images, opts, ...
%                             'numThreads', numThreads, ...
%                             'prefetch', nargout == 0);
im = reshape(imdb.codes(:,batch), 1,1,size(imdb.codes,1), numel(batch));
labels = imdb.images.label(batch) ;





function [w, bias] = logistic_reg(labels_, x_, imageSet, l_rate_array, maxIter, epsilon, scal)


w_decay = 0.1;
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
bias = ones(numClass, 1, 'single');

w_momentum = zeros(size(w), 'single');
bias_momentum = zeros(size(bias), 'single');

iter = 1;
e = 1000;
ce_old = [];
while iter<=maxIter && e>epsilon
    l_rate = l_rate_array(min(iter, numel(l_rate_array)));
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
        
        d_b = squeeze(sum(y_t, 2));
        
        
        d_w = arrayfun(@(z) sum(repmat(y_t(z,:), d, 1).*x, 2), 1:numClass, 'UniformOutput', false);
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
        
        t_ce(iter) = ce;
        t_acc(iter) = accuracy;
        
        if isempty(ce_old)
            e = 1000;
        else
            e = abs(ce - ce_old);
        end
    
	ce_old = ce;
	
    valid_acc = sum(pred_(validIdx)==labels_(validIdx))/valid_N;
    
    
        valid_ce = mean(-1*logy_(validIdx));
    
        v_ce(iter) = valid_ce;
        v_acc(iter) = valid_acc;
        
	disp(['iteration: ', num2str(iter), ' training acc: ', num2str(accuracy), ' ce: ', ...
		num2str(ce), ' validation acc: ', num2str(valid_acc), ' ce: ', num2str(valid_ce)]);

    
%     save(modelPath(epoch), 'net', 'info') ;
%     
    figure(1) ; clf ;
    subplot(1,2,1) ;
    semilogy(1:iter, t_ce, 'k') ; hold on ;
    semilogy(1:iter, v_ce, 'b') ;
    xlabel('training iteration') ; ylabel('energy') ;
    grid on ;
    h=legend('train', 'val') ;
    set(h,'color','none');
    title('Cross Entropy') ;
    subplot(1,2,2) ;
    
            plot(1:iter, 1-t_acc, 'k') ; hold on ;
            plot(1:iter, 1-v_acc, 'b') ;
            h=legend('train','val') ;
            
    grid on ;
    xlabel('training iteration') ; ylabel('error') ;
    set(h,'color','none') ;
    title('error') ;
    drawnow ;
    print(1, fullfile('data', 'bcnn-train_v1', 'cubcrop-seed-01-kepler', 'log-train'), '-dpdf') ;
    
    
    
	iter = iter + 1;
end

w = gather(w);
bias = gather(bias);


