function tryLiblinearSVM
featPath = fullfile('data', 'bcnn-train_v1', 'cubcrop-seed-01-kepler');
addpath('/scratch1/tsungyulin/matlabToolbox/liblinear-1.96/matlab/')

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

X = sparse(double(code));

trainIdx = ismember(imdb.images.set, [1,2]);
testIdx = ismember(imdb.images.set, 3); 
labels = imdb.images.label';

model = train(labels(trainIdx), X(:,trainIdx), '-s 3 -B 1', 'col');
[predicted_label, accuracy, decision_values] = predict(labels(testIdx), X(:,testIdx), model, [], 'col');

save('liblinearSVMResult', 'model', 'predicted_label', 'accuracy', 'decision_values');

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
    
	disp(['iteration: ', num2str(iter), ' training acc: ', num2str(accuracy), ' ce: ', ...
		num2str(ce), ' validation acc: ', num2str(valid_acc), ' ce: ', num2str(valid_ce)]);
	iter = iter + 1;
% 	pause;
end

w = gather(w);
bias = gather(bias);




