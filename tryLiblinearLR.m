function tryLiblinearLR
featPath = fullfile('data', 'bcnn-train_v1', 'cubcrop-seed-01-kepler');
addpath('/scratch1/tsungyulin/matlabToolbox/liblinear-1.5-dense-float/matlab/')

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

model = train(labels(trainIdx), X(:,trainIdx), '-s 7 -B 1', 'col');
[predicted_label, accuracy, decision_values] = predict(labels(testIdx), X(:,testIdx), model, [], 'col');

save('liblinearResult_dual', 'model', 'predicted_label', 'accuracy', 'decision_values');

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


function info = traintest(opts, imdb, psi)
% -------------------------------------------------------------------------

% Train using verification or not
verificationTask = isfield(imdb, 'pairs');

if verificationTask, 
    train = ismember(imdb.pairs.set, [1 2]) ;
    test = ismember(imdb.pairs.set, 3) ;
else % classification task
    multiLabel = (size(imdb.images.label,1) > 1) ; % e.g. PASCAL VOC cls
    train = ismember(imdb.images.set, [1 2]) ;
    test = ismember(imdb.images.set, 3) ;
    info.classes = find(imdb.meta.inUse) ;
    
    w = {} ;
    b = {} ;
    
    for c=1:numel(info.classes)
      if ~multiLabel
        y = 2*(imdb.images.label == info.classes(c)) - 1 ;
      else
        y = imdb.images.label(c,:) ;
      end
    switch opts.normalization
        case 'sqrt'
            C = 1 ;
        case 'none'
            L2norm = sqrt(sum(psi(:,train & y ~=0).*psi(:,train & y ~=0),1));
            C = 1/mean(L2norm);
            
            disp(['svm C: ', num2str(C)])
            
        otherwise
            C = 1;
    end
      np = sum(y(train) > 0) ;
      nn = sum(y(train) < 0) ;
      n = np + nn ;

      [w{c},b{c}] = vl_svmtrain(psi(:,train & y ~= 0), y(train & y ~= 0), 1/(n* 1), ...
        'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
        'maxNumIterations', n * 200) ;

      pred = w{c}'*psi + b{c} ;

      % try cheap calibration
      mp = median(pred(train & y > 0)) ;
      mn = median(pred(train & y < 0)) ;
      b{c} = (b{c} - mn) / (mp - mn) ;
      w{c} = w{c} / (mp - mn) ;
      pred = w{c}'*psi + b{c} ;

      scores{c} = pred ;

      [~,~,i]= vl_pr(y(train), pred(train)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(test), pred(test)) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(train), pred(train), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
      [~,~,i]= vl_pr(y(test), pred(test), 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
    end
    
    % Book keeping
    info.w = cat(2,w{:}) ;
    info.b = cat(2,b{:}) ;
    info.scores = cat(1, scores{:}) ;
    info.train.ap = ap ;
    info.train.ap11 = ap11 ;
    info.train.nap = nap ;
    info.train.map = mean(ap) ;
    info.train.map11 = mean(ap11) ;
    info.train.mnap = mean(nap) ;
    info.test.ap = tap ;
    info.test.ap11 = tap11 ;
    info.test.nap = tnap ;
    info.test.map = mean(tap) ;
    info.test.map11 = mean(tap11) ;
    info.test.mnap = mean(tnap) ;
    clear ap nap tap tnap scores ;
    fprintf('mAP train: %.1f, test: %.1f\n', ...
      mean(info.train.ap)*100, ...
      mean(info.test.ap)*100);

    % Compute predictions, confusion and accuracy
    [~,preds] = max(info.scores,[],1) ;
    [~,gts] = ismember(imdb.images.label, info.classes) ;
    [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(train), preds(train)) ;
    [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;
end


