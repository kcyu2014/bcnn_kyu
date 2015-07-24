function tryBinaryLiblinear(method, reweight)
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
                'useGpu', false);
        end
    end
end

% X = sparse(double(code));




addpath('/scratch1/tsungyulin/matlabToolbox/liblinear-1.5-dense-float/matlab/')

multiLabel = (size(imdb.images.label,1) > 1) ; % e.g. PASCAL VOC cls
    trainIdx = ismember(imdb.images.set, 1) ;
    test = ismember(imdb.images.set, 2) ;
    info.classes = find(imdb.meta.inUse) ;


    w = {} ;
    b = {} ;
    
    
    
    for c=1:numel(info.classes)
      if ~multiLabel
        y = 2*(imdb.images.label == info.classes(c)) - 1 ;
      else
        y = imdb.images.label(c,:) ;
      end
      
      
      np = sum(y(trainIdx) > 0) ;
      nn = sum(y(trainIdx) < 0) ;
      n = np + nn ;
      
      
      [scores{c}, w{c}, b{c}] = trainClassifier(y, code, trainIdx, test, np, nn, reweight, method);

	  pred = scores{c};

      [~,~,i]= vl_pr(y(trainIdx), pred(trainIdx)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(test), pred(test)) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(trainIdx), pred(trainIdx), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
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

  
    [~,preds] = max(info.scores,[],1) ;
    [~,gts] = ismember(imdb.images.label, info.classes) ;
    [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(trainIdx), preds(trainIdx)) ;
    [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;
	
	if(reweight)
		fname = ['ova_', method, '_rew'];
	else
		fname = ['ova_', method];
	end    
	save(fname, 'info');
  


function [scores, w, b] = trainClassifier(y, x, trainIdx, test, np, nn, reweight, method)

n = np + nn;

if(reweight)
    wp = n/(2*np);
    wn = n/(2*nn);
else
    wp = 1;
    wn = 1;
end

switch method
    case 'vlsvm'
        
      if(reweight)
          wv = wn.*ones(numel(find(trainIdx)), 1);
          wv(y(trainIdx & y == 1)) = wp;
          
          [w,b] = vl_svmtrain(x(:,trainIdx & y ~= 0), y(trainIdx & y ~= 0), 1/(n* 1), ...
              'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
              'maxNumIterations', n * 200, 'weights', wv) ;
      else
          [w,b] = vl_svmtrain(x(:,trainIdx & y ~= 0), y(trainIdx & y ~= 0), 1/(n* 1), ...
              'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
              'maxNumIterations', n * 200) ;
      end

      pred = w'*x + b ;

      % try cheap calibration
      mp = median(pred(trainIdx & y > 0)) ;
      mn = median(pred(trainIdx & y < 0)) ;
      b = (b - mn) / (mp - mn) ;
      w = w / (mp - mn) ;
      pred = w'*x + b ;

      scores = pred ;

    case 'libsvm'
    	y = (y + 3)/2;
    	if(reweight)
	    	model = train(y(trainIdx & y ~= 0)', x(:,trainIdx & y ~= 0), ['-s 3 -B 1 -w1 ', num2str(wn), ' -w2 ', num2str(wp)], 'col');
    	else
        	model = train(y(trainIdx & y ~= 0)', x(:,trainIdx & y ~= 0), '-s 3 -B 1', 'col');
        end
        
        w = transpose(model.w(1:end-1));
        b = model.w(end);
        
      	pred = w'*x + b ;

      % try cheap calibration
      	mp = median(pred(trainIdx & y > 0)) ;
      	mn = median(pred(trainIdx & y < 0)) ;
      	b = (b - mn) / (mp - mn) ;
      	w = w / (mp - mn) ;
      	pred = w'*x + b ;

      	scores = pred ;

    case 'vllr'
        
      if(reweight)
          wv = wn.*ones(numel(find(trainIdx)), 1);
          wv(y(trainIdx & y == 1)) = wp;
          
          [w,b] = vl_svmtrain(x(:,trainIdx & y ~= 0), y(trainIdx & y ~= 0), 1/(n* 1), ...
              'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
              'maxNumIterations', n * 200, 'loss', 'LOGISTIC', 'weights', wv) ;
      else
          [w,b] = vl_svmtrain(x(:,trainIdx & y ~= 0), y(trainIdx & y ~= 0), 1/(n* 1), ...
              'epsilon', 0.001, 'verbose', 'biasMultiplier', 1, ...
              'maxNumIterations', n * 200, 'loss', 'LOGISTIC') ;
      end

      pred = 1./(1+exp(-(w'*x + b))) ;


      scores = pred ;
      
      
    case 'liblr'
    	y = (y + 3)/2;
    	if(reweight)
	    	model = train(y(trainIdx & y ~= 0)', x(:,trainIdx & y ~= 0), ['-s 0 -B 1 -w1 ', num2str(wn), ' -w2 ', num2str(wp)], 'col');
    	else
        	model = train(y(trainIdx & y ~= 0)', x(:,trainIdx & y ~= 0), '-s 0 -B 1', 'col');
        end
        
        w = transpose(model.w(1:end-1));
        b = model.w(end);
        
        
      	pred = 1./(1+exp(-(w'*x + b))) ;


      	scores = pred ;
end








%{

for c=1:numel(info.classes)
      if ~multiLabel
        y = 2*(imdb.images.label == info.classes(c)) - 1 ;
      else
        y = imdb.images.label(c,:) ;
      end
      
      
      np = sum(y(trainIdx) > 0) ;
      nn = sum(y(trainIdx) < 0) ;
      n = np + nn ;
      
      
     pred = info.w(:,c)'*code+info.b(c);
scores{c}=pred;

      [~,~,i]= vl_pr(y(trainIdx), pred(trainIdx)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(test), pred(test)) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
      [~,~,i]= vl_pr(y(trainIdx), pred(trainIdx), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
      [~,~,i]= vl_pr(y(test), pred(test), 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
    end
    
    % Book keeping

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

  
    [~,preds] = max(info.scores,[],1) ;
    [~,gts] = ismember(imdb.images.label, info.classes) ;
    [info.train.confusion, info.train.acc] = compute_confusion(numel(info.classes), gts(trainIdx), preds(trainIdx)) ;
    [info.test.confusion, info.test.acc] = compute_confusion(numel(info.classes), gts(test), preds(test)) ;
    %}