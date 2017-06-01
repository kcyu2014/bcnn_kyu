function run_experiments_train()


% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% fine tuning standard cnn model
[opts, imdb] = model_setup('dataset', 'dtd', ...
			  'encoders', {}, ...
			  'prefix', 'ft-cnn', ...
			  'model', 'imagenet-vgg-m.mat',...
			  'batchSize', 128, ...
			  'gpus', []);
imdb_cnn_train(imdb, opts);
