function [net, stats] = bcnn_train_dag(net, imdb, getBatch_train, getBatch_val, varargin)

% BNN_TRAIN_DAG   training an asymmetric BCNN 
%    BCNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train an asymmetric BCNN for image classification.
%    The net should built in dagnn structure

% INPUT
% net: a network with a bcnn layer
% imdb: imdb structure of a dataset
% getBatch_train: function to read a batch of training images
% getBatch_val: function to read a batch of validation images

% OUTPUT
% net: bcnn network after fine-tuning
% stats: log of training and validation

% A asymmetric BCNN consists of two networks which are bilinearly combined.
% The bilinear layer is followed by square-root
% and L2 normalization and a sofmaxloss layer.

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% This function is modified from CNN_TRAIN_DAG of MatConvNet

opts.expDir = fullfile('data','exp') ;
opts.continue = false ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.derOutputs = {'objective', 1} ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.extractStatsFn = @extractStats ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

state.getBatch_train = getBatch_train ;
state.getBatch_val = getBatch_val ;
evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end
stats = [] ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('resuming by loading epoch %d\n', start) ;
  [net, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val ;
  state.imdb = imdb ;

  if numGpus <= 1
%     state.getBatch = getBatch_train;
    stats.train(epoch) = process_epoch(net, state, opts, 'train') ;
%     state.getBatch = getBatch_val;
    stats.val(epoch) = process_epoch(net, state, opts, 'val') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
%       state.getBatch = getBatch_train;
      stats_.train = process_epoch(net_, state, opts, 'train') ;
%       state.getBatch = getBatch_val;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_, ~isempty(opts.val)) ;
    stats.train(epoch) = stats__.train ;
    if ~isempty(opts.val)
        stats.val(epoch) = stats__.val ;
    end
    clear net_ stats_ stats__ savedNet_ ;
  end

  % save
  if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
  end

  
  
  figure(1) ; clf ;
  if ~isempty(opts.val)
      plots = setdiff(...
          cat(2,...
          fieldnames(stats.train)', ...
          fieldnames(stats.val)'), {'num', 'time'}) ;
  else
      plots = setdiff(...
          fieldnames(stats.train)', {'num', 'time'}) ;
  end
  for p = plots
    p = char(p) ;
    values = zeros(0, epoch) ;
    leg = {} ;
    if ~isempty(opts.val)
        fset = {'train', 'val'};
    else
        fset = {'train'};
    end
    for f = fset
      f = char(f) ;
      if isfield(stats.(f), p)
        tmp = [stats.(f).(p)] ;
        values(end+1,:) = tmp(1,:)' ;
        leg{end+1} = f ;
      end
    end
    subplot(1,numel(plots),find(strcmp(p,plots))) ;
    plot(1:epoch, values','o-') ;
    xlabel('epoch') ;
    title(p) ;
    legend(leg{:}) ;
    grid on ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
 
end

% -------------------------------------------------------------------------
function stats = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

if strcmp(mode,'train')
    getBatchFn = state.getBatch_train;
else
    getBatchFn = state.getBatch_val;
end

if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    sate.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

stats.time = 0 ;
stats.scores = [] ;
subset = state.(mode) ;
start = tic ;
num = 0 ;

for t=1:opts.batchSize:numel(subset)
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = getBatchFn(state.imdb, batch) ;
%     inputs = state.getBatch(state.imdb, batch) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
%       state.getBatch(state.imdb, nextBatch) ;
      getBatchFn(state.imdb, nextBatch) ;
    end

    if strcmp(mode, 'train')
      net.accumulateParamDers = (s ~= 1) ;
      net.eval(inputs, opts.derOutputs) ;
    else
      net.eval(inputs) ;
    end
  end

  % extract learning stats
  stats = opts.extractStatsFn(net) ;

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % print learning statistics
  time = toc(start) ;
  stats.num = num ;
  stats.time = toc(start) ;

  fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
    mode, ...
    state.epoch, ...
    fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
    stats.num/stats.time * max(numGpus, 1)) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:', f) ;
    fprintf(' %.3f', stats.(f)) ;
  end
  fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  thisDecay = opts.weightDecay * net.params(i).weightDecay ;
  thisLR = state.learningRate * net.params(i).learningRate ;

  if ~isempty(mmap)
    tmp = zeros(size(mmap.Data(labindex).(net.params(i).name)), 'single') ;
    for g = setdiff(1:numel(mmap.Data), labindex)
      tmp = tmp + mmap.Data(g).(net.params(i).name) ;
    end
    net.params(i).der = net.params(i).der + tmp ;
  end

  state.momentum{i} = opts.momentum * state.momentum{i} ...
    - thisDecay * net.params(i).value ...
    - (1 / batchSize) * net.params(i).der ;

  net.params(i).value = net.params(i).value + thisLR * state.momentum{i} ;
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_, isVal)
% -------------------------------------------------------------------------

stats = struct() ;
if isVal;
    fset = {'train', 'val'};
else
    fset = {'train'};
end
for s = fset
  s = char(s) ;
  total = 0 ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;

      if g == 1
        stats.(s).(f) = 0 ;
      end
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
