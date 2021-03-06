function net = net_deploy(net)
%CNN_IMAGENET_DEPLOY  Deploy a CNN

isDag = isa(net, 'dagnn.DagNN') ;
if isDag
  dagRemoveLayersOfType(net, 'dagnn.Loss') ;
  dagRemoveLayersOfType(net, 'dagnn.DropOut') ;
else
  net = simpleRemoveLayersOfType(net, 'softmaxloss') ;
  net = simpleRemoveLayersOfType(net, 'dropout') ;
end

if isDag
  net.addLayer('prob', dagnn.SoftMax(), 'prediction', 'prob', {}) ;
else
  net.layers{end+1} = struct('name', 'prob', 'type', 'softmax') ;
end

if isDag
  dagMergeBatchNorm(net) ;
  dagRemoveLayersOfType(net, 'dagnn.BatchNorm') ;
else
  net = simpleMergeBatchNorm(net) ;
  net = simpleRemoveLayersOfType(net, 'bnorm') ;
end

if ~isDag
  net = simpleRemoveMomentum(net) ;
end

% Switch to use MatConvNet default memory limit for CuDNN (512 MB)
if ~isDag
  for l = simpleFindLayersOfType(net, 'conv')
    net.layers{l}.opts = removeCuDNNMemoryLimit(net.layers{l}.opts) ;
  end
else
  for name = dagFindLayersOfType(net, 'dagnn.Conv')
    l = net.getLayerIndex(char(name)) ;
    net.layers(l).block.opts = removeCuDNNMemoryLimit(net.layers(l).block.opts) ;
  end
end

% -------------------------------------------------------------------------
function opts = removeCuDNNMemoryLimit(opts)
% -------------------------------------------------------------------------
remove = false(1, numel(opts)) ;
for i = 1:numel(opts)
  if isstr(opts{i}) && strcmp(lower(opts{i}), 'CudnnWorkspaceLimit')
    remove([i i+1]) = true ;
  end
end
opts = opts(~remove) ;

% -------------------------------------------------------------------------
function net = simpleRemoveMomentum(net)
% -------------------------------------------------------------------------
for l = 1:numel(net.layers)
  if isfield(net.layers{l}, 'momentum')
    net.layers{l} = rmfield(net.layers{l}, 'momentum') ;
  end
end

% -------------------------------------------------------------------------
function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;

% -------------------------------------------------------------------------
function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = simpleFindLayersOfType(net, type) ;
net.layers(layers) = [] ;

% -------------------------------------------------------------------------
function layers = dagFindLayersWithOutput(net, outVarName)
% -------------------------------------------------------------------------
layers = {} ;
for l = 1:numel(net.layers)
  if any(strcmp(net.layers(l).outputs, outVarName))
    layers{1,end+1} = net.layers(l).name ;
  end
end

% -------------------------------------------------------------------------
function layers = dagFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = [] ;
for l = 1:numel(net.layers)
  if isa(net.layers(l).block, type)
    layers{1,end+1} = net.layers(l).name ;
  end
end

% -------------------------------------------------------------------------
function dagRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, type) ;
for i = 1:numel(names)
  layer = net.layers(net.getLayerIndex(names{i})) ;
  net.removeLayer(names{i}) ;
  net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end

% -------------------------------------------------------------------------
function dagMergeBatchNorm(net)
% -------------------------------------------------------------------------
names = dagFindLayersOfType(net, 'dagnn.BatchNorm') ;
for name = names
  name = char(name) ;
  layer = net.layers(net.getLayerIndex(name)) ;

  % merge into previous conv layer
  player = dagFindLayersWithOutput(net, layer.inputs{1}) ;
  player = net.layers(net.getLayerIndex(player)) ;
  if ~isa(player.block, 'dagnn.Conv')
    error('Batch normalization cannot be merged as it is not preceded by a conv layer.') ;
  end

  filters = net.getParamIndex(player.params{1}) ;
  biases = net.getParamIndex(player.params{2}) ;
  multipliers = net.getParamIndex(layer.params{1}) ;
  offsets = net.getParamIndex(layer.params{2}) ;
  moments = net.getParamIndex(layer.params{3}) ;

  [filtersValue, biasesValue] = mergeBatchNorm(...
    net.params(filters).value, ...
    net.params(biases).value, ...
    net.params(multipliers).value, ...
    net.params(offsets).value, ...
    net.params(moments).value) ;

  net.params(filters).value = filtersValue ;
  net.params(biases).value = biasesValue ;
end

% -------------------------------------------------------------------------
function net = simpleMergeBatchNorm(net)
% -------------------------------------------------------------------------

for l = 1:numel(net.layers)
  if strcmp(net.layers{l}.type, 'bnorm')
    if ~strcmp(net.layers{l-1}.type, 'conv')
      if ~strcmp(net.layers{l+1}.type, 'conv')
        error('Batch normalization cannot be merged as it is not preceded or connected by a conv layer.') ;
      end
      % For l+1 type.
      [filters, biases] = mergeBatchNormAfterConv(...
      net.layers{l+1}.weights{1}, ...
      net.layers{l+1}.weights{2}, ...
      net.layers{l}.weights{1}, ...
      net.layers{l}.weights{2}, ...
      net.layers{l}.weights{3}) ;
      net.layers{l+1}.weights = {filters, biases} ;
      break;
    end
    [filters, biases] = mergeBatchNorm(...
      net.layers{l-1}.weights{1}, ...
      net.layers{l-1}.weights{2}, ...
      net.layers{l}.weights{1}, ...
      net.layers{l}.weights{2}, ...
      net.layers{l}.weights{3}) ;
    net.layers{l-1}.weights = {filters, biases} ;
  end
end

% -------------------------------------------------------------------------
function [filters, biases] = mergeBatchNorm(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
% wk / sqrt(sigmak^2 + eps)
% bk - wk muk / sqrt(sigmak^2 + eps)
a = multipliers(:) ./ moments(:,2) ;
b = offsets(:) - moments(:,1) .* a ;
biases(:) = biases(:) + b(:) ;
sz = size(filters) ;
numFilters = sz(4) ;
filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;
