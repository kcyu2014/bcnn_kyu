function nparam = computeNetNumParams(net, varargin)

setup;

if(numel(varargin)==1)
    nlayer = varargin{1};
else
    nlayer = numel(net.layers);
end

info = vl_simplenn_display(net);

nparam = 0;
for i=1:nlayer
    if ~strcmp(net.layers{i}.type, 'conv'), continue ; end
    [h,w,inputDim,outputDim] = size(net.layers{i}.filters);
    nparam = nparam + info.support(1,i)*info.support(2,i)*inputDim*outputDim;
end