function y = vl_nncovariance(x, varargin)
% VL_NNCOVARIANCE computes self outer product with mapping of a feature output and pool the features across 
% all locations
%
% Author: Kaicheng Yu

%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
end

gpuMode = isa(x, 'gpuArray');
[h, w, ch, bs] = size(x);

indmap = (eye(ch) - ones(ch, ch)/(h*w))/(w*h);

if backMode
    if gpuMode
        y = gpuArray(zeros(size(x), 'single'));
    else
        y = zeros(size(x), 'single');
    end
    for b=1:bs
        dzdy_b = reshape(dzdy(1,1,:,b), [ch, ch]);
        dzdy_b = dzdy_b + dzdy_b';
        a = reshape(x(:,:,:,b), [h*w, ch]);
        y(:, :, :, b) = reshape(a * indmap * dzdy_b, [h, w, ch]);
    end
else
    if gpuMode
        y = gpuArray(zeros([ch, ch, bs], 'single'));
    else
        y = zeros([ch, ch, bs], 'single');
    end
    for b = 1:bs
        a = reshape(x(:,:,:,b), [h*w, ch]);
        y(:, :, b) = reshape(a'* ind_map * a, [ch, ch]);
    end
end

