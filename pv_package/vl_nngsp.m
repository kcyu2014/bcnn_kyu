function y = vl_nngsp(x, varargin)
% VL_NNGSP perform global square pooling for the input features
% at each location
%
% Author: Kaicheng

%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
end

[h, w, ch, bs] = size(x);

if backMode
    y = 2 .* x ./ (h*w);
    y = y .* repmat(dzdy, h, w);
else
    a = x.^ 2;
    a = reshape(a, [h*w, ch, bs]);
    y = reshape(sum(a, 1), [1, 1, ch, bs]) / (h*w);
  
end
