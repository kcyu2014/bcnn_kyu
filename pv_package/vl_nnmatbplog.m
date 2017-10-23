function y = vl_nnmatbplog(x, param, varargin)
% kyu: modified from Matrix backprop paper to make it work
thresh = param(1);

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

if backMode
    y = 0.5./sqrt(abs(x)+thresh);
    y = y.*dzdy;
else
    y = sign(x).*sqrt(abs(x));
end
