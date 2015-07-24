function y = vl_nnsqrt(x, varargin)
% VL_NNSQRT signed square-root feature

thres = 10^-8;
backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

if backMode
    y = 0.5./sqrt(abs(x)+thres);
    y = y.*dzdy;
else
    y = sign(x).*sqrt(abs(x));
end

