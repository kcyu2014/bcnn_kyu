function y = vl_nnbilinear(x, varargin)
% VL_NNBILINEAR  bilinear features at each location


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end


gpuMode = isa(x, 'gpuArray');



if backMode
    
    if gpuMode
        y = gpuArray(zeros(size(x), 'single'));
    else
        y = zeros(size(x), 'single');
    end
    for b=1:bs
        for yy=1:h
            for xx=1:w
                dzdy_b = reshape(dzdy(yy,xx,:,b), [ch, ch]);
                a = squeeze(x(yy,xx,:,b));
                y(yy, xx, :, b) = reshape(a*dzdy_b, [h, w, ch]);
            end
        end
    end
else
    
    if gpuMode
        y = gpuArray(zeros([h, w, ch*ch, bs], 'single'));
    else
        y = zeros([h, w, ch*ch, bs], 'single');
    end
    for b = 1:bs,
        for yy = 1:h,
            for xx = 1:w,
                a = squeeze(x(yy,xx,:,b));
                y(yy,xx,:, b) = reshape(a*a', [1 ch*ch]);
            end
        end
    end
end