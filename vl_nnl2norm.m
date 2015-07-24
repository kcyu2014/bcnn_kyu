function y = vl_nnl2norm(x, varargin)
% VL_L2NORM computes l2 normalization at each location


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end


gpuMode = isa(x, 'gpuArray');

[h, w, ch, bs] = size(x);
if gpuMode
    y = gpuArray(zeros([h, w, ch, bs], 'single'));
else
    y = zeros([h, w, ch, bs], 'single');
end


x_norm = sqrt(sum(x.*x, 3)+eps);
if backMode
%     Bj = dzdy.*x;
%     B = bsxfun(@plus, -Bj, sum(Bj,3));
%     A = bsxfun(@times, -x, x_norm.^(-3));
%     C = bsxfun(@times, x.^2, x_norm.^(-3));
%     C = bsxfun(@plus, -C, x_norm.^(-1));
%     C = dzdy.*C;
%     y = A.*B+C;
    
    E = bsxfun(@times, dzdy, x_norm.^(-1));
    F = sum(x.*dzdy,3);
    F = F.*x_norm.^(-3);
    F = bsxfun(@times, x, F);
    y = E-F;
else
    %{
    for b = 1:bs,
        for yy = 1:h,
            for xx = 1:w,
                A = squeeze(x(yy,xx,:,b));
                y(yy,xx,:, b) = A./sqrt(A'*A+eps);
            end
        end
    end
    %}
    
%     y = bsxfun(@rdivide, x, x_norm);
    
    y = x./repmat(x_norm, [1, 1, ch, 1]);
end