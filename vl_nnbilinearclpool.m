function [y, varargout] = vl_nnbilinearclpool(x1, x2, varargin)
% VL_NNBILINEARPOOL pools bilinear feature across all locations


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
end


gpuMode = isa(x1, 'gpuArray');

[h1, w1, ch1, bs] = size(x1);
[h2, w2, ch2, ~] = size(x2);

assert(h1==w1&&h2==w2, 'vl_nnbilinearclpool only supports square images ');
assert(h1==h2&&w1==w2, 'two layer outputs must be the same resolution');
h = h1; w = w1;


if backMode
   
    if gpuMode
        y1 = gpuArray(zeros(size(x1), 'single'));
        y2 = gpuArray(zeros(size(x2), 'single'));
    else
        y1 = zeros(size(x1), 'single');
        y2 = zeros(size(x2), 'single');
    end
    for b=1:bs
        dzdy_b = reshape(dzdy(1,1,:,b), [ch1, ch2]);
        A = reshape(x1(:,:,:,b), [h1*w1, ch1]);
        B = reshape(x2(:,:,:,b), [h2*w2, ch2]);
        y2(:,:,:,b) = reshape(A*dzdy_b, [h1, w1, ch2]);
        y1(:,:,:,b) = reshape(B*dzdy_b', [h2, w2, ch1]);
        %{
        dydx = reshape(A*dzdy_b, [h1, w1, ch2]);
        y2(:, :, :, b) = imresize(dydx, h2/h1);
        dydx = reshape(B*dzdy_b', [h2, w2, ch1]);
        y1(:, :, :, b) = imresize(dydx, h1/h2);
        %}
    end
%     y = 0.5.*y2;
%     varargout{1} = y;
    y = y1;
    varargout{1} = y2;
else
    
    %{
    if w1*h1 <= w2*h2,
        %downsample B
        B = array_resize(x2, w1, h1);
        A = x1;
        w = w1;
        h = h1;
        %     A = reshape(A, [w1*h1 size(A,3)]);
    else
        %downsample A
        A = array_resize(x1, w2, h2);
        B = x2;
        w = w2;
        h = h2;
        %     B = reshape(B, [w2*h2 size(B,3)]);
    end
    %}
    if gpuMode
        y = gpuArray(zeros([1, 1, ch1*ch2, bs], 'single'));
    else
        y = zeros([1, 1, ch1*ch2, bs], 'single');
    end
    for b = 1:bs,
        %{
        xa = reshape(A(:,:,:,b), [h*w, ch1]);
        xb = reshape(B(:,:,:,b), [h*w, ch2]);
        %}
        
        xa = reshape(x1(:,:,:,b), [h*w, ch1]);
        xb = reshape(x2(:,:,:,b), [h*w, ch2]);
        
        y(1,1,:, b) = reshape(xa'*xb, [1 ch1*ch2]);
    end
end


function Ar = array_resize(A, w, h)
numChannels = size(A, 3);
indw = round(linspace(1,size(A,2),w));
indh = round(linspace(1,size(A,1),h));

Ar = A(indh, indw, :, :);

% gpuMode = isa(A, 'gpuArray');
% if gpuMode
%     Ar = gpuArray(zeros(w*h, numChannels, 'single'));
% else
%     Ar = zeros(w*h, numChannels, 'single');
% end
% for i = 1:numChannels,
%     Ai = A(indh,indw,i);
%     Ar(:,i) = Ai(:);
% end
