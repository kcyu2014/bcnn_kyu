function [code, varargout]= get_bcnn_features_onescale_batch_dropoutv2(neta, netb, im, varargin)
% GET_BCNN_FEATURES  Get bilinear cnn features for an image
%   This function extracts the binlinear combination of CNN features
%   extracted from two different networks.

nVargOut = max(nargout,1)-1;

if nVargOut==1 || nVargOut==2 || nVargOut==3
    assert(true, 'Number of output problem')
end

opts.crop = true ;
%opts.scales = 2.^(1.5:-.5:-3); % try a bunch of scales
opts.scales = 2;
opts.encoder = [] ;
opts.regionBorder = 0.05;
opts.normalization = 'sqrt';
opts.drop_rate = 0;
opts = vl_argparse(opts, varargin) ;

sd = single(1 / ((1 - opts.drop_rate)^2));

% get parameters of the network
info = vl_simplenn_display(neta) ;
borderA = round(info.receptiveField(end)/2+1) ;
averageColourA = mean(mean(neta.normalization.averageImage,1),2) ;
imageSizeA = neta.normalization.imageSize;

info = vl_simplenn_display(netb) ;
borderB = round(info.receptiveField(end)/2+1) ;
averageColourB = mean(mean(netb.normalization.averageImage,1),2) ;
imageSizeB = netb.normalization.imageSize;

assert(all(imageSizeA == imageSizeB));

if ~iscell(im)
  im = {im} ;
end

code = cell(1, numel(im));

    im_resA = cell(numel(im), 1);
    im_resB = cell(numel(im), 1);
    resA = [] ;
    resB = [] ;
    
% for each image
for k=1:numel(im)
    im_cropped = imresize(single(im{k}), imageSizeA([2 1]), 'bilinear');
    crop_h = size(im_cropped,1) ;
    crop_w = size(im_cropped,2) ;
    %psi = cell(1, numel(opts.scales));
    
    s = 1;
    
    
    if min(crop_h,crop_w) * opts.scales(s) < min(borderA, borderB), continue ; end
    if sqrt(crop_h*crop_w) * opts.scales(s) > 1024, continue ; end
    
    % resize the cropped image and extract features everywhere
    im_resized = imresize(im_cropped, opts.scales(s)) ;
    im_resizedA = bsxfun(@minus, im_resized, averageColourA) ;
    im_resizedB = bsxfun(@minus, im_resized, averageColourB) ;
    
    im_resA{k} = im_resizedA;
    im_resB{k} = im_resizedB;
    
    
end

im_resA = cat(4, im_resA{:});
im_resB = cat(4, im_resB{:});
  
if neta.useGpu
    im_resA = gpuArray(im_resA) ;
    im_resB = gpuArray(im_resB) ;
end
resA = vl_simplenn_v2(neta, im_resA, [], resA, ...
    'conserveMemory', true, 'sync', true);
resB = vl_simplenn_v2(netb, im_resB, [], resB, ...
    'conserveMemory', true, 'sync', true);
A = gather(resA(end).x);
B = gather(resB(end).x);

if(opts.drop_rate)
    dropA = (single(rand(size(A,3), size(A,4)))<opts.drop_rate);
    dropB = (single(rand(size(B,3), size(B,4)))<opts.drop_rate);
end

for k=1:numel(im)
    if opts.drop_rate
        code{k} = bilinear_pool(squeeze(A(:,:,:,k)), squeeze(B(:,:,:,k)), dropA(:,k), dropB(:,k));
    else
        code{k} = bilinear_pool(squeeze(A(:,:,:,k)), squeeze(B(:,:,:,k)), [], []);
    end
    assert(~isempty(code{k}));
    if(opts.drop_rate)
        code{k} = sd.*code{k};
    end
end
        
        
        


% square-root and l2 normalize (like: Improved Fisher?)
switch opts.normalization
    case 'sqrt'
        for k=1:numel(im),
            code{k} = sign(code{k}).*sqrt(abs(code{k}));
            code{k} = code{k}./(norm(code{k}+eps));
        end
    case 'L2'
        for k=1:numel(im),
            code{k} = code{k}./(norm(code{k}+eps));
        end
    case 'none'
end

if nVargOut>=4
    varargout{1} = im_resA;
    varargout{2} = im_resB;
    varargout{3} = resA;
    varargout{4} = resB;
end
if opts.drop_rate
    varargout{5} = dropA;
    varargout{6} = dropB;
end


function psi = bilinear_pool(A, B, dropA, dropB)
w1 = size(A,2) ;
h1 = size(A,1) ;
w2 = size(B,2) ;
h2 = size(B,1) ;

%figure(1); clf;
%montage(reshape(A, [h1 w1 1 size(A,3)]));
%figure(2); clf;
%montage(reshape(B, [h2 w2 1 size(B,3)]));
%pause;

if w1*h1 <= w2*h2,
    %downsample B
    B = array_resize(B, w1, h1);
    A = reshape(A, [w1*h1 size(A,3)]);
else
    %downsample A
    A = array_resize(A, w2, h2);
    B = reshape(B, [w2*h2 size(B,3)]);
end

if(~isempty(dropA))
    A(:,dropA) = 0;
    B(:,dropB) = 0;
end

% bilinear pool
psi = A'*B;
psi = psi(:);

function Ar = array_resize(A, w, h)
numChannels = size(A, 3);
indw = round(linspace(1,size(A,2),w));
indh = round(linspace(1,size(A,1),h));
Ar = zeros(w*h, numChannels, 'single');
for i = 1:numChannels,
    Ai = A(indh,indw,i);
    Ar(:,i) = Ai(:);
end
