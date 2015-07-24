function [code, varargout]= get_bcnn_features_appendfeature(neta, netb, im, varargin)
% GET_BCNN_FEATURES  Get bilinear cnn features for an image
%   This function extracts the binlinear combination of CNN features
%   extracted from two different networks.

nVargOut = max(nargout,1)-1;

if nVargOut==1 
    assert(true, 'Number of output should not be two.')
end

opts.crop = true ;
%opts.scales = 2.^(1.5:-.5:-3); % try a bunch of scales
opts.scales = 2;
opts.encoder = [] ;
opts.regionBorder = 0.05;
opts.normalization = 'sqrt';
opts.appft = [];
opts.colorGmm = [];
opts = vl_argparse(opts, varargin) ;

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

if nVargOut==2
    im_resA = cell(numel(im), 1);
    im_resB = cell(numel(im), 1);
end

netbAppLayers = [];
appColor = false;
if(~isempty(opts.appft))
    for i=1:numel(opts.appft)
        for j=1:numel(netb.layers)
            if(strcmp(netb.layers{j}.name, opts.appft{i}))
                netbAppLayers(end+1) = j;
            end
        end
        if(strcmp('color', opts.appft{i}))
            appColor = true;
        end
    end
end


% for each image
for k=1:numel(im)
    im_cropped = imresize(single(im{k}), imageSizeA([2 1]), 'bilinear');
    crop_h = size(im_cropped,1) ;
    crop_w = size(im_cropped,2) ;
    resA = [] ;
    resB = [] ;
    psi = cell(1, numel(opts.scales));
    % for each scale
    for s=1:numel(opts.scales)
        if min(crop_h,crop_w) * opts.scales(s) < min(borderA, borderB), continue ; end
        if sqrt(crop_h*crop_w) * opts.scales(s) > 1024, continue ; end

        % resize the cropped image and extract features everywhere
        im_resized = imresize(im_cropped, opts.scales(s)) ;
        im_resizedA = bsxfun(@minus, im_resized, averageColourA) ;
        im_resizedB = bsxfun(@minus, im_resized, averageColourB) ;
        if nVargOut==2
            im_resA{k} = im_resizedA;
            im_resB{k} = im_resizedB;
        end
        if neta.useGpu
            im_resizedA = gpuArray(im_resizedA) ;
            im_resizedB = gpuArray(im_resizedB) ;
        end
        if(isempty(netbAppLayers))
            resA = vl_simplenn(neta, im_resizedA, [], resA, ...
                'conserveMemory', true, 'sync', true);
            resB = vl_simplenn(netb, im_resizedB, [], resB, ...
                'conserveMemory', true, 'sync', true);
        else
            resA = vl_simplenn(neta, im_resizedA, [], resA, ...
                'conserveMemory', false, 'sync', true);
            resB = vl_simplenn(netb, im_resizedB, [], resB, ...
                'conserveMemory', false, 'sync', true);
        end
        A = gather(resA(end).x);
        B = gather(resB(end).x);
        
        if(~isempty(netbAppLayers))
            for ap=1:numel(netbAppLayers)
                apB = gather(resB(netbAppLayers(ap)+1).x);
                apB = array_resize(apB, size(B,1), size(B,2), false);
                B = cat(3, B, apB);
            end
        end
        
        if(appColor)
            gmm_num = size(opts.colorGmm.gmm_means, 2);
            if(size(A,1)*size(A,2)>size(B,1)*size(B,2))
                fSize = [size(B,1), size(B,2)];
            else
                fSize = [size(A,1), size(A,2)];
            end
            im2 = im_cropped./255;
            im2 = imresize(im2, fSize, 'bilinear' );
            im2 = transpose(reshape(im2, fSize(1)*fSize(2), 3));
            
%             p = gmmLikilehood(im2, opts.colorGmm);
            p = gmmPosterior(im2, opts.colorGmm);
            p = reshape(p', fSize(1), fSize(2), gmm_num);
            B = cat(3, B, p);
        end
        

        psi{s} = bilinear_pool(A,B);
        feat_dim = max(cellfun(@length,psi));
        code{k} = zeros(feat_dim, 1);
    end
    % pool across scales
    for s=1:numel(opts.scales),
        if ~isempty(psi{s}),
            code{k} = code{k} + psi{s};
        end
    end
    assert(~isempty(code{k}));
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

if nVargOut==2
    varargout{1} = cat(4, im_resA{:});
    varargout{2} = cat(4, im_resB{:});
end


function psi = bilinear_pool(A, B)
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

% bilinear pool
psi = A'*B;
psi = psi(:);

function Ar = array_resize(A, w, h, varargin)

if(isempty(varargin))
    doreshape = true;
else
    doreshape = varargin{1};
end

numChannels = size(A, 3);
indw = round(linspace(1,size(A,2),w));
indh = round(linspace(1,size(A,1),h));
if(doreshape)
    Ar = zeros(w*h, numChannels, 'single');
    for i = 1:numChannels,
        Ai = A(indh,indw,i);
        Ar(:,i) = Ai(:);
    end
else
    Ar = zeros(h, w, numChannels, 'single');
    for i = 1:numChannels,
        Ai = A(indh,indw,i);
        Ar(:,:, i) = Ai;
    end
end

function p = gmmLikilehood(data, model)

[d, M] = size(model.gmm_means);
N = size(data, 2);
p = zeros(M, N);
for i=1:M
    x = bsxfun(@minus, data, model.gmm_means(:,i));
    x = bsxfun(@rdivide, x.^2, model.gmm_covariances(:,i));
    cprod = cumprod(model.gmm_covariances(:,i));
    p(i,:) = exp(-0.5.*sum(x, 1))./(sqrt(cprod(end)*((2*pi)^d)));
end

function p = gmmPosterior(data, model)

[d, M] = size(model.gmm_means);
N = size(data, 2);
% p = zeros(M, N);
logp = zeros(M, N);
for i=1:M
    x = bsxfun(@minus, data, model.gmm_means(:,i));
    x = bsxfun(@rdivide, x.^2, model.gmm_covariances(:,i));
    cprod = cumprod(model.gmm_covariances(:,i));
    logp(i,:) = -0.5.*sum(x, 1)./(sqrt(cprod(end)*((2*pi)^d)));
    logp(i,:) = logp(i,:) + log(model.gmm_priors(i));
end

partZ = logsumexp(logp, 1);
p = bsxfun(@minus, logp, partZ);
p = exp(p);
