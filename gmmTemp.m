function gmmTemp

load gmmUsed

if(size(A,1)*size(A,2)>size(B,1)*size(B,2))
    fSize = [size(B,1), size(B,2)];
else
    fSize = [size(A,1), size(A,2)];
end
im2 = im_cropped./255;
im2 = imresize(im2, fSize, 'bilinear' );
im2 = transpose(reshape(im2, fSize(1)*fSize(2), 3));

p = gmmLikilehood(im2, opts.colorGmm);


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

