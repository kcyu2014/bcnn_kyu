clear
setup;

svmPath = fullfile('data', 'models', 'svm_cub_vdm.mat');
imgPath = './test_image.jpg';
cubDir = fullfile('data', 'cub');

bcnn.opts = {...
    'type', 'bcnn', ...
    'modela', 'data/models/bcnn-cub-dm-neta.mat', ...
    'layera', 30,...
    'modelb', 'data/models/bcnn-cub-dm-netb.mat', ...
    'layerb', 14
} ;

imdb = cub_get_database(cubDir, false, false);


opts.type = 'bcnn' ;
opts.modela = '';
opts.modelb = '';
opts.layera = 0 ;
opts.layerb = 0 ;
opts.useGpu = 1 ;
opts.regionBorder = 0.05 ;
opts.normalization = 'sqrt_L2';

topi = 5;

im = imread(imgPath);
im = single(im);

classifier = load(svmPath);



optsStr = horzcat(bcnn.opts);
opts = vl_argparse(opts, optsStr);

% initialize the model
neta = load(opts.modela);
neta.layers = neta.layers(1:opts.layera);
netb = load(opts.modelb);
netb.layers = netb.layers(1:opts.layerb);
if opts.useGpu,
    neta = vl_simplenn_move(neta, 'gpu');
    netb = vl_simplenn_move(netb, 'gpu');
    neta.useGpu = true;
    netb.useGpu = true;
else
    neta = vl_simplenn_move(neta, 'cpu');
    netb = vl_simplenn_move(netb, 'cpu');
    neta.useGpu = false;
    netb.useGpu = false;
end

% compute B-CNN feature
code = get_bcnn_features(neta, netb,...
    im, ...
    'regionBorder', opts.regionBorder, ...
    'normalization', opts.normalization);

% svm prediction
scores = classifier.w'*code{1} + classifier.b';
[~, pred] = sort(scores, 'descend');


pred_class = classifier.classes(pred(1:topi));
fprintf('Top %d prediction:\n', topi);
fprintf('%s\n', pred_class{:});



N = 4;
w = 224;
h = 224;

i=pred(1);

list = find(imdb.images.label==i);
idx = list(randperm(numel(list), N));
    
for j=1:N
    ims{j} = imresize(imread(fullfile(imdb.imageDir, imdb.images.name{idx(j)})), [w, h]);
    if(size(ims{j}, 3)==1)
        ims{j} = repmat(ims{j}, 1, 1, 3);
    end
end   
    
im2 = cat(1, ims{:});
    
figure

subplot(4, 5, [1:4, 6:9, 11:14, 16:19])
imshow(mat2gray(im));

subplot(4, 5, 5*[1:4])
imshow(mat2gray(im2));

set(gcf,'NextPlot','add');
axes;
h = title(pred_class{1}, 'interpret' ,'none');
set(gca,'Visible','off');
set(h,'Visible','on'); 

