function bird_demo(varargin)
setup;
% Default options
opts.model = 'data/ft_models/bcnn-cub-dm.mat';
opts.cubDir = 'data/cub';
opts.useGpu = false;
opts.svmPath = fullfile('data', 'models','svm_cub_vdm.mat');
opts.imgPath = 'test_image.jpg';
opts.regionBorder = 0.05 ;
opts.normalization = 'sqrt_L2';
opts.topK = 5; % Number of labels to display

% Parse user supplied options
opts = vl_argparse(opts, varargin);

% Load CUB database
tic;
imdb = cub_get_database(opts.cubDir, false, false);
fprintf('%.2fs to load imdb.\n', toc);

% Read image
origIm = imread(opts.imgPath);
im = single(origIm);

% Load classifier
tic;
classifier = load(opts.svmPath);

% Load the bilinear models and move to GPU if necessary
load(opts.model);
if opts.useGpu
    net = net_move_to_device(net, 'gpu');
else
    net = net_move_to_device(net, 'cpu');
end

fprintf('%.2fs to load models into memory.\n', toc);

tic;
% Compute B-CNN feature for this image
code = get_bcnn_features(net, im, ...
        'regionBorder', opts.regionBorder, ...
        'normalization', opts.normalization);

% Make predictions
scores = classifier.w'*code{1} + classifier.b';
[~, pred] = sort(scores, 'descend');

% Predict class labels
pred_class = classifier.classes(pred(1:opts.topK));
fprintf('Top %d prediction for %s:\n', opts.topK, opts.imgPath);
fprintf('%s\n', pred_class{:});
fprintf('%.2fs to make predictions [GPU=%d]\n', toc, opts.useGpu);

% Display 4 other images from the training set from the top class
N = 4; w = 224; h = 224;
classId = pred(1);
imageInd = find(imdb.images.label == classId & imdb.images.set == 1);
imageInd = imageInd(randperm(length(imageInd)));
classImage = cell(4,1);
for j=1:N
    classImage{j} = imresize(imread(fullfile(imdb.imageDir, imdb.images.name{imageInd(j)})), [w, h]);
    if(size(classImage{j}, 3) == 1) % Make color
        classImage{j} = repmat(classImage{j}, 1, 1, 3);
    end
end
montageImage = cat(1, classImage{:});
figure(1); clf;
subplot(4, 5, [1:4, 6:9, 11:14, 16:19]); imshow(origIm);
subplot(4, 5, 5*(1:4)); imshow(mat2gray(montageImage));
set(gcf,'NextPlot','add');axes;
h = title(pred_class{1}, 'interpret' ,'none');
set(gca,'Visible','off');set(h,'Visible','on'); 