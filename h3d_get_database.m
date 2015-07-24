function imdb = h3d_get_database(h3dDir, useCropped, doval)

% Automatically change directories
if useCropped
    imdb.imageDir = fullfile(h3dDir, 'images_cropped') ;
else
    imdb.imageDir = fullfile(h3dDir);
end

imdb.maskDir = fullfile(h3dDir, 'masks'); % doesn't exist
imdb.sets = {'train', 'val', 'test'};

% Class names
% [~, classNames] = textread(fullfile(h3dDir, 'classes.txt'), '%d %s');
% imdb.classes.name = horzcat(classNames(:));
imdb.classes.name = {'is_male', ...
                    'has_long_hair', ...
                    'has_glasses', ...
                    'has_hat', ...
                    'has_t-shirt', ...
                    'has_long_sleeves', ...
                    'has_shorts', ...
                    'has_jeans', ...
                    'has_long_pants'};

% Image names
fid = fopen(fullfile(h3dDir, 'train', 'labels.txt'));
trainRead = textscan(fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f');
fclose(fid);
imdb.images.name = cellfun(@(x) fullfile('train', x), trainRead{1}, 'UniformOutput', false);

fid = fopen(fullfile(h3dDir, 'test', 'labels.txt'));
testRead = textscan(fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f');
fclose(fid);
imdb.images.name = vertcat(imdb.images.name, cellfun(@(x) fullfile('test', x), testRead{1}, 'UniformOutput', false));
imdb.images.id = (1:numel(imdb.images.name));

% Class labels
imdb.images.label = horzcat(trainRead{6:end});
imdb.images.label = vertcat(imdb.images.label, horzcat(testRead{6:end}));

% Bounding boxes
imdb.images.bounds = horzcat(trainRead{2:5});
imdb.images.bounds = vertcat(imdb.images.bounds, horzcat(testRead{2:5}));
imdb.images.bounds = round(imdb.images.bounds);


% Image sets
imdb.images.set = 3.*ones(1,length(imdb.images.id));
imdb.images.set(1:numel(trainRead{1})) = 1;


if(doval)
    trainSize = numel(find(imdb.images.set==1));
    validSize = round(trainSize/3);
    
    trainIdx = find(imdb.images.set==1);
    
    % set 1/3 of train set to validation
    valIdx = trainIdx(randperm(trainSize, validSize));
    imdb.images.set(valIdx) = 2;
end



% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.images.difficult = false(1, numel(imdb.images.id)) ; 
