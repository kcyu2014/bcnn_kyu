function imdb = stanford_dogs_get_database(dogDir, useCropped)
% Automatically change directories
if useCropped
    imdb.imageDir = fullfile(dogDir, 'images_cropped') ;
else
    imdb.imageDir = fullfile(dogDir, 'images');
end

% Load structure file_list, labels, annotation_list
load(fullfile(dogDir, 'lists', 'file_list.mat'));

imdb.maskDir = fullfile(dogDir, 'masks'); % doesn't exist
imdb.sets = {'train', 'val', 'test'};


% Image names
imdb.images.name = file_list;
imdb.images.id = (1:numel(imdb.images.name));

% Class labels
imdb.images.label = reshape(labels, 1, numel(labels));

% Class names
numClass = max(imdb.images.label);
imdb.classes.name = cell(1, numClass);
for i = 1:numClass, 
    ind = find(imdb.images.label == i, 1); % first image with this label
    imagenetName = fileparts(imdb.images.name{ind});
    imdb.classes.name{i} = imagenetName;
end    

% Image sets
tr = load(fullfile(dogDir, 'lists','train_list.mat'));
te = load(fullfile(dogDir, 'lists','test_list.mat'));
imdb.images.set = zeros(1,length(imdb.images.id));
trainSet = ismember(imdb.images.name, tr.file_list);
testSet = ismember(imdb.images.name, te.file_list);
imdb.images.set(trainSet) = 1;
imdb.images.set(testSet) = 3;

% Sanity check
assert(all(trainSet ~= testSet));
assert(all(imdb.images.set > 0));

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.images.difficult = false(1, numel(imdb.images.id)) ; 
