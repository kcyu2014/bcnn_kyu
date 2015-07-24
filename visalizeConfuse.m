clear

dataset = 'cub';
modelname = 'bcnnvdm';
Ni = 10;

opts.facescrubDir = 'data/facescrub' ;
opts.mitDir = 'data/mit_indoor';
opts.cubDir = 'data/cub';

switch dataset
    case 'cubcrop'
        imdb = cub_get_database_validation(opts.cubDir, true);
%         imdb = cub_get_database(opts.cubDir, true);
    case 'cub'
        imdb = cub_get_database(opts.cubDir, false);
    case 'mitindoor'
        imdb = mit_indoor_get_database(opts.mitDir);
    case 'facescrub'
        imdb = facescrub_get_database(opts.facescrubDir) ;
    otherwise
        error('Unknown dataset %s', opts.dataset) ;
end

resultPrefix = 'bcnn-train-fine-tuned_vdm';
resultPath = fullfile('data', resultPrefix, [dataset, '-seed-01']);

load(fullfile(resultPath, ['result-', modelname, '.mat']))

test_acc = diag(test.confusion);

[sort_test_acc, sort_test_idx] = sort(test_acc, 'ascend');

vc = sort_test_idx(3);


confusing = test.confusion(vc, :);
[confuseScore, confuse_class] = sort(confusing, 'descend');

idx = (confuse_class == vc);
confuse_class(idx) = [];
confuseScore(idx) = [];

imgIdx = arrayfun(@(x) find(imdb.images.label==x, Ni), confuse_class(1:8), 'UniformOutput', false);
imgIdx = cat(1, imgIdx{:});

imgIdx = [imgIdx; find(imdb.images.label==vc, Ni)];

for i=1:Ni
    figure(1)
    for j=1:9
        im = imread(fullfile(imdb.imageDir, imdb.images.name{imgIdx(j,i)}));
        subplot(3, 3, j);
        imagesc(im);
        if(j==1)
            title(imdb.classes.name{vc}, 'Interpreter', 'none')
        else
            title(imdb.classes.name{confuse_class(j-1)}, 'Interpreter', 'none')
        end
        axis off
    end
    pause
end