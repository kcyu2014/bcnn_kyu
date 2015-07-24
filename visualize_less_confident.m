clear

showImg = false;

dataset = 'aircraft-variant-seed-01';
modelPath = 'bcnn-train-fine-tuned_vdm';
modelName = 'bcnnvdm';


load(fullfile('data', modelPath, dataset, ['result-', modelName, '.mat']))
imdb = load(fullfile('data', modelPath, dataset, 'imdb', 'imdb-seed-1.mat'));


if(~exist(fullfile('figOutput', 'confusing_imgs', dataset)))
    mkdir(fullfile('figOutput', 'confusing_imgs', dataset));
end

classNum = numel(imdb.classes.name);
classFileName = cell(classNum, 1);
for i=1:classNum
    classFileName{i} = textscan(imdb.classes.name{i}, '%s', 'Delimiter', '.');
    classFileName{i} = cat(2, classFileName{i}{1}{:});
end

Nc = 6;
Ni = 5;



[pred_score, pred] = max(scores, [], 1);

score_confuse = bsxfun(@rdivide, scores, pred_score);



consfuse_pair = test.confusion + test.confusion';
consfuse_pair = triu(consfuse_pair, 1);
[confuse, pair_idx] = sort(consfuse_pair(:), 'descend');

im_vis = cell(2, Nc);
class_vis = cell(2, Nc);
imgSize = [224, 224];
for i=1:Nc
    [c1, c2] = ind2sub([classNum, classNum], pair_idx(i));
    
    img_c1 = find((imdb.images.label == c1));
    score_confuse1 = score_confuse(c2, img_c1);
    
    [~, sidx1] = sort(score_confuse1, 'descend');
    
    img_c2 = find((imdb.images.label == c2));
    score_confuse2 = score_confuse(c1, img_c2);
    
    [~, sidx2] = sort(score_confuse2, 'descend');
    

    for j=1:Ni
        im_vis{1, i}{j} = imresize(imread(fullfile(imdb.imageDir, imdb.images.name{img_c1(sidx1(j))})), imgSize);
        im_vis{2, i}{j} = imresize(imread(fullfile(imdb.imageDir, imdb.images.name{img_c2(sidx2(j))})), imgSize);
        if(size(im_vis{1, i}{j}, 3)==1)
            im_vis{1, i}{j} = uint8(255.*repmat(double(im_vis{1, i}{j})./max(max(double(im_vis{1, i}{j}))), [1 1 3]));
        end
        if(size(im_vis{2, i}{j}, 3)==1)
            im_vis{2, i}{j} = uint8(255.*repmat(double(im_vis{2, i}{j})./max(max(double(im_vis{2, i}{j}))), [1 1 3]));
        end
        
        if(showImg)
            figure(2)
            subplot(1,2,1)
            imshow(im_vis{1, i}{j})
            title([imdb.classes.name{c1}, ' -> ', imdb.classes.name{c2}], 'interpreter', 'none')
            subplot(1,2,2)
            imshow(im_vis{2, i}{j})
            title([imdb.classes.name{c2}, ' -> ', imdb.classes.name{c1}], 'interpreter', 'none')
            pause
        end
    end
      
    im_vis{1,i} = cat(2, im_vis{1,i}{:});
    im_vis{2,i} = cat(2, im_vis{2,i}{:});
    class_vis{1,i} = classFileName{c1};
    class_vis{2,i} = classFileName{c2};
%     [~, class_vis{1,i}] = strtok(imdb.classes.name{c1}, '.');
%     [~, class_vis{2,i}] = strtok(imdb.classes.name{c2}, '.');
%     class_vis{1,i}(1) = [];
%     class_vis{2,i}(1) = [];
    
    if(~exist(fullfile('figOutput', 'confusing_imgs', dataset, ['pair_', num2str(i, '%.2d')]), 'dir'))
        mkdir(fullfile('figOutput', 'confusing_imgs', dataset, ['pair_', num2str(i, '%.2d')]));
    end
    
    imwrite(im_vis{1,i}, fullfile('figOutput', 'confusing_imgs', dataset, ['pair_', num2str(i, '%.2d')], [class_vis{1,i}, '.jpg']), 'jpg');
    imwrite(im_vis{2,i}, fullfile('figOutput', 'confusing_imgs', dataset, ['pair_', num2str(i, '%.2d')], [class_vis{2,i}, '.jpg']), 'jpg');
end



%% generate latex



fid = fopen(fullfile('figOutput', 'confusing_imgs', dataset, 'confuse-imgs.tex'), 'w');


fprintf(fid, '\\documentclass[10pt,twocolumn,letterpaper]{article}\n');
fprintf(fid, '\\usepackage{graphicx}\n\\usepackage[space]{grffile}\n\n\\begin{document}\n\n\n\n');

fprintf(fid, '\\begin{figure}\n\\begin{center}\n\\begin{tabular}{@{}c|c@{}}\n');

for i=1:Nc
    fprintf(fid, '\\includegraphics[width=0.45\\linewidth]{%s} &\n', fullfile(['pair_', num2str(i, '%.2d')], [class_vis{1,i}, '.jpg']));
    fprintf(fid, '\\includegraphics[width=0.45\\linewidth]{%s} \\\\\n', fullfile(['pair_', num2str(i, '%.2d')], [class_vis{2,i}, '.jpg']));
    
    str1 = class_vis{1,i};
    str2 = class_vis{2,i};
    idx1 = find(str1=='_');
    idx2 = find(str2=='_');
    for j=1:numel(find(idx1))
        str1 = [str1(1:idx1(end-j+1)-1), '\', str1(idx1(end-j+1):end)];
    end
    for j=1:numel(find(idx2))
        str2 = [str2(1:idx2(end-j+1)-1), '\', str2(idx2(end-j+1):end)];
    end
    
    fprintf(fid, '%s &\n%s \\\\\n', str1, str2);
end

fprintf(fid, '\\end{tabular}\n\\end{center}\n\\end{figure}\n\n');

fprintf(fid, '\\end{document}');

fclose(fid);


%{
[pred_score, pred] = max(scores, [], 1);

score_confuse = bsxfun(@minus, scores, pred_score);

[confuse, confuse_idx] = max(score_confuse, [], 1);

test_acc = diag(test.confusion);

[sort_test_acc, sort_test_idx] = sort(test_acc, 'ascend');

im_vis = cell(1, Nc);

for i=1:Nc
    vc = sort_test_idx(i);
    
    row_confuse = test.confusion(vc, :);
    row_confuse(vc) = 0;
    [~, confuse_class] = sort(row_confuse, 'descend');
    confuse_class = confuse_class(1);
    
    disp([imdb.classes.name{vc}, ' -> ', imdb.classes.name{confuse_class}])
    
    img_class_vc = find((imdb.images.label == vc));
    img_idx = (confuse_idx(img_class_vc) == confuse_class);
    img_idx = img_class_vc(img_idx);
    im_vis{i} = cell(1, Ni);
    
    for j=1:Ni
        im_vis{i}{j} = imread(fullfile(imdb.imageDir, imdb.images.name{img_idx(j)}));
        subplot(1,2,1)
        imshow(im_vis{i}{j})
        subplot(1,2,2)
        idx = find(imdb.images.label==confuse_class);
        im = imread(fullfile(imdb.imageDir, imdb.images.name{idx(randperm(numel(idx),1))}));
        imshow(im)
        title([imdb.classes.name{vc}, ' -> ', imdb.classes.name{confuse_class}], 'interpreter', 'none')
        pause
    end
end
%}
