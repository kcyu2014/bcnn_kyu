% function [ output_args ] = visualize_topNImg( input_args )

loadBatchFileFlag = false;

numImg = 11788;
numBatch = 93;
allACell = cell(1, numImg); 
allBCell = cell(1, numImg);
allImCell = cell(1, numImg);

% save max values and locations
allAMax = zeros(512, numImg);
allBMax = zeros(512, numImg);
allAMaxLoc = zeros(512, 2, numImg);
allBMaxLoc = zeros(512, 2, numImg);

filterSize = [28, 28, 512; 27, 27, 512];

%% collect A-response, B-response, Imgs 
if loadBatchFileFlag == true
    imgIndex = 1; 
    for b = 1:numBatch
        loadFileName = sprintf('./output/response/res_batchID=%d.mat', b);
        load(loadFileName, 'ACell', 'BCell', 'ImCell');
        fprintf('loading %s...\n', loadFileName);

        for bi = 1:numel(ACell)
            allACell{imgIndex} = ACell{bi};
            allBCell{imgIndex} = BCell{bi};
            allImCell{imgIndex} = ImCell{bi};

            % Get max values and locations of resA each filter
            resA_ = reshape(ACell{bi}, [28*28, 512]);
            [maxA, maxLocA] = max(resA_);
            maxA_x = ceil(maxLocA/28); 
            maxA_y = maxLocA - (maxA_x-1)*28;
            allAMax(:, imgIndex) = maxA'; 
            allAMaxLoc(:, 1, imgIndex) = maxA_x';
            allAMaxLoc(:, 2, imgIndex) = maxA_y';
            % Get max values and locations of resB each filter
            resB_ = reshape(BCell{bi}, [27*27, 512]);
            [maxB, maxLocB] = max(resB_);
            maxB_x = ceil(maxLocB/27); 
            maxB_y = maxLocB - (maxB_x-1)*27;
            allBMax(:, imgIndex) = maxB'; 
            allBMaxLoc(:, 1, imgIndex) = maxB_x';
            allBMaxLoc(:, 2, imgIndex) = maxB_y';

            imgIndex = imgIndex + 1;
        end
    end

    save('./output/response/allACell.mat', 'allACell', '-v7.3');
    save('./output/response/allBCell.mat', 'allBCell', '-v7.3');
    save('./output/response/allImCell.mat', 'allImCell', '-v7.3');
    save('./output/response/allAMax.mat', 'allAMax', '-v7.3');
    save('./output/response/allAMaxLoc.mat', 'allAMaxLoc', '-v7.3');
    save('./output/response/allBMax.mat', 'allBMax', '-v7.3');
    save('./output/response/allBMaxLoc.mat', 'allBMaxLoc', '-v7.3');
else
    load('./output/response/allACell.mat', 'allACell');
    load('./output/response/allBCell.mat', 'allBCell');
    load('./output/response/allImCell.mat', 'allImCell');
    load('./output/response/allAMax.mat', 'allAMax');
    load('./output/response/allAMaxLoc.mat', 'allAMaxLoc');
    load('./output/response/allBMax.mat', 'allBMax');
    load('./output/response/allBMaxLoc.mat', 'allBMaxLoc');
end

%% Retrieve top N highest images for each filter 
topN = 10; % number of images to be retrieved for each filter

for f = 1:512
%     [~, topNImg_A] = sort(allAMax(f, :), 'descend');
%     figure('units', 'normalized', 'outerposition', [0 0 1 0.4]);
%     for i = 1:topN
%         topNindex = topNImg_A(i);
%         im_resized_ = allImCell{topNindex}/255;
%         patchSize = size(im_resized_)/28;
%         
%         if allAMax(f, topNindex) == 0
%             continue;
%         else
%             x = (allAMaxLoc(f, 1, topNindex) - 1) * patchSize(1) + 1;
%             y = (allAMaxLoc(f, 2, topNindex) - 1) * patchSize(1) + 1;
%             im_resized_ = insertShape(im_resized_, ...
%                 'Rectangle', [x, y, patchSize(1), patchSize(2)], ...
%                 'LineWidth', 4, ...
%                 'Color', 'green');
%         end
%         
%         subplot_tight(2, topN, i); imshow(im_resized_); 
%         title(i); 
%         subplot_tight(2, topN, i + topN); 
%         imagesc(allACell{topNindex}(:, :, f)); colormap gray;
%         set(gca, 'xtick', []);
%         set(gca, 'ytick', []);
%         axis square
%         title(allAMax(f, topNindex));
%     end
%     titleText = sprintf('./output/response/A/top%d_[netA,filter%d].png', topN, f);  
%     windowImg = getframe(gcf);
%     imwrite(windowImg.cdata, titleText);
%     close all

    [~, topNImg_B] = sort(allBMax(f, :), 'descend');
    figure('units', 'normalized', 'outerposition', [0 0 1 0.4]);
    for i = 1:topN
        topNindex = topNImg_B(i);
        im_resized_ = allImCell{topNindex}/255;
        patchSize = size(im_resized_)/27;
        
        if allBMax(f, topNindex) == 0
            continue;
        else
            x = (allBMaxLoc(f, 1, topNindex) - 1) * patchSize(1) + 1;
            y = (allBMaxLoc(f, 2, topNindex) - 1) * patchSize(1) + 1;
            im_resized_ = insertShape(im_resized_, ...
                'Rectangle', [x, y, patchSize(1), patchSize(2)], ...
                'LineWidth', 4, ...
                'Color', 'green');
        end
        
        subplot_tight(2, topN, i); imshow(im_resized_);
        title(i); 
        subplot_tight(2, topN, i + topN); 
        imagesc(allBCell{topNindex}(:, :, f)); colormap gray;
        set(gca, 'xtick', []);
        set(gca, 'ytick', []);
        axis square
        title(topNindex);
    end
    titleText = sprintf('./output/response/B/top%d_[netB,filter%d].png', topN, f);  
    windowImg = getframe(gcf);
    imwrite(windowImg.cdata, titleText);
    close all  
end

