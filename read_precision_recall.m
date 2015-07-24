clear
run setup
%run vlfeat/toolbox/vl_setup
% model_path = 'bcnn-train-svmtest-mm_one_pca_64_nonrelu';
% model_path = 'bcnn-train-fine-tuned_mm';
model_path = 'bcnn-train-fine-tuned_mm_one_pca_64_nonrelu';
model_name = 'bcnnmmpca';
dataset = 'cub-seed-01';

% model_path = 'v2';
% model_name = 'bcnnmm';
% dataset = 'cub-seed-01';


load(fullfile('data', model_path, dataset, ['result-', model_name, '.mat']))
imdb = load(fullfile('data', model_path, dataset, 'imdb', 'imdb-seed-1.mat'))

[C, N] = size(scores);

train = ismember(imdb.images.set, [1 2]);
test = ismember(imdb.images.set, 3);

trainInfo = cell(C,1);
testInfo = cell(C,1);
for c=1:C
    y = 2*(imdb.images.label == c) - 1 ;
    [trainInfo{c}.recall, trainInfo{c}.precision, i]= vl_pr(y(train), scores(c, train)) ; ap(c) = i.ap ; ap11(c) = i.ap_interp_11 ;
    [testInfo{c}.recall, testInfo{c}.precision, i]= vl_pr(y(test), scores(c, test)) ; tap(c) = i.ap ; tap11(c) = i.ap_interp_11 ;
    [trainInfo{c}.nrecall,trainInfo{c}.nprecision, i]= vl_pr(y(train), scores(c, train), 'normalizeprior', 0.01) ; nap(c) = i.ap ;
    [testInfo{c}.nrecall, testInfo{c}.nprecision, i]= vl_pr(y(test), scores(c, test), 'normalizeprior', 0.01) ; tnap(c) = i.ap ;
    
end
%{
plotRecall = 0:0.05:1;
plotPrecision = zeros(C, numel(plotRecall));
for c=1:C
    minRecall = min(testInfo{c}.recall);
    plotPrecision(c,:) = interp1(testInfo{c}.precision, testInfo{c}.recall, plotRecall, 'linear', 'extrap');    
end
plotPrecision(plotPrecision>1) = 1;
plotPrecision(plotPrecision<0) = 0;
plotPrecision = mean(plotPrecision, 1);


plot(plotPrecision, plotRecall, 'r-')
%}

%
plotRecall = 0:0.01:1;

plotPrecision = zeros(C, numel(plotRecall));
plotPrecision2 = zeros(C, numel(plotRecall));
for c=1:C
    [u_recall, ia, ib] = unique(testInfo{c}.recall);
    u_recall = testInfo{c}.recall(ia);
    u_precision = testInfo{c}.precision(ia);
    plotPrecision2(c,:) = interp_pr(testInfo{c}.precision, testInfo{c}.recall, plotRecall);
    plotPrecision(c,:) = interp1(u_recall, u_precision, plotRecall);
end
plotPrecision = mean(plotPrecision, 1);
plot(plotRecall, plotPrecision, 'r-');
axis([0,1,0,1])

save(fullfile('data', model_path, dataset, 'result-precision-recall.mat'), 'plotPrecision', 'plotRecall')

% hold on


%{
recall = zeros(1, numel(testInfo{1}.recall));
precision = zeros(1, numel(testInfo{1}.precision));
for c=1:C
    recall = recall + testInfo{c}.recall;
    precision = precision + testInfo{c}.precision;
end

recall = recall./C;
precision = precision./C;

plot(recall, precision, 'b--')
axis([0,1,0,1])
%}

%{
[recall, ia, ~] = unique(recall);
precision = interp1(precision(ia), recall, plotRecall);
precision(1) = 1;

hold on
plot(plotRecall, precision, 'b--');
axis([0,1,0,1])
%}
% hold off
