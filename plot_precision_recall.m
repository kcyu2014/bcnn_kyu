clear

addpath('/scratch1/tsungyulin/matlabToolbox/export_fig-master/')
dataset = 'cub-seed-01';

% outputFigName = 'vdmPRCurve';
% settingNameList = {'bcnnvdm', 'bcnnvdmpca64', 'bcnnvdmftpca64'};
% modelPathList = {'bcnn-train-fine-tuned_vdm',...
%                 'bcnn-train-svmtest-vdm_one_pca_64_nonrelu',...
%                 'bcnn-train-fine-tuned_vdm_one_pca_64_nonrelu'};
% modelNameList = {'bcnnvdm', 'bcnnvdmpca', 'bcnnvdmpca'};

% outputFigName = 'mmPRCurve';
% settingNameList = {'bcnnmm', 'bcnnmmpca64', 'bcnnftmmpca64'};
% 
% modelPathList = {'bcnn-train-fine-tuned_mm',...
%                 'bcnn-train-svmtest-mm_one_pca_64_nonrelu',...
%                 'bcnn-train-fine-tuned_mm_one_pca_64_nonrelu'};
% modelNameList = {'bcnnmm', 'bcnnmmpca', 'bcnnmmpca'};

outputFigName = 'mmPRCurve';
settingNameList = {'(m,m)', '(m,m)+ft', '(m,m_{64}^{pca})', '(m,m_{64}^{pca})+ft'};

modelPathList = {'v2',...
				'bcnn-train-fine-tuned_mm',...
                'bcnn-train-svmtest-mm_one_pca_64_nonrelu',...
                'bcnn-train-fine-tuned_mm_one_pca_64_nonrelu'};
modelNameList = {'bcnnmm', 'bcnnmm', 'bcnnmmpca', 'bcnnmmpca'};


% colorSpec = ['rbkcmg'];
% lineSpec = {'-', '--',  '-.', '-', '--', '-.'};

colorSpec = ['rrbb'];
lineSpec = {'--', '-', '--', '-'};

figure
hold on

for i=1:numel(modelPathList)
    
    load(fullfile('data', modelPathList{i}, dataset, ['result-', modelNameList{i}, '.mat']))
    load(fullfile('data', modelPathList{i}, dataset, ['result-precision-recall.mat']))
    plot(plotRecall, plotPrecision, [lineSpec{i}, colorSpec(i)], 'LineWidth', 1);
    settingNameList{i} = [num2str(100.*test.acc, '%.1f'), ' ', settingNameList{i}];
end

axis([0,1,0,1])

h = legend(settingNameList);
set(h, 'fontSize', 16, 'location', 'SouthWest')
leg_pos = get(h, 'position');
% set(h, 'PlotBoxAspectRatio', [1 1 1]);
set(h, 'position', [leg_pos(1), leg_pos(2), leg_pos(3)*2, leg_pos(4)*1.5]);
% set(h, 'fontSize', 18, 'location', 'SouthWest');
legend boxoff

xlabel('recall', 'fontSize', 24)
ylabel('precision', 'fontSize', 24)
set(gca, 'fontsize', 14)

box on

hold off

export_fig('-transparent', '-eps', '-pdf', fullfile('figOutput', outputFigName))