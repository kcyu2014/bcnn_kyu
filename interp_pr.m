function interp_precision = interp_pr(precision, recall, interp_recall) 

%interp_recall = 0:0.05:1;

inRange = find(interp_recall>min(recall) & interp_recall<max(recall));

xn = numel(interp_recall);


M = cat(2, recall', precision');
sortrows(M, [1, -2]);
[~, ia, ~] = unique(M(:,1));
M = M(ia,:);

precision = M(:,2)';
recall = M(:,1)';

x1 = zeros(xn, 1);
x2 = zeros(xn, 1);
y1 = zeros(xn, 1);
y2 = zeros(xn, 1);
for i=1:numel(inRange)
    k = inRange(i);
    temp = recall - interp_recall(k);
    idx = find(temp>0, 1);
    x1(k) = recall(idx-1);
    x2(k) = recall(idx);
    y1(k) = precision(idx-1);
    y2(k) = precision(idx);
end

% interp_precision = ((y2-y1).*(x2-x1)./(x2-x1))+y1;
interp_precision = ((y2-y1).*(interp_recall'-x1)./(x2-x1))+y1;
interp_precision(1) = precision(1);
interp_precision(end) = precision(find(recall==1,1));
