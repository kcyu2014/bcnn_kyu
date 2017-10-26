function [filters, biases] = mergeBatchNormAfterConv(filters, biases, multipliers, offsets, moments)
% -------------------------------------------------------------------------
% Add for merge BNorm into Conv in l+1.
% 
a = multipliers(:) ./ moments(:,2) ;
b = offsets(:) - moments(:,1) .* a ;
filters = permute(filters, [1,2,4,3]);
sz = size(filters) ;
numFilters = sz(4);
numNewFilters = sz(3);

% Compose new bias
tmp_b = reshape(bsxfun(@times, reshape(filters, [], numFilters), b'), sz) ;
tmp_b = sum(reshape(permute(tmp_b, [1,2,4,3]), [], numNewFilters), 1);
tmp_b = tmp_b';
biases(:) = biases(:) + tmp_b(:);
% Compose new filters. W .* a
filters = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz) ;
filters = permute(filters, [1,2,4,3]);
end