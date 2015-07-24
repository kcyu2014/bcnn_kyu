function lse = logsumexp(a, varargin)

nVarIn = numel(varargin);

if(nVarIn==1)
    dim = varargin{1};
    c = max(a, [], dim);
    a_c = bsxfun(@minus, a, c);
    lse = c + log(sum(exp(a_c), dim));
%     lse = lse(:);
else
    
    c = max(a);
    
    lse = c + log(sum(exp(a-c)));
end