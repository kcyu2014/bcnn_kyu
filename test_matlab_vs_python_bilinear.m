function test = test_matlab_vs_python_bilinear(opts, varargin)
% Test the matlab bi-linear model vs true one.
% Implement the gradients in tensorflow like the bilinear model and 
% see the difference.

opts.saveMatDir = 'data/testGradient/';
opts.testLayer = 'gsp';

if(~exist(fullfile(['data', 'testGradient'])))
  mkdir('data/testGradient');
end

[opts, varargin] = vl_argparse(opts, varargin);

% Create random test matrix
switch opts.testLayer
  case 'gsp'
    test = test_gsp_gradient(opts);
  otherwise
    return 

end

end

function res = test_gsp_gradient(opts)
% create random
res.x = rand(32, 32, 128, 256);
res.y = vl_nngsp(res.x);
res.grad_y = rand(size(res.y));
res.grad_x = vl_nngsp(res.x, res.grad_y);

save(fullfile([opts.saveMatDir, 'test_gsp_gradient.mat']), 'res');
end
