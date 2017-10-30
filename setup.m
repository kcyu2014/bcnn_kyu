% setup vlfeat
run ./vlfeat/toolbox/vl_setup

% setup matconvnet
run ./matconvnet/matlab/vl_setupnn
addpath ./matconvnet/examples/

% add custom pv package
addpath ./pv_package/
addpath ./utils/
addpath ./myutils/

clear mex ;
