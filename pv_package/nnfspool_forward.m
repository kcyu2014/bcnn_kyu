function upper = nnfspool_forward(opts, lower, upper, masks)
% nnfspool is the function performing the forward pass of free shape pooling

  [M,N,D,L] = size(lower.x);
  if ~exist('masks','var')
    for i = 1: L
      masks{i} = ones(M,N,1,1,'single','gpuArray')/(M*N);
    end
  end
  
  n_masks = sum(cellfun(@(a) size(a,3), masks));
  
  switch opts.method
    case {'o2p_avg_log'}
        type = 'double';  
        x = zeros(1,1,D*D,n_masks,type);      

        % alocate matrices
        U = zeros(M*N,M*N,n_masks,type); % eigenvectors matrix
        S = zeros(M*N,D,n_masks,type); % eigenvalue matrix
        V = zeros(D,D,n_masks,type); 
        X = eval([type '(reshape(gather(lower.x),[M*N,D,L]));']);

        % normalized the covariance
        y = X/(M*N);

        % compute the eigenvectors
        parfor i = 1: L
          [U(:,:,i),S(:,:,i),V(:,:,i)] = svd(y(:,:,i));
        end


        % compute the log
        eyeS = opts.epsilon*eye(size(S,2),type);
        parfor i = 1:L
          x(1,1,:,i) = reshape(V(:,:,i)*diag(log(diag(S(:,:,i)'*S(:,:,i)+eyeS)))*V(:,:,i)',[1,1,D*D]);
        end

        % move the data back to the gpu for the rest of the computation
        upper.x = gpuArray(single(x));

        upper.aux{1} = U;
        upper.aux{2} = S;
        upper.aux{3} = V;

    otherwise
      error('Not supported!');
  end
end