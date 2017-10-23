function lower = nnfspool_backward(opts, lower, upper, masks)
% nnfspool is the function performing the backward pass of free shape pooling


  [M,N,D,L] = size(lower.x);
  
  if ~exist('masks','var')
    for i = 1: L
      masks{i} = ones(M,N,1,1,'single','gpuArray')/(M*N);
    end
  end
  
  switch opts.method
    case {'o2p_avg_log'}
      lower.dzdx = zeros(M,N,D, L,'single','gpuArray');
      upper_dzdx = double(gather(upper.dzdx));
      counter = 1;
      for a=1:L % iterate over images in batch
        for b=1:size(masks{a},3) % iterate over masks in image
          if strcmp(opts.method,'o2p_avg_log')
            S = upper.aux{2}(:,:,counter);
            U = upper.aux{1}(:,:,counter);
            V = upper.aux{3}(:,:,counter);

            diagS = diag(S);
            ind =diagS >(D*eps(max(diagS)));% 
            Dmin = gather(min(find(ind,1,'last'),D));
            S = S(:,ind); V = V(:,ind);
            dLdC = double(reshape(upper_dzdx(1,1,:,counter),[D D])); dLdC = symmetric(dLdC); 
            
            dLdV = 2*dLdC*V*diagLog(S'*S,opts.epsilon); 
            dLdS = 2*S*diagInv(S'*S+opts.epsilon*eye(Dmin))*(V'*dLdC*V);
            if sum(ind) == 1 % diag behaves badly when there is only 1d
              K = 1./(S(1).^2*ones(1,Dmin)-(S(1).^2*ones(1,Dmin))'); K(eye(size(K,1))>0)=0;
            else
              K = 1./(diag(S).^2*ones(1,Dmin)-(diag(S).^2*ones(1,Dmin))'); K(eye(size(K,1))>0)=0;
            end
            if all(diagS==1)
              dzdx = zeros(M*N,D);
            else
              dzdx = U*(2*S*symmetric(K'.*(V'*dLdV))+dDiag(dLdS))*V';
            end
          end
          assert(~any(isnan(dzdx(:))));
          lower.dzdx(:,:,:,a) = lower.dzdx(:,:,:,a) + reshape(dzdx./(M*N),[M N D]); %warning('no normalization');
          counter = counter + 1;
        end
      end
    otherwise
      error('Not supported!');
  end 