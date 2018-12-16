function K = covSEard_jitter_grad_vshgp(hyp, x, z, K, i)
% jitter version of the derivations of SE kernel wrt parameters
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.

if nargin<2, K = '(D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

jitter = 1e-6;

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance

if nargin>3                                                        % derivatives
  if i<=D                                              % length scale parameters
    if dg
      K = K*0;
    else
      if xeqz
        K = K - sf2*jitter*eye(n);
        K = K.*sq_dist_vshgp(x(:,i)'/ell(i));
      else
        K = K.*sq_dist_vshgp(x(:,i)'/ell(i),z(:,i)'/ell(i));
      end
    end
  elseif i==D+1                                            % magnitude parameter
    K = 2*K;
  else
    error('Unknown hyperparameter')
  end
end