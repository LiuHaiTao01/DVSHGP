function model = vshgpCreate(X, y, options, varargin)
%
% Inputs:
%        X       - an n*D matrix representing n training points
%        y       - an n*1 vector representing n training responses
%        options
%               .m - a 2*1 vector wherein m(1) is number of inducing points for f and m(2) is number of inducing points for g
%               .PseudoType - 'random', 'kmeans'
%               .InitialKernelParas - 'GP' - initilize the parameters by running a standard homegenuous GP
%                                     'User' - directly specify the initial parameter values by users
%               .GPnumIter - number of optimization iterations for standard GP if InitialKernelParas = 'GP'
% Outputs:
%        model
%             .D - dimensionality
%             .GPf
%                 .type - 'seard': aytomatic relevence determination (ard) SE kernel
%                 .logtheta - a (D+1)*1 log hyperparameter vector for seard kernel
%                            the first D values represents the log length scales along D dimensions, and is initialized as log((max(X)-min(X))'/2)
%                            the final value represents the log signal variance, and is initialized as 0.5*log(var(y, 1))
%                 .nParams - number of parameters
%                 .s2 - exp(2*logtheta(end)), signal variance
%             .jitter - 10^(-7)*vary
%             .GPg
%                 .type - 'seard': aytomatic relevence determination (ard) SE kernel
%                 .logtheta - a (D+1)*1 log hyperparameter vector for seard kernel
%                             the first D values represents the log length scales along D dimensions, and is initialized as log((max(X)-min(X))'/2)
%                             the final value represents the log signal variance, and is initialized as 0.5*log(var(y, 1)/4)
%                 .mu0 - a scalar to specify the mean of this GP
%                 .nParams - number of parameters
%                 .s2 - exp(2*logtheta(end)), signal variance 
%             .m - a 2*1 vector wherein m(1) is number of inducing points for f and m(2) is number of inducing points for g
%             .n - training size
%             .Pseudo
%                    .type - 'random', 'kmeans'
%                    .merge - 'on': Xm = Xu; 'off': Xm ~= Xu
%                    .Xm - a m(1)*D matrix representing m inducing points for latent function f
%                    .Xu - a m(2)*D matrix representing m inducing points for latent noise function g
%                    .nParams - number of parameters
%             .Variation
%                       %.varg - a [u+u(u+1)/2]*1 vector wherein the first u elements respond to the \mu_u, and the last u(u+1)/2 elements respond to
%                               the \Sigma_u
%                       %.mu - a u*1 vector responding to the mean of joint Gaussian distribution g_u
%                       %.Sigma - a u*u symmetric matrix responding to the covariance of joint Gaussian distribution g_u
%                       .loglambda - an n*1 vector
%                       .diaglambda - an n*1 vector, diagLambda = exp(loglambda)
%                       .nParams - number of parameters
%             .vary - = var(y)
%             .X - an n*D matrix representing n training points
%             .y - an n*1 vector representing n training responses
%             .yy - y'*y;
%
% Haitao Liu (htliu@ntu.edu.sg) 2017/11/27
    
model.GPf.type = 'seard'; % 'seard'
model.GPg.type = 'seard';

[n D] = size(X);
model.n = n;    % traning size
model.D = D;    % dimensionality
model.y = y(:); % ytrain
model.X = X;    % xtrain    

% [Variational paras (g_u); kernel paras (f and g); inducing paras (Xm for f and Xu for g)]
% initial variational parameters
model.Variation.nParams = model.n;
model.Variation.loglambda = log(0.5)*ones(model.Variation.nParams,1); % log to ensure the lambda values are non-negative
model.Variation.diaglambda = exp(model.Variation.loglambda);

% initial kernel parameters for kf and kg
if strcmp(options.InitialKernelParas,'GP')
    logKernelParas = varargin{1};
    model.GPf.logtheta(1:D,1) = logKernelParas.lengthscales_f; model.GPf.logtheta(D+1,1) = logKernelParas.SignalPower_f;
    model.GPg.logtheta(1:D,1) = logKernelParas.lengthscales_g; model.GPg.logtheta(D+1,1) = logKernelParas.SignalPower_g;     
    model.GPg.mu0 = logKernelParas.mu0;
elseif strcmp(options.InitialKernelParas,'User')
    lengthscales_f = log(0.5)*ones(D,1); lengthscales_g = log(0.2)*ones(D,1); 
    sn2 = 1;  
    SignalPower = 1;
    NoisePower  = SignalPower/4;
    model.GPf.logtheta(1:D,1) = lengthscales_f; model.GPf.logtheta(D+1,1) = log(sqrt(SignalPower));
    model.GPg.logtheta(1:D,1) = lengthscales_g; model.GPg.logtheta(D+1,1) = log(sqrt(sn2));     
    model.GPg.mu0 = log(NoisePower);
end
model.GPf.nParams = size(model.GPf.logtheta,1); 
model.GPg.nParams = size(model.GPg.logtheta,1) + 1; 
model.GPf.s2 = exp(2*model.GPf.logtheta(D+1,1));
model.GPg.s2 = exp(2*model.GPg.logtheta(D+1,1));

% initialize inducing parameters of the inducing variables  
model.m = options.m; 
model.Pseudo.type = options.PseudoType;
if strcmp(model.Pseudo.type, 'random')        
    % randomly initialize pseudo inputs from the training inputs
    %perm = randperm(n);
    %model.Pseudo.Xm = X(perm(1:model.m(1)),:);   
    %model.Pseudo.Xu = X(perm(1:model.m(2)),:);

    if model.m(1) < model.n
        %model.Pseudo.Xm = linspace(min(X),max(X),model.m(1))';
        perm = randperm(n);
        model.Pseudo.Xm = X(perm(1:model.m(1)),:);   
    elseif model.m(1) == model.n 
        model.Pseudo.Xm = model.X;
    end
    if model.m(2) < model.n 
        %model.Pseudo.Xu = linspace(min(X),max(X),model.m(2))';
        perm = randperm(n);
        model.Pseudo.Xu = X(perm(1:model.m(2)),:); 
    elseif model.m(2) == model.n
        model.Pseudo.Xu = model.X;
    end

    model.Pseudo.nParams = model.D;
elseif strcmp(model.Pseudo.type, 'kmeans') % preferred
    % inducing points for f
    m = model.m(1);
    if m < model.n
        opts = statset('Display','off');
        [idx,C] = kmeans(X,m,'MaxIter',500,'Options',opts);
        model.Pseudo.Xm = findNeighbors(X,idx,C);
    elseif m == model.n
        model.Pseudo.Xm = model.X;
    elseif m > model.n
        X1 = model.X;
        perm = randperm(n);
        X2 = 0.5*(X(1:m-model.n,:) + X(perm(1:m-model.n),:));
        model.Pseudo.Xm = [X1;X2];
    end
    
    % inducing points for g
    u = model.m(2);
    if u < model.n
        opts = statset('Display','off');
        [idx,C] = kmeans(X,u,'MaxIter',500,'Options',opts);
        model.Pseudo.Xu = findNeighbors(X,idx,C);
    elseif u == model.n
        model.Pseudo.Xu = model.X;
    elseif u > model.n
        X1 = model.X;
        perm = randperm(n);
        X2 = 0.5*(X(1:u-model.n,:) + X(perm(1:u-model.n),:));
        model.Pseudo.Xu = [X1;X2];
    end

    model.Pseudo.nParams = model.D;
end 

model.jitter = 1e-6; 

end % end main function



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Cc = findNeighbors(x,idx,C) 
%
[n,d] = size(C) ;
Cc = zeros(n,d) ;
for i = 1:n 
    x_i = x(idx==i,:) ;
    [n_i,d] = size(x_i) ;
    dists = sum((x_i - repmat(C(i,:),n_i,1)).^2,2) ;
    [a,b] = find(dists == min(dists)) ;
    Cc(i,:) = x_i(a(1),:) ;
end

end
