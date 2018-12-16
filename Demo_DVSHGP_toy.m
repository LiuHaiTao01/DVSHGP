% demo of distributed VSHGP (distributed + sparse) for 1D toy example
clear all 
rng(100) % to ensure reproducibility

% --------------------------------------
% --------------Load data---------------
% --------------------------------------
display('Loading data ...')
data = 'sinc';

if strcmp(data,'sinc')
    fx = @(x) sinc(x); % true function
    sx = @(x) 0.05+(1./(1+exp(-0.2*x))).*(1+sin(2*x))*0.2; % true output noise function
    
    n = 500;   % number of samples
    X  = linspace(-10,10,n)'; % input
    Y = fx(X)+randn(size(X)).*sx(X);  % add noise to output
    
    Xs = linspace(-10,10,1000)'; % input
    Ys = fx(Xs); Ys_noiseVar = (sx(Xs)).^2;
end

% --------------------------------------
% -----Configurations-------------------
% --------------------------------------
options.Xnorm = 'N'; options.Ynorm = 'N';
options.m = [10;10]; % number of inducing points for f and g of each expert, respectively
options.M = 5; % number of experts
if n < options.m(1) || n < options.m(2); warning('The inducing size exceeds the expert training size.'); end
options.numIter = [-100]; % number of opt iterations, the length of options.numIter is consistent to that of options.flag
                         % negative: number of function evaluations; positive: number of line searches
options.flag = [1]; % if length(options.flag), we perform a one-stage inference; otherwise, a multi-stage inference
                    % flag = 1, paras=[variational paras; kernel paras; inducing paras]
                    % flag = 2, paras=[variational paras; kernel paras]
                    % flag = 3, paras=[variational paras; g kernel paras]
                    % flag = 4, paras=[variational paras]
options.PseudoType = 'kmeans'; % type of initial inducing points including 'random', 'kmeans'
options.InitialKernelParas = 'User'; % initilize kernel paras by standard GP or mannually, 'GP', 'User'
options.partitionCriterion = 'pureKmeans';
options.criterion = 'RBCM'; % aggregation

% --------------------------------------
% -----------Distributed VSHGP model----
% --------------------------------------
% train the model
display('Running distributed VSHGP ...')
[experts,vshgp_model,t_train] = DistributedVSHGP_fast_train(X, Y, options);

% prediction in test data
[mustar,varstar,mu_fstar,var_fstar,mu_gstar,var_gstar,t_predict] = DistributedVSHGP_fast_predict(Xs,vshgp_model,experts,options);


% --------------------------------------
% ------------------Plot----------------
% --------------------------------------
figure('position',[530         106        1137         653])
m = options.m(1); u = options.m(2); % number of inducing points for f and g
colors = [153,153,153; 96,157,202; 56,194,93; 255,150,65; 255,91,78]/255; % colors for 5 experts
subplot(2,2,1) % plot of dvshgp
% training data
if strcmp(options.Xnorm,'N')
    factor.meanX = 0; factor.stdX = 1;
elseif strcmp(options.Xnorm,'Y')
    factor.meanX = mean(X); factor.stdX = std(X);
end
if strcmp(options.Ynorm,'N')
    factor.meanY = 0; factor.stdY = 1;
elseif strcmp(options.Ynorm,'Y')
    factor.meanY = mean(Y); factor.stdY = std(Y);
end
for i = 1:5
    Xnorm_i = experts{i}.X_norm; Ynorm_i = experts{i}.Y_norm;
    ni = length(Ynorm_i );
    Xi = Xnorm_i.*repmat(factor.stdX,ni,1) + repmat(factor.meanX,ni,1);
    Yi = Ynorm_i*factor.stdY+factor.meanY;

    Xm_i = experts{i}.Xm; Xu_i = experts{i}.Xu;
    Xm_i = Xm_i.*repmat(factor.stdX,m,1) + repmat(factor.meanX,m,1);
    Xu_i = Xu_i.*repmat(factor.stdX,u,1) + repmat(factor.meanX,u,1);

    % training subsets
    plot(Xi,Yi,'+','LineWidth',1.0,'color',colors(i,:)); hold on;
    % inducing subsets
    plot(Xm_i,max(Y)*ones(m,1),'o','markersize',8,'LineWidth',1.0,'color',colors(i,:)); hold on;
    plot(Xu_i,min(Y)*ones(u,1),'s','markersize',8,'LineWidth',1.0,'color',colors(i,:)); hold on;
end
% dvshgp
plot(Xs,mustar,'r-','linewidth',2.0); hold on;
plot(Xs,mustar + 2*sqrt(varstar),'k-','linewidth',2.0); hold on;
plot(Xs,mustar - 2*sqrt(varstar),'k-','linewidth',2.0); hold on;
axis([-10 10 -1 1.7]) ;
set(gcf,'color','w');
grid on;
set(gca,'fontsize',16);
xlabel('$x$','interpreter', 'latex', 'fontsize', 24); 
ylabel('$y$','interpreter', 'latex', 'fontsize', 24, 'rotation', 0);
y_range = get(gca,'ylim');
text(-13.5,y_range(2),'(a)','FontSize',20);

%figure('position',[236    85   759   496])
subplot(2,2,2) % plot of underlying function f
plot(X,Y,'+','LineWidth',1.0,'color',[179,179,179]/256); hold on;
plot(Xs,mu_fstar,'r-','linewidth',2.0); hold on;
plot(Xs,mu_fstar + 2*sqrt(var_fstar),'k-','linewidth',2.0); hold on;
plot(Xs,mu_fstar - 2*sqrt(var_fstar),'k-','linewidth',2.0); hold on;
axis([-10 10 -1 1.7]) ;
set(gcf,'color','w');
grid on;
set(gca,'fontsize',16);
xlabel('$x$','interpreter', 'latex', 'fontsize', 24); 
ylabel('$f$','interpreter', 'latex', 'fontsize', 24, 'rotation', 0);
y_range = get(gca,'ylim');
text(-13.5,y_range(2),'(b)','FontSize',20);

%figure('position',[236    85   759   496])
subplot(2,2,3) % plot of noise function f
plot(Xs,mu_gstar,'r-','linewidth',2.0); hold on;
plot(Xs,mu_gstar + 2*sqrt(var_gstar),'k-','linewidth',2.0); hold on;
plot(Xs,mu_gstar - 2*sqrt(var_gstar),'k-','linewidth',2.0); hold on;
%axis([-10 10 -1.5 1.0]) ;
set(gcf,'color','w');
grid on;
set(gca,'fontsize',16);
xlabel('$x$','interpreter', 'latex', 'fontsize', 24); 
ylabel('$g$','interpreter', 'latex', 'fontsize', 24, 'rotation', 0);
y_range = get(gca,'ylim');
text(-13.5,0,'(c)','FontSize',20);
