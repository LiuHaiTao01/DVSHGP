function [mu,s2,muf,s2f,mug,s2g,t_predict] = DistributedVSHGP_fast_predict(xt,vshgp_model,models,options)
% Distributed VSHGP for prediction
% Inputs:
%        xt - a nt x d matrix representing nt test points
%        vshgp_model - a model structure wherein vshgp_model{i} contains the configurations for vshgp
%        models - a model structure wherein model{i} contains the configurations for the ith expert
%        options - configurations for distributed vshgp
%              .m - a 2 x 1 vector representing the numbers of inducing variables for f and g of each expert
%              .PseudoType - type of initial inducing points including 'random', 'kmeans'
%              .InitialKernelParas - initilize kernel paras by standard GP or mannually, 'GP', 'User'
%              .flag - mode of optimizaing hyperparameters
%                     = 1: optimize variational + f&g kernel + inducing paras
%                     = 2: optimize variational + f&g kernel paras
%                     = 3: optimize variational + g kernel paras
%                     = 4: optimize variational paras
%              .M - number of experts
%              .numIter - optimization setting
%                         negative: number of function evaluations; positive: number of line searches
%              .partitionCriterion - type of data partition;
%              .criterion - aggregation criterion, including 'PoE', 'GPoE', 'BCM' and 'RBCM'
% Outputs:
%        mu - aggregated prediction means
%        s2 - aggregated prediction variances
%        muf - aggregated prediction means for f
%        s2f - aggregated prediction variances for f
%        mug - aggregated prediction means for g
%        s2g - aggregated prediction variances for g
%        t_predict - predicting time
%
% H.T. Liu 2018/02/04 (htliu@ntu.edu.sg)

% configurations
criterion = options.criterion;
M = options.M;

% normalization of test points xt
[nt, d] = size(xt) ;
if strcmp(options.Xnorm,'Y'); xt = (xt - repmat(models{1}.x_train_mean,nt,1)) ./ (repmat(models{1}.x_train_std,nt,1)); end 

t1 = clock ;
% pre-extract predictions of each expert
for i = 1:M 
    [out1,out2] = vshgp_fast(models{i}.hyp, vshgp_model{i}, 1, xt); % flag = 1 to update all paras for prediction
    mu_experts{i} = out1.mustar;  muf_experts{i} = out1.mu_fstar;  mug_experts{i} = out1.mu_gstar;
    s2_experts{i} = out2.varstar; s2f_experts{i} = out2.var_fstar; s2g_experts{i} = out2.var_gstar;
end

% use an aggregation criterion to combine predictions from experts
mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
switch criterion 
    case 'PoE' % product of GP experts
        % for f
        muf = zeros(nt,1) ; s2f = zeros(nt,1) ;
        for i = 1:M
            s2f = s2f + 1./s2f_experts{i} ; 
        end
        s2f = 1./s2f ;   
        for i = 1:M 
            muf = muf + s2f.*(muf_experts{i}./s2f_experts{i}) ;
        end

        % for g
        mug = zeros(nt,1) ; s2g = zeros(nt,1) ;
        for i = 1:M
            s2g = s2g + 1./s2g_experts{i} ; 
        end
        s2g = 1./s2g ;   
        for i = 1:M 
            mug = mug + s2g.*(mug_experts{i}./s2g_experts{i}) ;
        end

        % combine predictions from GP experts for f+g
        mu = muf;
        s2 = s2f + exp(mug + 0.5*s2g);
    case 'GPoE' % generalized product of GP experts using beta_i = 1/M  
        % for f
        muf = zeros(nt,1) ; s2f = zeros(nt,1) ;
        for i = 1:M
            betaf{i} = 1/M*ones(length(s2f_experts{i}),1) ;
            s2f = s2f + betaf{i}./s2f_experts{i} ; 
        end
        s2f = 1./s2f ;   
        for i = 1:M 
            muf = muf + s2f.*(betaf{i}.*muf_experts{i}./s2f_experts{i}) ;
        end

        % for g
        mug = zeros(nt,1) ; s2g = zeros(nt,1) ;
        for i = 1:M
            betag{i} = 1/M*ones(length(s2g_experts{i}),1) ;
            s2g = s2g + betag{i}./s2g_experts{i} ; 
        end
        s2g = 1./s2g ;   
        for i = 1:M 
            mug = mug + s2g.*(betag{i}.*mug_experts{i}./s2g_experts{i}) ;
        end

        % combine predictions from GP experts for f+g
        mu = muf;
        s2 = s2f + exp(mug + 0.5*s2g);
    case 'BCM' % Bayesian committee machine    
        % for f
        muf = zeros(nt,1) ; s2f = zeros(nt,1) ;
        hypf = models{1}.hyp_kernel(1:d+1); sf2_f = exp(2*hypf(d+1));
        kss_f = sf2_f*ones(nt,1);         
        for i = 1:M
            s2f = s2f + 1./s2f_experts{i} ; 
        end
        s2f = 1./(s2f + (1-M)./kss_f) ;   
        for i = 1:M 
            muf = muf + s2f.*(muf_experts{i}./s2f_experts{i}) ;
        end

        % for g
        mug = zeros(nt,1) ; s2g = zeros(nt,1) ;
        hypg = models{1}.hyp_kernel(d+2:end-1); mug_prior = models{1}.hyp_kernel(end); sf2_g = exp(2*hypg(d+1));    
        kss_g = sf2_g*ones(nt,1);      
        for i = 1:M
            s2g = s2g + 1./s2g_experts{i} ; 
        end
        s2g = 1./(s2g + (1-M)./kss_g) ;   
        for i = 1:M 
            mug = mug + s2g.*(mug_experts{i}./s2g_experts{i}) ;
        end
        mug = mug - (M-1)*mug_prior*(1./kss_g).*s2g;

        % combine predictions from GP experts for f+g
        mu = muf;
        s2 = s2f + exp(mug + 0.5*s2g);
    case 'RBCM' % robust Bayesian committee machine
        % for f
        muf = zeros(nt,1) ; s2f = zeros(nt,1) ;
        hypf = models{1}.hyp_kernel(1:d+1); sf2_f = exp(2*hypf(d+1));
        kss_f = sf2_f*ones(nt,1);
        betaf_total = zeros(nt,1) ;
        for i = 1:M
            betaf{i} = 0.5*(log(kss_f) - log(s2f_experts{i})) ;
            betaf_total = betaf_total + betaf{i} ;

            s2f = s2f + betaf{i}./s2f_experts{i} ; 
        end
        s2f = 1./(s2f + (1-betaf_total)./kss_f) ;   
        for i = 1:M 
            muf = muf + s2f.*(betaf{i}.*muf_experts{i}./s2f_experts{i}) ;
        end

        % for g
        mug = zeros(nt,1) ; s2g = zeros(nt,1) ;
        hypg = models{1}.hyp_kernel(d+2:end-1); mug_prior = models{1}.hyp_kernel(end); sf2_g = exp(2*hypg(d+1));
        kss_g = sf2_g*ones(nt,1); 
        betag_total = zeros(nt,1) ;
        for i = 1:M
            betag{i} = 0.5*(log(kss_g) - log(s2g_experts{i})) ;
            betag_total = betag_total + betag{i} ;

            s2g = s2g + betag{i}./s2g_experts{i} ; 
        end
        s2g = 1./(s2g + (1-betag_total)./kss_g) ;   
        for i = 1:M 
            mug = mug + s2g.*(betag{i}.*mug_experts{i}./s2g_experts{i}) ;
        end
        mug = mug - mug_prior*(betag_total-1).*(1./kss_g).*s2g;

        % combine predictions from GP experts for f+g
        mu = muf;
        s2 = s2f + exp(mug + 0.5*s2g);
    otherwise
        error('No such aggregation model.') ;
end

t2 = clock ;
t_predict = etime(t2,t1) ;

% recover predictions if needed
if strcmp(options.Ynorm,'Y')
    mu  = mu*models{1}.y_train_std + models{1}.y_train_mean;
    muf = muf*models{1}.y_train_std + models{1}.y_train_mean;
    s2  = s2*(models{1}.y_train_std)^2;
    s2f = s2f*(models{1}.y_train_std)^2;
end

end
