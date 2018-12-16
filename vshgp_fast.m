function [out1, out2] = vshgp_fast(paras, model, flag, Xtest)
% Fast implementation of VSHGP
% Inputs:
%        paras - when flag = 1, update [variational paras; f&g kernel paras; inducing paras]
%                when flag = 2, update [variational paras; f&g kernel paras]
%                when flag = 3, update [variational paras; g kernel paras]
%                when flag = 4, update [variational paras]
%        model
%             .D - dimensionality
%             .GPf
%                 .type - 'seard': aytomatic relevence determination (ard) SE kernel
%                 .logtheta - a (D+1)*1 log hyperparameter vector for seard kernel
%                            the first D values represents the log length scales along D dimensions, and is initialized as log((max(X)-min(X))'/2)
%                            the final value represents the log signal variance, and is initialized as 0.5*log(var(y, 1))
%                 .nParams - number of parameters
%                 .constDiag - = 1 indicates that the Knn covariance matrix has the same diagonal elements 
%             .jitter - 10^(-7)*vary
%             .GPg
%                 .type - 'seard': aytomatic relevence determination (ard) SE kernel
%                 .logtheta - a (D+1)*1 log hyperparameter vector for seard kernel
%                             the first D values represents the log length scales along D dimensions, and is initialized as log((max(X)-min(X))'/2)
%                             the final value represents the log signal variance, and is initialized as 0.5*log(var(y, 1)/4)
%                 .nParams - number of parameters
%             .m - a 2*1 vector wherein m(1) is number of inducing points for f and m(2) is number of inducing points for g
%             .n - training size
%             .Pseudo
%                    .type - 'random', 'kmeans'
%                    .merge - 'on': Xm = Xu; 'off': Xm ~= Xu
%                    .Xm - a m(1)*D matrix representing m inducing points for latent function f
%                    .Xu - a m(2)*D matrix representing m inducing points for latent noise function g
%                    .nParams - number of parameters
%             .Variation
%                       .loglambda - an n*1 vector to determine the variational distribution p(g_u|mu_u, Sigma_u)
%                       .nParams - number of parameters, = n
%             .vary - = var(y)
%             .X - an n*D matrix representing n training points
%             .y - an n*1 vector representing n training responses
%             .yy - y'*y;
%        flag - = 1: extract variational paras, f&g kernel paras and inducing paras 
%               = 2: extract variational paras and f&g kernel paras
%               = 3: extract variational paras and g kernel paras
%               = 4: extract only variational paras
%
% Outputs:
%        out1 - training mode (nargin = 3): a scalar representing the negative variational bound
%               predicting mode (nargin = 4): a structure where out1.
%                                                                   mustar is the prediction mean of vshgp at Xtest
%                                                                   mufstar is the prediction mean of f at Xtest
%                                                                   mugstar is the prediction mean of g at Xtest
%        out2 - training mode (nargin = 3): a vector represents the derivatives of F wrt paras
%               predicting mode (nargin = 4): a structure where out2.
%                                                                   varstar is the prediction variance of vshgp at Xtest
%                                                                   varfstar is the prediction variance of f at Xtest
%                                                                   vargstar is the prediction variance of g at Xtest
%
% Haitao Liu (htliu@ntu.edu.sg) 2018/01/22

%----------------------------
% Evidence lower bound (ELBO)
%----------------------------
% Place paras to the model structure
model = returnOptimizedParams_vshgp(model, paras, flag);

m = model.m(1); % number of inducing points for f_m
u = model.m(2); % number of inducing points for g_u
n = size(model.X,1); % training size

mu0 = model.GPg.mu0; % prior mean for g, this parameter is needed to enhance the flexibility of g

% Covariance matrics calculated by a *jitter* SE kernel    
model.diagKnn_f = covSEard_jitter_vshgp(model.GPf.logtheta, model.X, 'diag');     % n x 1, avoid using n x n Knn_f!
model.diagKnn_g = covSEard_jitter_vshgp(model.GPg.logtheta, model.X, 'diag');     % n x 1
model.Kmm_f = covSEard_jitter_vshgp(model.GPf.logtheta, model.Pseudo.Xm);         % m x m: kf(Xm,Xm)
model.Kuu_g = covSEard_jitter_vshgp(model.GPg.logtheta, model.Pseudo.Xu);         % u x u: kg(Xu,Xu)
model.Knm_f = covSEard_jitter_vshgp(model.GPf.logtheta, model.X, model.Pseudo.Xm);% n x m: kf(X, Xm)
model.Knu_g = covSEard_jitter_vshgp(model.GPg.logtheta, model.X, model.Pseudo.Xu);% n x u: kg(X, Xu)

% Cholesky decompositions
Lmm_f = chol(model.Kmm_f)';                    % m x m, a lower triangular matrix
invLmm_f = Lmm_f\eye(m);                       % m x m, (L_mm^f)^{-1}                            ---- O(m^3)
invKmm_f = Lmm_f'\(Lmm_f\eye(m));              % m x m, K_mm^f = (L_mm^f^{-T})*L_mm^f^{-1}, equal to Lmm_f'\(Lmm_f\eye(m))  ---- O(m^3)
Qnn_f_half = model.Knm_f*invLmm_f';            % n x m, K_nm*L_mm_f^{-T}                     ---- O(n m^2)  !!!expensive!!! 
diagQnn_f = diagAB(Qnn_f_half,Qnn_f_half');    % n x 1
Onm_f = model.Knm_f*invLmm_f'*invLmm_f;        % n x m, Omega_nm^f = K_nm^f*(K_mm^f)^{-1} 

Luu_g = chol(model.Kuu_g)';                    % u x u, L_uu^g is a lower triangular           ---- O(u^3)
invLuu_g = Luu_g\eye(u);                       % u x u, (L_uu^g)^{-T}                            ---- O(m^3)
invKuu_g = Luu_g'\(Luu_g\eye(u));              % u x u, K_uu^g = (L_uu^g)^{-T}*L_uu^g^{-1}     ---- O(m^3)
Qnn_g_half = model.Knu_g*invLuu_g';            % n x u, K_nu*L_uu_g^{-T}                     ---- O(n m^2)  !!!expensive!!! 
diagQnn_g = diagAB(Qnn_g_half,Qnn_g_half');    % n x 1
Onu_g = model.Knu_g*invLuu_g'*invLuu_g;        % n x u, Omega_nu^g = K_nu^g*(K_uu^g)^{-1}

% Variational parameters
diaglambda     = exp(model.Variation.loglambda); % n x 1, to save memory
diagslambda    = diaglambda.^0.5;                % n x 1
diagInvlambda  = 1./diaglambda;                  % n x 1
diagsInvlambda = diagInvlambda.^0.5;             % n x 1
% mu_u and Sigma_u
mu_u = model.Knu_g'*(diaglambda - 0.5) + mu0*ones(u,1);  % u x 1
% -->> **a stable cholesky decomposition** <<--
% -->>      **(Kmm + Kmn*Knm)^{-1}**       <<--
% (K_uu^g + K_{un}^g*Lambda_{nn}*K_{nu}^g)^{-1}
%    = (L_uu^g)^{-T}*(I + (L_uu^g)^{-1}*K_{mn}^g*K_{nm}^g*(L_uu^g)^{-T})^{-1}*(L_uu^g)^{-1}
%    = (L_uu^g)^{-T}*AA*(L_uu^g)^{-1} 
%AA = eye(u) + invLuu_g*(model.Knu_g'*diag(diaglambda)*model.Knu_g)*invLuu_g';
temp = invLuu_g*(model.Knu_g'.*repmat(diagslambda',u,1));      % u x n
AA = eye(u) + temp*temp';                                      % u x u
invL_AA = chol(AA)'\eye(u);                                    % u x u
invL_KLambda = invLuu_g'*invL_AA';                             % u x u, an upper triangular
invKLambda = invL_KLambda*invL_KLambda';                       % u x u
temp = model.Kuu_g*invL_KLambda;                               % u x u
Sigma_u = temp*temp';                                          % u x u

s2 = abs(mean(diag(Sigma_u)));                                 % for stable chol decomposition
L_Sigma_u = chol(Sigma_u + model.jitter*s2*eye(u))';           % 
invL_Sigma_u = L_Sigma_u\eye(u);                               % u x u
invSigma_u = L_Sigma_u'\(L_Sigma_u\eye(u));                    % u x u


% Calculation of mu_g and Sigma_g
% \int p(g|g_u) q(g_u) dg_u = N(g|mu_g, Sigma_g)
mu_g = Qnn_g_half*(Qnn_g_half'*(diaglambda - 0.5)) + mu0*ones(n,1); % n x 1
temp = model.Knu_g*invL_KLambda;                                    % n x u
diagSigma_g = model.diagKnn_g - diagQnn_g + diagAB(temp,temp');     % n x 1

% Calculation of R_g
% R_g is a diagonal matrix with (R_g)_ii = exp((mu_g)_i - 0.5(Sigma_g)_ii)
diagR_g    = exp(mu_g - 0.5*diagSigma_g);   % n x 1, the diagonal elements of R_g, R_g = diag(diagR_g)
diagsR_g   = diagR_g.^0.5;                  % n x 1
diagInvR_g = 1./diagR_g;                    % n x 1, the diagonal elements of R_g, invR_g = diag(diagInvR_g)
diagsInvR_g = diagInvR_g.^0.5;              % n x 1

% precalculations
invRgKnmf = repmat(diagInvR_g,1,m).*model.Knm_f; % n x m 

% Calculation of K_{Lambda}^{-1}
BB = eye(m) + invLmm_f*(model.Knm_f'*invRgKnmf)*invLmm_f';% m x m
L_BB = chol(BB)';                                         % m x m, a lower triangular
invL_BB = L_BB\eye(m);                               % m x m, a lower triangular
L_KR = Lmm_f*L_BB;                                   % m x m, a lower triangular
invL_KR = invLmm_f'*invL_BB';                             % m x m, an upper triangular
invKR = invL_KR*invL_KR';                                 % m x m,

%
invRgKnmfinvKR_half = invRgKnmf*invL_KR; % n x m

% two modes: training + prediction
if nargin == 3 % training mode
    % MV bound F = F1 + F2 + F3 + F4
    % F1 = logN(y|0, Q_nn^f + R_g)
    term1 = sum(log(diag(Lmm_f))) - sum(log(diag(L_KR))) - sum(log(diagsR_g)); % scalar, term1 = -0.5*log|Qnn_f + R_g|
    term2 = -0.5*(model.y').^2*diagInvR_g;                                     % scalar
    %A0    = (model.y.*diagInvR_g)'*model.Knm_f*invL_KR;                        % 1 x m
    A0 = model.y'*invRgKnmfinvKR_half;
    term3 = 0.5*(A0*A0');                                                      % term2 + term3 = -0.5*y^T (Qnn_f + R_g)^{-1} y 
    F1 = -0.5*model.n*log(2*pi) + term1 + term2 + term3;
    % F2 = -0.25Tr(Sigma_g)
    F2 = -0.25*sum(diagSigma_g);
    % F3 = -0.5*Tr(R_g^{-1}*(K_nn^f - Q_nn^f))
    F3 = -0.5*sum((model.diagKnn_f - diagQnn_f).*diagInvR_g);
    % F4 = -KL{N(g|mu_u, Sigma_u) || N(g_u|0, K_uu^g)}
    temp = invLuu_g*(mu_u - mu0);     % u x 1
    invL1L0 = invLuu_g*L_Sigma_u;     % u x u
    F4 = -(sum(log(diag(Luu_g))) - sum(log(diag(L_Sigma_u)))) + 0.5*u - 0.5*sum(sum(invL1L0.*invL1L0)) - 0.5*temp'*temp; 
    
    F = F1 + F2 + F3 + F4;
    out1 = -F; % negative ELBO

    clear Qnn_f_half Lmm_f diagQnn_g invLmm_f A0 invL1L0
    
    
    
    %--------------------------------
    % Derivatives of F wrt parameters
    %--------------------------------
    % precomputations for the derivatives
    temp = invRgKnmfinvKR_half;
    diag_invQnnfPLUSRg = diagInvR_g - diagAB(temp,temp');     % n x 1
    beta_n = diagInvR_g.*model.y - temp*(temp'*model.y);      % n x 1
    diaglambda_a = (beta_n.^2 - diag_invQnnfPLUSRg).*diagR_g; % n x 1                     
    diaglambda_b = (model.diagKnn_f - diagQnn_f).*diagInvR_g; % n x 1
    diagAandB = diaglambda_a + diaglambda_b;                  % n x 1 

    clear diag_invQnnfPLUSRg diagQnn_f
    
    out2 = zeros(model.Variation.nParams + model.GPf.nParams + model.GPg.nParams + m*model.Pseudo.nParams + u*model.Pseudo.nParams,1);
    if length(out2) ~= length(paras); error('Wrong number of parameters.'); end

    if flag == 1 % Derivatives wrt variational + kernel f,g + inducing parameters
        %------------------------------------------------
        % Derivatives of F wrt variational parameters
        %------------------------------------------------
        temp = model.Knu_g*invL_KLambda;                      % n x u
        coef = 0.25*diagAandB + 0.25;                         % n x 1
        temp2 = ((temp'.*repmat(coef',u,1))*temp)*temp';      % u x n
        term1 = diagAB(temp,temp2);                           % n x 1

        Bun = model.Kuu_g*invKLambda*model.Knu_g';            % u x n
        term2 = -0.5*diagAB(Bun'*(invSigma_u - invKuu_g),Bun);% n x 1 
    
        DFDloglambda = 0.5*Qnn_g_half*(Qnn_g_half'*diagAandB) + term1 + term2 - mu_g + mu0; % n x 1
        DFDloglambda = DFDloglambda.*diaglambda;                                            % n x 1

        clear coef Qnn_g_half

        %-----------------------------------------
        % Derivatives of F wrt f kernel parameters
        %-----------------------------------------
        DFDlogthetaf = zeros(model.GPf.nParams,1);
        jitter_SE = 1e-6; % used in covSEard_jitter_vshgp.m
        ell_f = exp(model.GPf.logtheta(1:model.D)); sf2_f = exp(2*model.GPf.logtheta(model.D+1)); % D x 1

        % preconputations
        A1 = Onm_f'*beta_n;                                 % m x 1
        A2 = invRgKnmfinvKR_half;
        A3 = Onm_f'*A2*A2';                                 % m x n
        A4 = A1*beta_n';                                    % m x n
        A5 = A2'*Onm_f;                                     % m x m

        DdiagKnn_fDsf2_f = 2*model.diagKnn_f; % diagKnn_f wrt ell is zero <<- SE ard kernel

        % -->> avoid for loop to improve computing efficiency <<--
        % suppose A is a n x m matrix, and B is a m x n matrix, then
        % trace(A*B) = sum(sum(A.*B,2)) = reshape(A',1,n*m)*reshape(B,n*m,1)
        % precalculate Bs
        coef_Knm_f = reshape(0.5*(A4 + A3), m*n, 1);            % (m*n) x 1
        coef_Kmn_f = reshape(0.5*A4' + 0.5*A2*A5, m*n, 1);      % (m*n) x 1
        coef_Kmm_f = reshape(-0.5*A1*A1' - 0.5*A5'*A5, m*m, 1); % (m*m) x 1
      
        % precalculate As
        for i = 1:model.GPf.nParams % D + 1
            DKnm_fDtheta_i = covSEard_jitter_grad_vshgp(model.GPf.logtheta, model.X, model.Pseudo.Xm, model.Knm_f, i); % n x m
            DKmm_fDtheta_i = covSEard_jitter_grad_vshgp(model.GPf.logtheta, model.Pseudo.Xm, [], model.Kmm_f, i);      % m x m
            DFDlogthetaf(i) = reshape(DKnm_fDtheta_i', 1, n*m)*coef_Knm_f ...
                            + reshape(DKnm_fDtheta_i,  1, m*n)*coef_Kmn_f ...
                            + reshape(DKmm_fDtheta_i', 1, m*m)*coef_Kmm_f;
        end

        DFDlogthetaf(end) = DFDlogthetaf(end) - 0.5*(diagInvR_g'*DdiagKnn_fDsf2_f); % DdiagKnn_fDsf2_f is only available for sf2_f

        %-------------------------------------------------
        % Derivatives of F wrt g kernel parameters and mu0
        %-------------------------------------------------  
        % WRT kernel paras
        DFDlogthetag = zeros(model.GPg.nParams,1);
        ell_g = exp(model.GPg.logtheta(1:model.D)); sf2_g = exp(2*model.GPg.logtheta(model.D+1));

        % precomputations
        A1 = invKLambda*model.Knu_g';                          % u x n
        A3 = Onu_g'*(diaglambda - 0.5);                        % u x 1
        A5 = (mu_u - mu0)'*invKuu_g;                           % 1 x u
        A6 = (invKLambda*model.Kuu_g)*(invSigma_u - invKuu_g); % u x u
        A6A2T = A6*(invKLambda*model.Kuu_g)';                  % u x u
        A7 = model.Knu_g.*repmat(diaglambda,1,u);              % n x u
        A8 = invKuu_g - invKuu_g*(L_Sigma_u*L_Sigma_u')*invKuu_g ...
           - A6 - A6' + A6A2T;                                 % u x u
        A9 = A7*A6A2T;                                         % n x u
        A10 = A6A2T*A7';                                       % u x n

        DdiagKnn_gDsf2_g = 2*model.diagKnn_g; % diagKnn_f wrt ell is zero <<- SE ard kernel

        % -->> avoid for loop to improve computing efficiency <<--
        % suppose A is a n x m matrix, and B is a m x n matrix, then
        % trace(A*B) = sum(sum(A.*B,2)) = reshape(A',1,n*m)*reshape(B,n*m,1)
        % precalculate Bs
        coef_Knu_g = 0.5*A3*diagAandB' ...
                   - 0.25*(2*A1.*repmat(diagAandB'+1,u,1) - (A1.*repmat(diagAandB'+1,u,1))*A1'*A7') ...
                   - 0.5*A10;                    % u x n
        coef_Knu_g = reshape(coef_Knu_g,n*u,1);  % n*u x 1 = n x u
        coef_Kun_g = 0.5*(diaglambda - 0.5)*(diagAandB'*Onu_g) ...
                   - 0.25*(-2*Onu_g.*repmat(diagAandB+1,1,u) - A7*((A1.*repmat(diagAandB'+1,u,1))*A1')) ...
                   - 0.5*A9 ...
                   - 0.5*2*(diaglambda - 0.5)*A5;% n x u
        coef_Kun_g = reshape(coef_Kun_g,n*u,1);  % n*u x 1 = n x u
        coef_Kuu_g = -0.5*A3*diagAandB'*Onu_g ...
                   -0.25*((Onu_g'.*repmat(diagAandB'+1,u,1))*Onu_g - (A1.*repmat(diagAandB'+1,u,1))*A1') ...
                   - 0.5*A8 ...
                   + 0.5*(A5'*A5);               % u x u
        coef_Kuu_g = reshape(coef_Kuu_g,u*u,1);  % u*u x 1 = u x u

        % precalculate As
        for i = 1:model.GPg.nParams - 1 % D + 1
            DKnu_gDtheta_i = covSEard_jitter_grad_vshgp(model.GPg.logtheta, model.X, model.Pseudo.Xu, model.Knu_g, i); % n x m
            DKuu_gDtheta_i = covSEard_jitter_grad_vshgp(model.GPg.logtheta, model.Pseudo.Xu, [], model.Kuu_g, i);      % m x m
            DFDlogthetag(i) = reshape(DKnu_gDtheta_i', 1, n*u)*coef_Knu_g ...
                            + reshape(DKnu_gDtheta_i,  1, u*n)*coef_Kun_g ...
                            + reshape(DKuu_gDtheta_i', 1, u*u)*coef_Kuu_g;
        end

        DFDlogthetag(end-1) = DFDlogthetag(end-1) - 0.25*((diagAandB'+1)*DdiagKnn_gDsf2_g); % DdiagKnn_gDsf2_g is only available for sf2_g
    
        % wrt mu0
        DFDlogthetag(model.GPg.nParams) = 0.5*sum(diagAandB);   

        %-------------------------------------------------
        % Derivatives of F wrt inducing parameters
        %-------------------------------------------------
        % WRT inducing paras for f
        DFDXm = zeros(m,model.Pseudo.nParams);

        % precomputations
        temp = invRgKnmfinvKR_half; % n x m, temp*temp' = inv(R_g) - (Qnn_f + R_g)^{-1}
        term1 = Onm_f'*beta_n;                                % m x 1
        term2 = Onm_f'*temp;                                  % m x m
        Amn = term1*beta_n' + term2*temp';                    % m x n
        Anm = Amn';                                           % n x m
        Amm = - term1*term1' - term2*term2';                  % m x m
       
        % avoid for loop to save computing efficiency
        % suppose DX is a m x d matrix, A is a m x k matrix, B is a k x d matrix, C is a m x d matrix
        % if DX(:,i) = sum(A.*E,2) where E = repmat(B(:,i)',m,1), we have
        %                         DX = A*B
        % if DX(:,i) = sum(A.*E,2) where E = repmat(C(:,i),1,d), we have, given a = sum(A,2),
        %                    DX = C.*repmat(a,1,d)

        % transform data by lengthscales
        X_bar = model.X*diag((1./ell_f).^2);         % n x D
        Xm_bar = model.Pseudo.Xm*diag((1./ell_f).^2);% m x D
        
        AA = Amn.*model.Knm_f'; AA1 = AA*ones(n,1); 
        DFDXm_t1 = AA*X_bar - Xm_bar.*repmat(AA1,1,model.D);
        AA = -Anm'.*model.Knm_f'; AA1 = AA*ones(n,1);
        DFDXm_t2 = Xm_bar.*repmat(AA1,1,model.D) - AA*X_bar;
        AA = Amm.*(model.Kmm_f - sf2_f*jitter_SE*eye(m))'; AA1 = AA*ones(m,1);
        DFDXm_t3 = AA*Xm_bar - Xm_bar.*repmat(AA1,1,model.D);
        AA = Amm'.*(model.Kmm_f - sf2_f*jitter_SE*eye(m)); AA1 = AA*ones(m,1);
        DFDXm_t4 = Xm_bar.*repmat(AA1,1,model.D) - AA*Xm_bar; 

        DFDXm = 0.5*(DFDXm_t1 + DFDXm_t2 + DFDXm_t3 - DFDXm_t4);

        clear Amn Anm Amm AA Xm_bar DFDXm_t1 DFDXm_t2 DFDXm_t3 DFDXm_t4


        % WRT inducing paras for g
        DFDXu = zeros(u,model.Pseudo.nParams);

        % precomputations for F1 + F2 + F3
        Aun = (Onu_g'*(diaglambda - 0.5))*diagAandB'; % u x n
        Anu = (diaglambda - 0.5)*(diagAandB'*Onu_g);  % n x u
        Auu = -Aun*Onu_g;                             % u x u
        
        temp = invKLambda*model.Knu_g';                            % u x n
        temp_uu = (temp.*repmat((diagAandB+1)',u,1))*temp';        % u x u
        temp_un = temp_uu*(model.Knu_g'.*repmat(diaglambda',u,1)); % u x n
        Bun = (invKLambda - invKuu_g)*(model.Knu_g'.*repmat((diagAandB+1)',u,1)) - temp_un; % u x n
        Bnu = Bun';                                                % n x u
        Buu = (Onu_g'.*repmat((diagAandB+1)',u,1))*Onu_g - temp_uu;% u x u

        % precomputations for F4
        temp0 = invKLambda*model.Kuu_g;                       % u x u
        temp1 = invKLambda*Luu_g; temp2 = temp0*invL_Sigma_u';% u x u
        term = temp1*temp1' - temp2*temp2';                   % u x u
        r = invKuu_g*(mu_u - mu0*ones(u,1));                  % u x 1
        Cun = -term*(model.Knu_g'.*repmat(diaglambda',u,1)) + invKuu_g*(mu_u - mu0)*(diaglambda - 0.5)'; % u x n
        Cnu = Cun';                                           % n x u
        temp3 = invKuu_g*L_Sigma_u;                           % u x u
        temp4 = invKLambda - temp0*invSigma_u;                % u x u
        Cuu = temp4 + temp4' + invKuu_g - r*r' - temp3*temp3' - term; % u x u

        % avoid for loop to save computing efficiency
        % suppose DX is a m x d matrix, A is a m x k matrix, B is a k x d matrix, C is a m x d matrix
        % if DX(:,i) = sum(A.*E,2) where E = repmat(B(:,i)',m,1), we have
        %                         DX = A*B
        % if DX(:,i) = sum(A.*E,2) where E = repmat(C(:,i),1,d), we have, given a = sum(A,2),
        %                    DX = C.*repmat(a,1,d)

        X_bar = model.X*diag((1./ell_g).^2);          % n x D
        Xu_bar = model.Pseudo.Xu*diag((1./ell_g).^2); % u x D

        AA = (Aun-0.5*Bun-Cun).*model.Knu_g'; AA1 = AA*ones(n,1);
        DFDXu_t1 = AA*X_bar - Xu_bar.*repmat(AA1,1,model.D);
        AA = -(Anu'-0.5*Bnu'-Cnu').*model.Knu_g'; AA1 = AA*ones(n,1);
        DFDXu_t2 = Xu_bar.*repmat(AA1,1,model.D) - AA*X_bar;
        AA = (Auu-0.5*Buu-Cuu).*(model.Kuu_g - sf2_g*jitter_SE*eye(u))'; AA1 = AA*ones(u,1);
        DFDXu_t3 = AA*Xu_bar - Xu_bar.*repmat(AA1,1,model.D);
        AA = (Auu'-0.5*Buu'-Cuu').*(model.Kuu_g - sf2_g*jitter_SE*eye(u)); AA1 = AA*ones(u,1);
        DFDXu_t4 = Xu_bar.*repmat(AA1,1,model.D) - AA*Xu_bar;
        
        DFDXu = 0.5*(DFDXu_t1 + DFDXu_t2 + DFDXu_t3 - DFDXu_t4);

        clear Bun Bnu Buu Cun Cnu Cuu AA X_bar Xu_bar DFDXu_t1 DFDXu_t2 DFDXu_t3 DFDXu_t4

        % Combine derivatives of paras
        st = 1; en = model.Variation.nParams;
        out2(st:en) = - DFDloglambda;
        st = en + 1; en = en + model.GPf.nParams;
        out2(st:en) = - DFDlogthetaf;
        st = en + 1; en = en + model.GPg.nParams;
        out2(st:en) = - DFDlogthetag;
        st = en + 1; en = en + m*model.Pseudo.nParams;
        out2(st:en) = - reshape(DFDXm,m*model.Pseudo.nParams,1);
        st = en + 1; en = en + u*model.Pseudo.nParams;
        out2(st:en) = - reshape(DFDXu,u*model.Pseudo.nParams,1);
    elseif flag == 2 % Derivatives wrt variational + kernel f,g parameters
        %------------------------------------------------
        % Derivatives of F wrt variational parameters
        %------------------------------------------------
        temp = model.Knu_g*invL_KLambda;                      % n x u
        coef = 0.25*diagAandB + 0.25;                         % n x 1
        temp2 = ((temp'.*repmat(coef',u,1))*temp)*temp';      % u x n
        term1 = diagAB(temp,temp2);                           % n x 1

        Bun = model.Kuu_g*invKLambda*model.Knu_g';            % u x n
        term2 = -0.5*diagAB(Bun'*(invSigma_u - invKuu_g),Bun);% n x 1 
    
        DFDloglambda = 0.5*Qnn_g_half*(Qnn_g_half'*diagAandB) + term1 + term2 - mu_g + mu0; % n x 1
        DFDloglambda = DFDloglambda.*diaglambda;                                            % n x 1

        clear coef Qnn_g_half

        %-----------------------------------------
        % Derivatives of F wrt f kernel parameters
        %-----------------------------------------
        DFDlogthetaf = zeros(model.GPf.nParams,1);
        jitter_SE = 1e-6; % used in covSEard_jitter_vshgp.m
        ell_f = exp(model.GPf.logtheta(1:model.D)); sf2_f = exp(2*model.GPf.logtheta(model.D+1)); % D x 1

        % preconputations
        A1 = Onm_f'*beta_n;                                 % m x 1
        A2 = invRgKnmfinvKR_half;
        A3 = Onm_f'*A2*A2';                                 % m x n
        A4 = A1*beta_n';                                    % m x n
        A5 = A2'*Onm_f;                                     % m x m

        DdiagKnn_fDsf2_f = 2*model.diagKnn_f; % diagKnn_f wrt ell is zero <<- SE ard kernel

        % -->> avoid for loop to improve computing efficiency <<--
        % suppose A is a n x m matrix, and B is a m x n matrix, then
        % trace(A*B) = sum(sum(A.*B,2)) = reshape(A',1,n*m)*reshape(B,n*m,1)
        % precalculate Bs
        coef_Knm_f = reshape(0.5*(A4 + A3), m*n, 1);            % (m*n) x 1
        coef_Kmn_f = reshape(0.5*A4' + 0.5*A2*A5, m*n, 1);      % (m*n) x 1
        coef_Kmm_f = reshape(-0.5*A1*A1' - 0.5*A5'*A5, m*m, 1); % (m*m) x 1
      
        % precalculate As
        for i = 1:model.GPf.nParams % D + 1
            DKnm_fDtheta_i = covSEard_jitter_grad_vshgp(model.GPf.logtheta, model.X, model.Pseudo.Xm, model.Knm_f, i); % n x m
            DKmm_fDtheta_i = covSEard_jitter_grad_vshgp(model.GPf.logtheta, model.Pseudo.Xm, [], model.Kmm_f, i);      % m x m
            DFDlogthetaf(i) = reshape(DKnm_fDtheta_i', 1, n*m)*coef_Knm_f ...
                            + reshape(DKnm_fDtheta_i,  1, m*n)*coef_Kmn_f ...
                            + reshape(DKmm_fDtheta_i', 1, m*m)*coef_Kmm_f;
        end

        DFDlogthetaf(end) = DFDlogthetaf(end) - 0.5*(diagInvR_g'*DdiagKnn_fDsf2_f); % DdiagKnn_fDsf2_f is only available for sf2_f

        %-------------------------------------------------
        % Derivatives of F wrt g kernel parameters and mu0
        %-------------------------------------------------  
        % WRT kernel paras
        DFDlogthetag = zeros(model.GPg.nParams,1);
        ell_g = exp(model.GPg.logtheta(1:model.D)); sf2_g = exp(2*model.GPg.logtheta(model.D+1));

        % precomputations
        A1 = invKLambda*model.Knu_g';                          % u x n
        A3 = Onu_g'*(diaglambda - 0.5);                        % u x 1
        A5 = (mu_u - mu0)'*invKuu_g;                           % 1 x u
        A6 = (invKLambda*model.Kuu_g)*(invSigma_u - invKuu_g); % u x u
        A6A2T = A6*(invKLambda*model.Kuu_g)';                  % u x u
        A7 = model.Knu_g.*repmat(diaglambda,1,u);              % n x u
        A8 = invKuu_g - invKuu_g*(L_Sigma_u*L_Sigma_u')*invKuu_g ...
           - A6 - A6' + A6A2T;                                 % u x u
        A9 = A7*A6A2T;                                         % n x u
        A10 = A6A2T*A7';                                       % u x n

        DdiagKnn_gDsf2_g = 2*model.diagKnn_g; % diagKnn_f wrt ell is zero <<- SE ard kernel

        % -->> avoid for loop to improve computing efficiency <<--
        % suppose A is a n x m matrix, and B is a m x n matrix, then
        % trace(A*B) = sum(sum(A.*B,2)) = reshape(A',1,n*m)*reshape(B,n*m,1)
        % precalculate Bs
        coef_Knu_g = 0.5*A3*diagAandB' ...
                   - 0.25*(2*A1.*repmat(diagAandB'+1,u,1) - (A1.*repmat(diagAandB'+1,u,1))*A1'*A7') ...
                   - 0.5*A10;                    % u x n
        coef_Knu_g = reshape(coef_Knu_g,n*u,1);  % n*u x 1 = n x u
        coef_Kun_g = 0.5*(diaglambda - 0.5)*(diagAandB'*Onu_g) ...
                   - 0.25*(-2*Onu_g.*repmat(diagAandB+1,1,u) - A7*((A1.*repmat(diagAandB'+1,u,1))*A1')) ...
                   - 0.5*A9 ...
                   - 0.5*2*(diaglambda - 0.5)*A5;% n x u
        coef_Kun_g = reshape(coef_Kun_g,n*u,1);  % n*u x 1 = n x u
        coef_Kuu_g = -0.5*A3*diagAandB'*Onu_g ...
                   -0.25*((Onu_g'.*repmat(diagAandB'+1,u,1))*Onu_g - (A1.*repmat(diagAandB'+1,u,1))*A1') ...
                   - 0.5*A8 ...
                   + 0.5*(A5'*A5);               % u x u
        coef_Kuu_g = reshape(coef_Kuu_g,u*u,1);  % u*u x 1 = u x u

        % precalculate As
        for i = 1:model.GPg.nParams - 1 % D + 1
            DKnu_gDtheta_i = covSEard_jitter_grad_vshgp(model.GPg.logtheta, model.X, model.Pseudo.Xu, model.Knu_g, i); % n x m
            DKuu_gDtheta_i = covSEard_jitter_grad_vshgp(model.GPg.logtheta, model.Pseudo.Xu, [], model.Kuu_g, i);      % m x m
            DFDlogthetag(i) = reshape(DKnu_gDtheta_i', 1, n*u)*coef_Knu_g ...
                            + reshape(DKnu_gDtheta_i,  1, u*n)*coef_Kun_g ...
                            + reshape(DKuu_gDtheta_i', 1, u*u)*coef_Kuu_g;
        end

        DFDlogthetag(end-1) = DFDlogthetag(end-1) - 0.25*((diagAandB'+1)*DdiagKnn_gDsf2_g); % DdiagKnn_gDsf2_g is only available for sf2_g

    
        % wrt mu0
        DFDlogthetag(model.GPg.nParams) = 0.5*sum(diagAandB);   

        % Combine derivatives of paras
        st = 1; en = model.Variation.nParams;
        out2(st:en) = - DFDloglambda;
        st = en + 1; en = en + model.GPf.nParams;
        out2(st:en) = - DFDlogthetaf;
        st = en + 1; en = en + model.GPg.nParams;
        out2(st:en) = - DFDlogthetag;
    elseif flag == 3 % Derivatives wrt variational + kernel g parameters
        %------------------------------------------------
        % Derivatives of F wrt variational parameters
        %------------------------------------------------
        temp = model.Knu_g*invL_KLambda;                      % n x u
        coef = 0.25*diagAandB + 0.25;                         % n x 1
        temp2 = ((temp'.*repmat(coef',u,1))*temp)*temp';      % u x n
        term1 = diagAB(temp,temp2);                           % n x 1

        Bun = model.Kuu_g*invKLambda*model.Knu_g';            % u x n
        term2 = -0.5*diagAB(Bun'*(invSigma_u - invKuu_g),Bun);% n x 1 
    
        DFDloglambda = 0.5*Qnn_g_half*(Qnn_g_half'*diagAandB) + term1 + term2 - mu_g + mu0; % n x 1
        DFDloglambda = DFDloglambda.*diaglambda;                                            % n x 1

        clear coef Qnn_g_half

        %-------------------------------------------------
        % Derivatives of F wrt g kernel parameters and mu0
        %-------------------------------------------------  
        % WRT kernel paras
        DFDlogthetag = zeros(model.GPg.nParams,1);
        ell_g = exp(model.GPg.logtheta(1:model.D)); sf2_g = exp(2*model.GPg.logtheta(model.D+1));

        % precomputations
        A1 = invKLambda*model.Knu_g';                          % u x n
        A3 = Onu_g'*(diaglambda - 0.5);                        % u x 1
        A5 = (mu_u - mu0)'*invKuu_g;                           % 1 x u
        A6 = (invKLambda*model.Kuu_g)*(invSigma_u - invKuu_g); % u x u
        A6A2T = A6*(invKLambda*model.Kuu_g)';                  % u x u
        A7 = model.Knu_g.*repmat(diaglambda,1,u);              % n x u
        A8 = invKuu_g - invKuu_g*(L_Sigma_u*L_Sigma_u')*invKuu_g ...
           - A6 - A6' + A6A2T;                                 % u x u
        A9 = A7*A6A2T;                                         % n x u
        A10 = A6A2T*A7';                                       % u x n

        DdiagKnn_gDsf2_g = 2*model.diagKnn_g; % diagKnn_f wrt ell is zero <<- SE ard kernel

        % -->> avoid for loop to improve computing efficiency <<--
        % suppose A is a n x m matrix, and B is a m x n matrix, then
        % trace(A*B) = sum(sum(A.*B,2)) = reshape(A',1,n*m)*reshape(B,n*m,1)
        % precalculate Bs
        coef_Knu_g = 0.5*A3*diagAandB' ...
                   - 0.25*(2*A1.*repmat(diagAandB'+1,u,1) - (A1.*repmat(diagAandB'+1,u,1))*A1'*A7') ...
                   - 0.5*A10;                    % u x n
        coef_Knu_g = reshape(coef_Knu_g,n*u,1);  % n*u x 1 = n x u
        coef_Kun_g = 0.5*(diaglambda - 0.5)*(diagAandB'*Onu_g) ...
                   - 0.25*(-2*Onu_g.*repmat(diagAandB+1,1,u) - A7*((A1.*repmat(diagAandB'+1,u,1))*A1')) ...
                   - 0.5*A9 ...
                   - 0.5*2*(diaglambda - 0.5)*A5;% n x u
        coef_Kun_g = reshape(coef_Kun_g,n*u,1);  % n*u x 1 = n x u
        coef_Kuu_g = -0.5*A3*diagAandB'*Onu_g ...
                   -0.25*((Onu_g'.*repmat(diagAandB'+1,u,1))*Onu_g - (A1.*repmat(diagAandB'+1,u,1))*A1') ...
                   - 0.5*A8 ...
                   + 0.5*(A5'*A5);               % u x u
        coef_Kuu_g = reshape(coef_Kuu_g,u*u,1);  % u*u x 1 = u x u

        % precalculate As
        for i = 1:model.GPg.nParams - 1 % D + 1
            DKnu_gDtheta_i = covSEard_jitter_grad_vshgp(model.GPg.logtheta, model.X, model.Pseudo.Xu, model.Knu_g, i); % n x m
            DKuu_gDtheta_i = covSEard_jitter_grad_vshgp(model.GPg.logtheta, model.Pseudo.Xu, [], model.Kuu_g, i);      % m x m
            DFDlogthetag(i) = reshape(DKnu_gDtheta_i', 1, n*u)*coef_Knu_g ...
                            + reshape(DKnu_gDtheta_i,  1, u*n)*coef_Kun_g ...
                            + reshape(DKuu_gDtheta_i', 1, u*u)*coef_Kuu_g;
        end

        DFDlogthetag(end-1) = DFDlogthetag(end-1) - 0.25*((diagAandB'+1)*DdiagKnn_gDsf2_g); % DdiagKnn_gDsf2_g is only available for sf2_g

    
        % wrt mu0
        DFDlogthetag(model.GPg.nParams) = 0.5*sum(diagAandB);   

        % Combine derivatives of paras
        st = 1; en = model.Variation.nParams;
        out2(st:en) = - DFDloglambda;
        st = en + model.GPf.nParams + 1; en = en + model.GPf.nParams + model.GPg.nParams;
        out2(st:en) = - DFDlogthetag;
    elseif flag == 4 % Derivatives wrt variational parameters
        %------------------------------------------------
        % Derivatives of F wrt variational parameters
        %------------------------------------------------
        temp = model.Knu_g*invL_KLambda;                      % n x u
        coef = 0.25*diagAandB + 0.25;                         % n x 1
        temp2 = ((temp'.*repmat(coef',u,1))*temp)*temp';      % u x n
        term1 = diagAB(temp,temp2);                           % n x 1

        Bun = model.Kuu_g*invKLambda*model.Knu_g';            % u x n
        term2 = -0.5*diagAB(Bun'*(invSigma_u - invKuu_g),Bun);% n x 1 
    
        DFDloglambda = 0.5*Qnn_g_half*(Qnn_g_half'*diagAandB) + term1 + term2 - mu_g + mu0; % n x 1
        DFDloglambda = DFDloglambda.*diaglambda;                                            % n x 1

        clear coef Qnn_g_half

        % Combine derivatives of paras
        st = 1; en = model.Variation.nParams;
        out2(st:en) = - DFDloglambda;
    elseif flag == 5 % Derivatives wrt variational parameters + inducing points
        %------------------------------------------------
        % Derivatives of F wrt variational parameters
        %------------------------------------------------
        temp = model.Knu_g*invL_KLambda;                      % n x u
        coef = 0.25*diagAandB + 0.25;                         % n x 1
        temp2 = ((temp'.*repmat(coef',u,1))*temp)*temp';      % u x n
        term1 = diagAB(temp,temp2);                           % n x 1

        Bun = model.Kuu_g*invKLambda*model.Knu_g';            % u x n
        term2 = -0.5*diagAB(Bun'*(invSigma_u - invKuu_g),Bun);% n x 1 
    
        DFDloglambda = 0.5*Qnn_g_half*(Qnn_g_half'*diagAandB) + term1 + term2 - mu_g + mu0; % n x 1
        DFDloglambda = DFDloglambda.*diaglambda;                                            % n x 1

        clear coef Qnn_g_half

        %-------------------------------------------------
        % Derivatives of F wrt inducing parameters
        %-------------------------------------------------
        jitter_SE = 1e-6; % used in covSEard_jitter_vshgp.m
        ell_f = exp(model.GPf.logtheta(1:model.D)); sf2_f = exp(2*model.GPf.logtheta(model.D+1)); % D x 1
        ell_g = exp(model.GPg.logtheta(1:model.D)); sf2_g = exp(2*model.GPg.logtheta(model.D+1));

        % WRT inducing paras for f
        DFDXm = zeros(m,model.Pseudo.nParams);

        % precomputations
        temp = invRgKnmfinvKR_half; % n x m, temp*temp' = inv(R_g) - (Qnn_f + R_g)^{-1}
        term1 = Onm_f'*beta_n;                                % m x 1
        term2 = Onm_f'*temp;                                  % m x m
        Amn = term1*beta_n' + term2*temp';                    % m x n
        Anm = Amn';                                           % n x m
        Amm = - term1*term1' - term2*term2';                  % m x m
       
        % avoid for loop to save computing efficiency
        % suppose DX is a m x d matrix, A is a m x k matrix, B is a k x d matrix, C is a m x d matrix
        % if DX(:,i) = sum(A.*E,2) where E = repmat(B(:,i)',m,1), we have
        %                         DX = A*B
        % if DX(:,i) = sum(A.*E,2) where E = repmat(C(:,i),1,d), we have, given a = sum(A,2),
        %                    DX = C.*repmat(a,1,d)

        % transform data by lengthscales
        X_bar = model.X*diag((1./ell_f).^2);         % n x D
        Xm_bar = model.Pseudo.Xm*diag((1./ell_f).^2);% m x D
        
        AA = Amn.*model.Knm_f'; AA1 = AA*ones(n,1); 
        DFDXm_t1 = AA*X_bar - Xm_bar.*repmat(AA1,1,model.D);
        AA = -Anm'.*model.Knm_f'; AA1 = AA*ones(n,1);
        DFDXm_t2 = Xm_bar.*repmat(AA1,1,model.D) - AA*X_bar;
        AA = Amm.*(model.Kmm_f - sf2_f*jitter_SE*eye(m))'; AA1 = AA*ones(m,1);
        DFDXm_t3 = AA*Xm_bar - Xm_bar.*repmat(AA1,1,model.D);
        AA = Amm'.*(model.Kmm_f - sf2_f*jitter_SE*eye(m)); AA1 = AA*ones(m,1);
        DFDXm_t4 = Xm_bar.*repmat(AA1,1,model.D) - AA*Xm_bar; 

        DFDXm = 0.5*(DFDXm_t1 + DFDXm_t2 + DFDXm_t3 - DFDXm_t4);

        clear Amn Anm Amm AA Xm_bar DFDXm_t1 DFDXm_t2 DFDXm_t3 DFDXm_t4


        % WRT inducing paras for g
        DFDXu = zeros(u,model.Pseudo.nParams);

        % precomputations for F1 + F2 + F3
        Aun = (Onu_g'*(diaglambda - 0.5))*diagAandB'; % u x n
        Anu = (diaglambda - 0.5)*(diagAandB'*Onu_g);  % n x u
        Auu = -Aun*Onu_g;                             % u x u
        
        temp = invKLambda*model.Knu_g';                            % u x n
        temp_uu = (temp.*repmat((diagAandB+1)',u,1))*temp';        % u x u
        temp_un = temp_uu*(model.Knu_g'.*repmat(diaglambda',u,1)); % u x n
        Bun = (invKLambda - invKuu_g)*(model.Knu_g'.*repmat((diagAandB+1)',u,1)) - temp_un; % u x n
        Bnu = Bun';                                                % n x u
        Buu = (Onu_g'.*repmat((diagAandB+1)',u,1))*Onu_g - temp_uu;% u x u

        % precomputations for F4
        temp0 = invKLambda*model.Kuu_g;                       % u x u
        temp1 = invKLambda*Luu_g; temp2 = temp0*invL_Sigma_u';% u x u
        term = temp1*temp1' - temp2*temp2';                   % u x u
        r = invKuu_g*(mu_u - mu0*ones(u,1));                  % u x 1
        Cun = -term*(model.Knu_g'.*repmat(diaglambda',u,1)) + invKuu_g*(mu_u - mu0)*(diaglambda - 0.5)'; % u x n
        Cnu = Cun';                                           % n x u
        temp3 = invKuu_g*L_Sigma_u;                           % u x u
        temp4 = invKLambda - temp0*invSigma_u;                % u x u
        Cuu = temp4 + temp4' + invKuu_g - r*r' - temp3*temp3' - term; % u x u

        % avoid for loop to save computing efficiency
        % suppose DX is a m x d matrix, A is a m x k matrix, B is a k x d matrix, C is a m x d matrix
        % if DX(:,i) = sum(A.*E,2) where E = repmat(B(:,i)',m,1), we have
        %                         DX = A*B
        % if DX(:,i) = sum(A.*E,2) where E = repmat(C(:,i),1,d), we have, given a = sum(A,2),
        %                    DX = C.*repmat(a,1,d)

        X_bar = model.X*diag((1./ell_g).^2);          % n x D
        Xu_bar = model.Pseudo.Xu*diag((1./ell_g).^2); % u x D

        AA = (Aun-0.5*Bun-Cun).*model.Knu_g'; AA1 = AA*ones(n,1);
        DFDXu_t1 = AA*X_bar - Xu_bar.*repmat(AA1,1,model.D);
        AA = -(Anu'-0.5*Bnu'-Cnu').*model.Knu_g'; AA1 = AA*ones(n,1);
        DFDXu_t2 = Xu_bar.*repmat(AA1,1,model.D) - AA*X_bar;
        AA = (Auu-0.5*Buu-Cuu).*(model.Kuu_g - sf2_g*jitter_SE*eye(u))'; AA1 = AA*ones(u,1);
        DFDXu_t3 = AA*Xu_bar - Xu_bar.*repmat(AA1,1,model.D);
        AA = (Auu'-0.5*Buu'-Cuu').*(model.Kuu_g - sf2_g*jitter_SE*eye(u)); AA1 = AA*ones(u,1);
        DFDXu_t4 = Xu_bar.*repmat(AA1,1,model.D) - AA*Xu_bar;
        
        DFDXu = 0.5*(DFDXu_t1 + DFDXu_t2 + DFDXu_t3 - DFDXu_t4);

        clear Bun Bnu Buu Cun Cnu Cuu AA X_bar Xu_bar DFDXu_t1 DFDXu_t2 DFDXu_t3 DFDXu_t4

        % Combine derivatives of paras
        st = 1; en = model.Variation.nParams;
        out2(st:en) = - DFDloglambda;
        st = en + 1; en = en + model.GPf.nParams;
        %out2(st:en) = - DFDlogthetaf;
        st = en + 1; en = en + model.GPg.nParams;
        %out2(st:en) = - DFDlogthetag;
        st = en + 1; en = en + m*model.Pseudo.nParams;
        out2(st:en) = - reshape(DFDXm,m*model.Pseudo.nParams,1);
        st = en + 1; en = en + u*model.Pseudo.nParams;
        out2(st:en) = - reshape(DFDXu,u*model.Pseudo.nParams,1);
    end
elseif nargin == 4 % predicting mode
    %------------
    % Predictions
    %------------
    [nt,D] = size(Xtest);

    nperbatch = 1e4; nact = 0 ;
    mustar = zeros(nt,1); mu_fstar = zeros(nt,1); mu_gstar = zeros(nt,1);
    varstar = zeros(nt,1); var_fstar = zeros(nt,1); var_gstar = zeros(nt,1);

    % fast version
    while nact < nt % process minibatches of test cases to save memory
        id = (nact+1):min(nact+nperbatch,nt);
    
        Kstarm_f = covSEard_jitter_vshgp(model.GPf.logtheta, Xtest(id,:), model.Pseudo.Xm); % nperbatch x m
        Kstaru_g = covSEard_jitter_vshgp(model.GPg.logtheta, Xtest(id,:), model.Pseudo.Xu); % nperbatch x u
        Kstarstar_f_diag = covSEard_jitter_vshgp(model.GPf.logtheta, Xtest(id,:), 'diag');  % nperbatch x 1
        Kstarstar_g_diag = covSEard_jitter_vshgp(model.GPg.logtheta, Xtest(id,:), 'diag');  % nperbatch x 1
        
        % predictive mean
        mu_fstar(id) = Kstarm_f*(invL_KR*invRgKnmfinvKR_half')*model.y;      % nperbatch x 1
        mustar(id) = mu_fstar(id);                                           % nperbatch x 1
        
        % predictive self-variance
        term1 = Kstarm_f*invLmm_f';                                                    % nperbatch x m
        term2 = Kstarm_f*invL_KR;                                                      % nperbatch x m
        var_fstar(id) = Kstarstar_f_diag - diagAB(term1,term1') + diagAB(term2,term2');% nperbatch x 1
        
        mu_gstar(id) = Kstaru_g*invKuu_g*(mu_u - mu0) + mu0;                 % nperbatch x 1
        term1 = Kstaru_g*invLuu_g';                                          % nperbatch x u
        term2 = Kstaru_g*invKuu_g*L_Sigma_u;                                           % nperbatch x u
        var_gstar(id) = Kstarstar_g_diag - diagAB(term1,term1') + diagAB(term2,term2');% nperbatch x 1
        
        varstar(id) = var_fstar(id) + exp(mu_gstar(id) + var_gstar(id)/2);   % nperbatch x 1
    
        nact = id(end);                                                      % set counter to index of last processed data point
    end

    out1.mustar = mustar;   out1.mu_fstar = mu_fstar;   out1.mu_gstar = mu_gstar;
    out2.varstar = varstar; out2.var_fstar = var_fstar; out2.var_gstar = var_gstar;
end
