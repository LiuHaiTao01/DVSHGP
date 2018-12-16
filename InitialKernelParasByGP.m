function logKernelParas = InitialKernelParasByGP(X, y, options)
% 

[n, d] = size(X);

% Use SOD GP hyperparameters to initialize GPR's hyperparameters   
n_SoD = options.n_SoD; 
idx = randperm(n);
X_train_SoD = X(idx(1:n_SoD),:); y_train_SoD = y(idx(1:n_SoD));

SignalPower = var(y_train_SoD,1); NoisePower  = SignalPower/4;
lengthscales= log((max(X_train_SoD)-min(X_train_SoD))'/5);  
display('Running standard, homoscedastic GP to initialize f kernel parameters ...')
covfunc = @covSEard; meanfunc = []; hypf.mean = []; likfunc = @likGauss; inffunc = @infGaussLik;
hypf.cov = [lengthscales; 0.5*log(SignalPower)]; hypf.lik = 0.5*log(NoisePower);
hypf_opt = minimize(hypf,@gp_ml,options.GPnumIter,inffunc,meanfunc,covfunc,likfunc,X_train_SoD,y_train_SoD);

[y_train_SoD_pred, ~, ~, s2f_train_SoD] = gp_ml(hypf_opt,inffunc,meanfunc,covfunc,likfunc,X_train_SoD,y_train_SoD,X_train_SoD);
y_train_SoD_noise = log((y_train_SoD - y_train_SoD_pred).^2);
%%%
y_train_SoD_noise_norm = (y_train_SoD_noise - mean(y_train_SoD_noise)) / std(y_train_SoD_noise);
%%%
SignalPower = var(y_train_SoD_noise_norm,1); NoisePower  = SignalPower/4;
lengthscales=log((max(X_train_SoD)-min(X_train_SoD))'/5); 
display('Running standard, homoscedastic GP to initialize g kernel parameters ...')
covfunc = @covSEard; meanfunc = @meanConst; hypg.mean = mean(y_train_SoD_noise_norm); likfunc = @likGauss; inffunc = @infGaussLik;
hypg.cov = [lengthscales; 0.5*log(SignalPower)]; hypg.lik = 0.5*log(NoisePower);
hypg_opt = minimize(hypg,@gp_ml,options.GPnumIter,inffunc,meanfunc,covfunc,likfunc,X_train_SoD,y_train_SoD_noise_norm);

% optimized kernel parameters
lengthscales_f = hypf_opt.cov(1:d); % log
lengthscales_g = hypg_opt.cov(1:d); % log
SignalPower_f  = hypf_opt.cov(d+1); % log
SignalPower_g  = hypg_opt.cov(d+1); % log
mu0            = hypg_opt.mean*std(y_train_SoD_noise) + mean(y_train_SoD_noise);

logKernelParas.lengthscales_f = lengthscales_f;
logKernelParas.lengthscales_g = lengthscales_g;
logKernelParas.SignalPower_f = SignalPower_f;
logKernelParas.SignalPower_g = SignalPower_g;
logKernelParas.mu0 = mu0;