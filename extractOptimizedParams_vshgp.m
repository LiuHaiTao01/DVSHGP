function paras = extractOptimizedParams_vshgp(model)
% Extract variational + kernel + inducing paras
% Haitao Liu (htliu@ntu.edu.sg) 2017/11/27

Xm = model.Pseudo.Xm; % inducing points for f
Xu = model.Pseudo.Xu; % inducing points for g
    
paras = [model.Variation.loglambda; ...                                                                  % variational paras
        model.GPf.logtheta; ...                                                                         % kernel f paras
        model.GPg.logtheta; model.GPg.mu0; ...                                                          % kernel g paras
        reshape(Xm, model.m(1)*model.Pseudo.nParams,1); reshape(Xu, model.m(2)*model.Pseudo.nParams,1)];% inducing paras
