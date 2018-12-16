function model = returnOptimizedParams_vshgp(model, paras, flag)
%
% model: a structure
% paras: optimized parameters
% flag: = 1: update variational paras, f&g kernel paras and inducing paras 
%       = 2: update variational paras and f&g kernel paras
%       = 3: update variational paras and g kernel paras
%       = 4: update variational paras
%


if flag == 1
    % variational parameters
    st = 1; en = model.Variation.nParams;
    model.Variation.loglambda = paras(st:en);
    model.Variation.diaglambda = exp(model.Variation.loglambda);

    % kernel parameters
    st = en + 1; en = en + model.GPf.nParams;
    model.GPf.logtheta = paras(st:en);
    model.GPf.s2 = exp(2*model.GPf.logtheta(end));
    st = en + 1; en = en + model.GPg.nParams;
    model.GPg.logtheta = paras(st:en-1);
    model.GPg.mu0 = paras(en);
    model.GPg.s2 = exp(2*model.GPg.logtheta(end)); 

    % inducing variables parameters
    st = en + 1; en = en + model.m(1)*model.Pseudo.nParams;
    model.Pseudo.Xm = reshape(paras(st:en), model.m(1), model.Pseudo.nParams);
    st = en + 1; en = en + model.m(2)*model.Pseudo.nParams;
    model.Pseudo.Xu = reshape(paras(st:en), model.m(2), model.Pseudo.nParams);    

    if en < length(paras); error('Unmatching parameters.'); end
elseif flag == 2
    % variational parameters
    st = 1; en = model.Variation.nParams;
    model.Variation.loglambda = paras(st:en);
    model.Variation.diaglambda = exp(model.Variation.loglambda);

    % kernel parameters
    st = en + 1; en = en + model.GPf.nParams;
    model.GPf.logtheta = paras(st:en);
    model.GPf.s2 = exp(2*model.GPf.logtheta(end));
    st = en + 1; en = en + model.GPg.nParams;
    model.GPg.logtheta = paras(st:en-1);
    model.GPg.mu0 = paras(en);
    model.GPg.s2 = exp(2*model.GPg.logtheta(end)); 
elseif flag == 3
    % variational parameters
    st = 1; en = model.Variation.nParams;
    model.Variation.loglambda = paras(st:en);
    model.Variation.diaglambda = exp(model.Variation.loglambda);

    % kernel g parameters
    st = en + model.GPf.nParams + 1; en = en + model.GPf.nParams + model.GPg.nParams;
    model.GPg.logtheta = paras(st:en-1);
    model.GPg.mu0 = paras(en);
    model.GPg.s2 = exp(2*model.GPg.logtheta(end)); 
elseif flag == 4
    % variational parameters
    st = 1; en = model.Variation.nParams;
    model.Variation.loglambda = paras(st:en);
    model.Variation.diaglambda = exp(model.Variation.loglambda);
end

