function sol = fmm_em_update(pb, sol)
%PLI_FMM_EM_UPDATE Update an EM solution of Finite mixture estimation
%
%   sol = PLI_FMM_EM_UPDATE(pb, sol);
%

mdl = pb.model;
n = pb.nobs;
obs = pb.obs;
w = pb.weights;
pric = pb.pricount;

% M-step

if isempty(w)
    pi = sol.Q * ones(n, 1);
    W = sol.Q.';
else
    pi = sol.Q * w;
    W = bsxfun(@times, sol.Q.', w);
end

if pric > 0
    pi = pi + pric;
end
pi = pi / sum(pi);

params = update_params(mdl, obs, W, [], sol.components);
L = evaluate_loglik(mdl, params, obs);

% E-step

Q = pli_nrmexp(bsxfun(@plus, L, log(pi)), 1);

% write update to sol

sol.pi = pi;
sol.components = params;
sol.logliks = L;
sol.Q = Q;

