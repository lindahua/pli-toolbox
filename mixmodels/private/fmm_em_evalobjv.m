function objv = fmm_em_evalobjv(pb, sol)
%FMM_EM_EVALOBJV Evaluates objective of EM estimation for FMM
%
%   objv = FMM_EM_EVALOBJV(pb, sol);
%

mdl = pb.model;
w = pb.weights;
pric = pb.pricount;

Q = sol.Q;
n = size(Q, 2);

if isempty(w)
    w = ones(n, 1);
end

% sum of observation log-likelihoods

L = sol.logliks;
if isempty(L)
    L = evaluate_loglik(mdl, sol.components, pb.obs);
end

sll_obs = sum(L .* Q, 1) * w;

% sum of assignment log-likelihoods

log_pi = log(sol.pi);
sll_q = log_pi' * (Q * w);

% sum of component log-priors

lpri_comp = evaluate_logpri(mdl, sol.components);
lpri_comp = sum(lpri_comp);

% the log-prior of pi

if pric > 0
    lpri_pi = pric * sum(log_pi);
else
    lpri_pi = 0;
end

% entropy of Q

ent_q = - (nansum(Q .* log(Q), 1) * w);

% overall

objv = sll_obs + sll_q + lpri_comp + lpri_pi + ent_q;

