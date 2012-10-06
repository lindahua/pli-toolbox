function v = pli_dirichlet_entropy(alpha, elog)
%PLI_DIRICHLET_ENTROPY Evaluates the entropy of Dirichlet distribution
%
%   v = PLI_DIRICHLET_ENTROPY(alpha);
%   
%       Evaluates the entropy for a Dirichlet distribution or 
%       multiple Dirichlet distributions, with parameter alpha.
%
%   v = PLI_DIRICHLET_ENTROPY(alpha, elog); 
%
%       This statement allows the user to supply the pre-computed
%       expectation of log(x). 
%
%       The size of elog should be the same as that of alpha.
%

%% main

a = alpha;
n = size(a, 2);

if nargin < 2
    if n == 1
        elog = psi(a) - psi(sum(a));
    else
        elog = bsxfun(@minus, psi(a), psi(sum(a, 1)));
    end
end

if n == 1
    v = sum(gammaln(a)) - gammaln(sum(a)) - (a - 1)' * elog;
else
    v = sum(gammaln(a)) - gammaln(sum(a)) - sum((a - 1) .* elog, 1);
end

