function C = pli_dirichlet_cov(alpha)
%PLI_DIRICHLET_COV Covariance matrix for a Dirichlet distribution
%
%   C = PLI_DIRICHLET_COV(alpha);
%
%       Computes the covariance distribution for a Dirichlet distribution
%       with parameter alpha.
%

%% argument checking

if ~(isfloat(alpha) && isreal(alpha) && size(alpha, 2) == 1)    
    error('pli_dirichlet_cov:invalidarg', ...
        'alpha should be a real column vector.');
end

%% main

a0 = sum(alpha);
s = (a0^2) * (a0 + 1);

C = pli_adddiag((-alpha) * alpha', a0 * alpha);
C = C * (1/s);


