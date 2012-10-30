function [phi, objv] = pli_gmrfest(pat, X, lambda, phi0, opts)
%PLI_GMRFEST Gaussian MRF Estimation
%
%   [PHI, OBJV] = PLI_GMRFEST(pat, X, lambda, phi0);
%   [PHI, OBJV] = PLI_GMRFEST(pat, X, lambda, phi0, opts);
%
%       Estimates the parameters of a Gaussian MRF based on a specific
%       pattern from given samples.
%
%       The objective here is to minimize the following function:
%
%           - sum_{i=1}^n log p(x_i | A) + (lambda/2)' * (phi)^2
%
%       Here, A is a concentration matrix defined by pli_pat2mat(pat, phi).
%
%       The log-pdf is defined to be
%
%           log p(x | A) 
%           = -(1/2) * ( x' * A * x - logdet(A) + d * log(2*pi) )
%    
%       This is a convex optimization problem, and this function invokes
%       pli_fminunc to solve it.
%
%   Arguments
%   ---------
%   - pat :     The covariance matrix pattern, which defines the basic
%               structure of the MRF, as follows
%
%               Let phi be the parameter, then the concentration matrix
%               A is defined by:
%
%                   A(i, j) = 0                 when pat(i, j) = 0
%                           = phi(pat(i, j))    when pat(i, j) > 0
%
%               To ensure A is a symmetric matrix, pat in itself should
%               also be symmetric.
%
%   - X :       The sample matrix, of size [d, n]. Here, d is the
%               sample dimension, and n is the number of samples.
%
%   - lambda :  The regularization coefficient, which can be
%               either a scalar or a vector of size [d 1].
%
%   - phi0 :    The initial guess of the parameter vector. The input phi0
%               should ensure A(phi0) is positive definite.
%
%   - opts :    The option struct to be supplied to the optimization
%               function. It can be made using pli_optimset.
%
%   Returns
%   -------
%   - phi :     The estimated parameter
%
%   - objv :    The objective function value at phi.
%                   

%% argument checking

if ~(isnumeric(pat) && ismatrix(pat) && isreal(pat) ...
        && size(pat,1) == size(pat,2))    
    error('pli_gmrfest:invalidarg', ...
        'pat should be a square numeric matrix.');    
end
d = size(pat, 1);

if ~(isfloat(X) && ismatrix(X) && isreal(X) && size(X,1) == d)
    error('pli_gmrfest:invalidarg', ...
        'X should be a real matrix with size(X,1) = d.');
end

if ~(isfloat(lambda) && isvector(lambda) && isreal(lambda))
    error('pli_gmrfest:invalidarg', ...
        'lambda should be a real vector/scalar.');
end

if ~(isfloat(phi0) && isreal(phi0) && isvector(phi0) && size(phi0, 2)==1)
    error('pli_gmrfest:invalidarg', ...
        'phi0 should be a real column vector.');
end
dp = size(phi0, 1);

if ~(isscalar(lambda) || isequal(size(lambda), [dp 1]))
    error('pli_gmrfest:invalidarg', ...
        'lambda should be either a scalar or a vector of size dp-by-1.');
end

if nargin < 5
    opts = pli_optimset('Display', 'on', 'tolfun', 1.0e-12);
else
    if ~isstruct(opts) 
        error('pli_gmrfest:invalidarg', 'opts should be a struct.');
    end
end


%% main

[I, J, IDX] = find(pat);
pats = {I, J, IDX, full(pat)};

n = size(X, 2);
S = sum(X(I, :) .* X(J, :), 2) * (1/n);
lambda_ = lambda * (1/n);

fobj = @(p) pli_gmrfest_objv(p, pats, S, lambda_);

[phi, fv] = pli_fminunc(fobj, phi0, opts);

if nargout >= 2
    fv = fv + (d/2) * log(2*pi);    
    objv = fv * n;
end


