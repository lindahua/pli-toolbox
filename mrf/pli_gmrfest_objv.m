function [objv, grad] = pli_gmrfest_objv(phi, pats, S, lambda, bw)
%PLI_GMRFEST_OBJV Gaussian MRF estimation objective
%
%   The objective function is defined as below
%
%       f(phi) := trace(A * S(X)) - log(det(A)) + lambda' * r(phi); 
%
%   Here, A is the concentration matrix that depends on phi, as
%
%       A(phi) := pli_pat2mat(pat, phi);
%
%   S is the sample covariance of S.
%
%   r(phi) is a smoothed L1 regularizer.
%
%
%   objv = PLI_GMRFEST_OBJV(phi, pat, S, lambda, bw);
%
%       Evaluates the objective function value for Gaussian MRF 
%       estimation at the given parameter phi. 
%
%   [objv, grad] = PLI_GMRFEST_OBJV(phi, pat, S, lambda, bw);    
%
%       Additionally evaluates and returns the gradient at phi.
%
%   Arguments
%   ---------
%   Let d be the sample dimension, and dp be the parameter dimension.
%
%   - phi :         The parameter vector [dp x 1]
%
%   - pats :        Preprocessed pattern in the form of {I, J, IDX, pat}
%
%   - S :           Accumulated covariance values [dp x 1].
%
%                   Let X be the sample matrix, then S is
%                   sum(X(I,:) .* X(J,:), 2) * (1/n)
%
%   - lambda :      The regularization coefficient vector [dp x 1]
%
%   - bw :          The smoothing band-width.
%
%   Returns
%   -------
%   - objv :        The objective function value
%   - grad :        The gradient at phi. [dp x 1]
%
%   Remarks
%   -------
%       This function is to be repeatedly invoked in an optimization 
%       procedure instead of being directly used by an end-user. 
%       There is no argument checking within the function for efficiency.
%


%% main

% compute objective

dp = size(phi, 1);

I = pats{1};
J = pats{2};
IDX = pats{3};
pat = pats{4};
d = size(pat, 1);

if issparse(pat)
    pat = full(pat);
end

V = phi(IDX);

A = sparse(I, J, V, d, d);
v_det = safe_logdet(A);

v_dot = V' * S;

if nargout < 2
    l1_v = pli_approxl1(phi, bw);
else
    [l1_v, l1_deriv] = pli_approxl1(phi, bw);
end

if isscalar(lambda)
    v_reg = lambda * sum(l1_v);
else
    v_reg = lambda' * l1_v;
end

objv = v_dot - v_det + v_reg;

% compute gradient

if nargout >= 2
    
    IA = inv(A);
    if issparse(A)
        IA = full(IA);
    end
    
    g_det = pli_aggregx(dp, IA, pat);    
    g_dot = pli_aggregx(dp, S, IDX);
    g_reg = lambda .* l1_deriv;
    
    grad = g_dot - g_det + g_reg;
end


function v = safe_logdet(A)

[R, p] = chol(A);

if p > 0
    v = -inf;
    return;
end

diagR = diag(R);
if issparse(diagR)
    diagR = full(diagR);
end

v = 2.0 * sum(log(diagR));


