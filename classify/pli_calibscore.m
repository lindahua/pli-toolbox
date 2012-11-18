function c = pli_calibscore(X, L, lambda)
%PLI_CALIBSCORE Prediction score calibration
%
%   c = PLI_CALIBSCORE(X, L, lambda);
%
%       Scores produced by classifiers may not be directly comparable
%       as they may have different magnitudes.
%
%       This function solves a set of calibration coefficients. 
%       These coefficients can be multiplied to scores to make them
%       calibrated.
%
%       Formally, the calibration coefficients are obtained by solving
%       the following problem:
%
%           minimize sum_i (-log p_i) + sum_k r(c(k))
%
%       Here, p_i is defined by 
%
%           exp(x_i(l_i)) / sum_k exp(x_i(k))
%
%       Here, l_i is the class label of the i-th sample. 
%
%       r is a regularization function defined by
%
%           r(c) = lambda * c^2    when c >= 0
%                = inf             when c < 0
%
%   Arguments
%   ---------
%   - X :       The score matrix of size K-by-n
%               Here, K is the number of classes and n is the number
%               of samples.
%
%   - L :       The vector of class labels. (length = n)
%
%   - lambda :  The regularization coefficient.
%
%   
%   Returns
%   -------
%   - c :       The calibration coefficient vector, size = K-by-1.
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_calibscore:invalidarg', 'X should be a real matrix.');
end
[K, n] = size(X);

if ~(isnumeric(L) && isreal(L) && isvector(L) && length(L) == n)
    error('pli_calibscore:invalidarg', ...
        'L should be a vector of length n.');
end
if size(L, 1) > 1; L = L.'; end

if ~(isfloat(lambda) && isreal(lambda) && isscalar(lambda) && lambda > 0)
    error('pli_calibscore:invalidarg', ...
        'lambda should be a positive real value.');
end

%% main

idx = sub2ind(size(X), L, 1:n);
objf = @(c) cs_objfun(X, idx, c, lambda);

opts = pli_optimset('Display', 'off', 'tolfun', 1.0e-12);
c = pli_fminunc(objf, ones(K, 1), opts);


%% objective function

function [objv, grad] = cs_objfun(X, idx, c, lambda)

Y = bsxfun(@times, c, X);
u = pli_logsumexp(Y, 1) - Y(idx);

r = (lambda/2) * (c.^2);
r(c < 0) = inf;

objv = sum(u) + sum(r);

if nargout >= 2
    P = pli_softmax(Y, 1);
    P(idx) = P(idx) - 1;
    grad = sum(P .* X, 2) + lambda * c;
end



