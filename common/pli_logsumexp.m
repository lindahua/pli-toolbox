function R = pli_logsumexp(X, dim)
%PLI_LOGSUMEXP Robust evaluation of log-sum-exp
%
%   R = PLI_LOGSUMEXP(X);
%   R = PLI_LOGSUMEXP(X, dim);
%
%       Evaluates the log-sum-exp along the first non-singleton dimension.
%
%       This function is functionally equivalent to log(sum(exp(X)). But
%       a careful implementation is used to prevent overflow.
%

%% argument checking

if ~(isfloat(X) && ismatrix(X) && isreal(X))
    error('pli_logsumexp:invalidarg', 'X should be a real matrix.');
end

%% main

if nargin < 2 
    if isvector(X)
        dim = 0;
    else
        dim = 1;
    end
end


if dim == 0
    mX = max(X);
    R = mX + log(sum(exp(X - mX)));    
else
    mX = max(X, [], dim);
    Y = bsxfun(@minus, X, mX);
    R = mX + log(sum(exp(Y), dim));
end


