function Y = normalize_exp(X, dim)
%NORMALIZE_EXP Normalizes the exponents
%
%   Y = normalize_exp(X);
%   Y = normalize_exp(X, dim);
%
%       Normalizes exp(X) such that the exponent values sum to ones
%       along the specified dimension. When dim is omitted, it runs
%       along the first non-singleton dimension.
%
%       This function uses a numerically robust way to implement the
%       computation.
%

%% main

if nargin < 2

    if isvector(X)
        Y = exp(X - max(X));
        Y = Y * (1 ./ sum(Y));
    else
        Y = exp(bsxfun(@minus, X, max(X)));
        Y = bsxfun(@times, Y, 1 ./ sum(Y));
    end
    
else
    Y = exp(bsxfun(@minus, X, max(X, [], dim)));
    Y = bsxfun(@times, Y, 1 ./ sum(Y, dim));
end

