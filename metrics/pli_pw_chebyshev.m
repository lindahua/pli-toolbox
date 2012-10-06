function D = pli_pw_chebyshev(X, Y)
%PLI_PW_CHEBYSHEV Pairwise Chebyshev distances
%
%   D = PLI_PW_CHEBYSHEV(X);
%
%       Evaluates pairwise chebyshev distances between columns in X.
%
%   D = PLI_PW_CHEBYSHEV(X, Y);
%
%       Evaluates pairwise chebyshev distances between columns in X and Y.
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pli_pw_chebyshev:invalidarg', 'X must be a real matrix.');
end

if nargin < 2 || isempty(Y)
    Y = [];
else
    if ~(ismatrix(Y) && isfloat(Y) && isreal(Y))
        error('pli_pw_chebyshev:invalidarg', 'Y must be a real matrix.');
    end
    if size(Y, 1) ~= size(X, 1)
        error('pli_pw_chebyshev:invalidarg', ...
            'The dimensions of X and Y are inconsistent.');
    end
end
    
%% main

if ~(isempty(Y) || isa(Y, class(X)))
    Y = cast(Y, class(X));
end

D = pw_metrics_cimp(int32(2), X, Y);

