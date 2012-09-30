function D = pw_cityblock(X, Y)
%PW_CITYBLOCK Pairwise cityblock distances
%
%   D = PW_CITYBLOCK(X);
%
%       Evaluates pairwise cityblock distances between columns in X.
%
%   D = PW_CITYBLOCK(X, Y);
%
%       Evaluates pairwise cityblock distances between columns in X and Y.
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pw_cityblock:invalidarg', 'X must be a real matrix.');
end

if nargin < 2 || isempty(Y)
    Y = [];
else
    if ~(ismatrix(Y) && isfloat(Y) && isreal(Y))
        error('pw_cityblock:invalidarg', 'Y must be a real matrix.');
    end
    if size(Y, 1) ~= size(X, 1)
        error('pw_cityblock:invalidarg', ...
            'The dimensions of X and Y are inconsistent.');
    end
end
    
%% main

D = pw_metrics_cimp(int32(1), X, Y);

