function D = pw_hamming(X, Y)
%PW_CITYBLOCK Pairwise cityblock distances
%
%   D = PW_HAMMING(X);
%
%       Evaluates pairwise hamming distances between columns in X.
%
%   D = PW_HAMMING(X, Y);
%
%       Evaluates pairwise hamming distances between columns in X and Y.
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pw_hamming:invalidarg', 'X must be a real matrix.');
end

if nargin < 2 || isempty(Y)
    Y = [];
else
    if ~(ismatrix(Y) && isfloat(Y) && isreal(Y))
        error('pw_hamming:invalidarg', 'Y must be a real matrix.');
    end
    if size(Y, 1) ~= size(X, 1)
        error('pw_hamming:invalidarg', ...
            'The dimensions of X and Y are inconsistent.');
    end
end
    
%% main

D = pw_metrics_cimp(int32(3), X, Y);

