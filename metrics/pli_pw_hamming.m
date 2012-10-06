function D = pli_pw_hamming(X, Y)
%PLI_PW_HAMMING Pairwise cityblock distances
%
%   D = PLI_PW_HAMMING(X);
%
%       Evaluates pairwise hamming distances between columns in X.
%
%   D = PLI_PW_HAMMING(X, Y);
%
%       Evaluates pairwise hamming distances between columns in X and Y.
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pli_pw_hamming:invalidarg', 'X must be a real matrix.');
end

if nargin < 2 || isempty(Y)
    Y = [];
else
    if ~(ismatrix(Y) && isfloat(Y) && isreal(Y))
        error('pli_pw_hamming:invalidarg', 'Y must be a real matrix.');
    end
    if size(Y, 1) ~= size(X, 1)
        error('pli_pw_hamming:invalidarg', ...
            'The dimensions of X and Y are inconsistent.');
    end
end
    
%% main

if ~(isempty(Y) || isa(Y, class(X)))
    Y = cast(Y, class(X));
end

D = pw_metrics_cimp(int32(3), X, Y);

