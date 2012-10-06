function D = pli_pw_cosine(X, Y)
%PLI_PW_COSINE Pairwise cosine distances
%
%   The cosine distance between two vectors x and y is defind to be
%
%       d(x, y) = 1 - (x' * y) / (norm(x) * norm(y))
%
%
%   D = PLI_PW_COSINE(X);
%
%       Evaluates pairwise cosine distances between columns in X.
%
%   D = PLI_PW_COSINE(X, Y);
%
%       Evaluates pairwise cosine distances between columns in X and Y.
%


%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pli_pw_cosine:invalidarg', 'X must be a real matrix.');
end

if nargin < 2
    with_self = 1;
else
    if isempty(Y)
        with_self = 1;
    else
        if ~(ismatrix(Y) && isfloat(Y) && isreal(Y))
            error('pli_pw_cosine:invalidarg', 'Y must be a real matrix.');
        end
        
        if size(X, 1) ~= size(Y, 1)
            error('pli_pw_cosine:invalidarg', ...
                'The sizes of X and Y are inconsistent.');
        end
        with_self = 0;
    end
end

%% main

if with_self
    inv_nx = 1 ./ sqrt(sum(X.^2, 1));    
    D = 1 - (X' * X) .* (inv_nx' * inv_nx);
else
    inv_nx = 1 ./ sqrt(sum(X.^2, 1));
    inv_ny = 1 ./ sqrt(sum(Y.^2, 1));
    D = 1 - (X' * Y) .* (inv_nx' * inv_ny);
end
D(D < 0) = 0;
