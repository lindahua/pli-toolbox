function D = pw_euclidean(X, Y, op)
%PW_SQEUCDIST Pairwise Euclidean distances
%
%   D = pw_euclidean(X)
%       
%       Evaluates pairwise euclidean distances between columns in X.
%
%   D = pw_euclidean(X, Y)
%
%       Evaluates pairwise euclidean distances between columns in X
%       and Y.
%       
%   D = pw_euclidean(X, [], 'sq')
%       
%       Evaluates squared pairwise euclidean distances between columns 
%       in X.
%
%   D = pw_euclidean(X, Y, 'sq')
%
%       Evaluates squared pairwise euclidean distances between columns 
%       in X and Y.
%

%% argument checking

if ~(ismatrix(X) && isreal(X))
    error('pw_eucdist:invalidarg', 'X must be a real matrix.');
end

if nargin < 2
    with_self = 1;
else
    if isempty(Y)
        with_self = 1;
    else
        if ~(ismatrix(Y) && isreal(Y))
            error('pw_eucdist:invalidarg', 'Y must be a real matrix.');
        end
        
        if size(X, 1) ~= size(Y, 1)
            error('pw_eucdist:invalidarg', ...
                'The sizes of X and Y are inconsistent.');
        end
        with_self = 0;
    end
end

if nargin < 3
    sq = 0;
else
    if ~strcmpi(op, 'sq')
        error('pw_eucdist:invalidarg', ...
            'The third input argument is not recognized.');
    end
    sq = 1;
end


%% main

if with_self
    sx = sum(X.^2, 1);
    D = bsxfun(@plus, sx', sx) - 2 * (X' * X);
else
    sx = sum(X.^2, 1);
    sy = sum(Y.^2, 1);
    D = bsxfun(@plus, sx', sy) - 2 * (X' * Y);
end
D(D < 0) = 0;
    
if ~sq
    D = sqrt(D);
end



