function D = pli_pw_mahdist(X, Y, C, op)
%PLI_PW_MAHDIST Pairwise Mahalanobis distances
%
%   D = PLI_PW_MAHDIST(X, [], C);
%       
%       Evaluates pairwise Mahalanobis distances between the columns
%       in X, with respect to C, a given covariance.
%
%   D = PLI_PW_MAHDIST(X, Y, C);
%
%       Evaluates pairwise Mahalanobis distances between the columns in 
%       X and Y. When Y is empty, it computes the 
%
%   D = PLI_PW_MAHDIST(X, [], C, 'sq');
%   D = PLI_PW_MAHDIST(X, Y, C, 'sq');
%
%       Evaluates squared pairwise Mahalanobis distances.
%
%   Arguments
%   ---------
%   - X, Y :    The input matrices.
%   - C :       The covariance matrix, in full or compact form.
%

%% argument checking

if ~(ismatrix(X) && isreal(X))
    error('pli_pw_mahdist:invalidarg', 'X should be a real matrix.');
end

if isempty(Y)
    with_self = 1;
else
    with_self = 0;
    if ~(ismatrix(Y) && isreal(Y))
        error('pli_pw_mahdist:invalidarg', 'Y must be a real matrix.');
    end
end

[~, f] = pli_chksmat(C);

if nargin < 4
    sq = 0;
else
    if ~strcmpi(op, 'sq')
        error('pli_pw_mahdist:invalidarg', ...
            'The third input argument is not recognized.');
    end
    sq = 1;
end

%% main

if f == 0
    Q = 1.0 / C;
    if with_self
        sxx = Q * sum(X.^2, 1);       
        XQX = Q * (X' * X);
    else
        sxx = Q * sum(X.^2, 1);
        syy = Q * sum(Y.^2, 1);
        XQY = Q * (X' * Y);
    end    
    
elseif f == 1
    Q = 1.0 ./ C;
    if with_self
        QX = bsxfun(@times, Q, X);
        sxx = sum(X .* QX, 1);
        XQX = X' * QX;
    else
        QX = bsxfun(@times, Q, X);
        QY = bsxfun(@times, Q, Y);
        sxx = sum(X .* QX, 1);
        syy = sum(Y .* QY, 1);
        XQY = X' * QY;
    end    
    
else % f == 2
    if with_self
        QX = C \ X;
        sxx = sum(X .* QX, 1);
        XQX = X' * QX;        
    else
        QX = C \ X;
        QY = C \ Y;
        sxx = sum(X .* QX, 1);
        syy = sum(Y .* QY, 1);
        XQY = X' * QY;
    end
    
end

if with_self    
    D = bsxfun(@plus, sxx', sxx) - 2 * XQX;
else
    D = bsxfun(@plus, sxx', syy) - 2 * XQY;
end
D(D < 0) = 0;

if ~sq
    D = sqrt(D);
end

