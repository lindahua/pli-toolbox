function D = pw_qdist(X, Y, Q, op)
%PW_QDIST Pairwise quadratic distances
%
%   D = PW_QDIST(X, [], Q);
%       
%       Evaluates pairwise quadratic distances between the columns
%       in X.
%
%   D = PW_QDIST(X, Y, Q);
%
%       Evaluates pairwise quadraic distances between the columns in 
%       X and Y. When Y is empty, it computes the 
%
%   D = PW_QDIST(X, [], Q, 'sq');
%   D = PW_QDIST(X, Y, Q, 'sq');
%
%       Evaluates squared pairwise quadratic distances.
%
%   Arguments
%   ---------
%   - X, Y :    The input matrices.
%   - Q :       The quadratic matrix, in full or compact form.
%

%% argument checking

if ~(ismatrix(X) && isreal(X))
    error('pw_qdist:invalidarg', 'X should be a real matrix.');
end

if isempty(Y)
    with_self = 1;
else
    with_self = 0;
    if ~(ismatrix(Y) && isreal(Y))
        error('pw_qdist:invalidarg', 'Y must be a real matrix.');
    end
end

[~, f] = check_smat(Q);

if nargin < 4
    sq = 0;
else
    if ~strcmpi(op, 'sq')
        error('pw_qdist:invalidarg', ...
            'The third input argument is not recognized.');
    end
    sq = 1;
end
    
%% main

if f == 0
    if with_self
        sxx = Q * sum(X.^2, 1);       
        XQX = Q * (X' * X);
    else
        sxx = Q * sum(X.^2, 1);
        syy = Q * sum(Y.^2, 1);
        XQY = Q * (X' * Y);
    end    
    
elseif f == 1
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
        QX = Q * X;
        sxx = sum(X .* QX, 1);
        XQX = X' * QX;        
    else
        QX = Q * X;
        QY = Q * Y;
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



