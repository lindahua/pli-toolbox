function [P, mu, vars, res] = prin_comp(X, k, varargin)
%PRIN_COMP Principal component analysis
%
%   [P, mu, vars, res] = PRIN_COMP(X, k, ...);
%
%       Performs principal component analysis on a set of data given 
%       as columns of X. It solves the first k principal components.
%
%   [P, mu, vars, res] = PRIN_COMP(X, p, ...);
%
%       Performs principal component analysis on X, and solves the
%       first k principal components. Here, k is determined such that
%       the principal subspace preserves at least ratio p of the
%       total variance. (Here, p < 1.0)
%
%   Arguments
%   ---------
%   - X :       The data matrix of size [d n]. Each column is a sample.
%
%   - k :       The number of principal components.
%
%   - p :       The ratio of variance to be preserved.
%
%  
%   Returns
%   -------
%   - P :       The matrix of principal eigenvectors, size = [d k].
%   
%   - mu :      The mean vector of the input samples.
%
%   - vars :    The variances of the principal components, size = [1 k].
%
%   - res :     The residual variance.
%
%   Note: you don't have to use all output arguments when calling this 
%   function. The function only performs the computation enough to
%   return the desired outputs.
%
%   One can further specify the following options in the form of
%   name/value pairs to customize the analysis.
%
%   Options
%   -------
%   - centered :        Whether the input samples have been centered.
%                       Note: if 'centered' is set to true, then the
%                       output argument mu will be a zero scalar.
%                       (default = false).
%
%   - weights :         The sample weights, which should be a vector
%                       of length n. (default = []).
%
%   - method :          The method used to perform the analysis, which
%                       can be either of the following:
%
%                       - 'svd':    Use SVD on X.
%
%                       - 'cov':    Use eigen-analysis on the covariance
%                                   (i.e. (1/n) * (X * X'))
%
%                       - 'tcov':   Use eigen-analysis on covariance of
%                                   the transposed matrix 
%                                   (i.e. (1/n) * (X' * X))
%                                   The results will be converted back
%                                   to the original space. 
%                                   It is efficient when n < d.
%
%                       Default = 'svd'.
%


%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X) && ~isempty(X))
    error('prin_comp:invalidarg', ...
        'X should be a non-empty real matrix.');
end
[d, n] = size(X);

if ~(isscalar(k) && isreal(k) && k > 0)
    error('prin_comp:invalidarg', ...
        'The second argument is invalid.');
end

if k >= 1    
    if k ~= fix(k)
        error('prin_comp:invalidarg', 'k should be a positive integer.');
    end
    if k >= min(d, n)
        error('prin_comp:invalidarg', ...
            'k should be less than min(d, n).');
    end
end
    
opts.centered = false;
opts.weights = [];
opts.method = 'svd';

if ~isempty(varargin)
    opts = parse_opts(opts, varargin);
    check_options(opts, n);
end

%% main

% pre-process weights

w = opts.weights;
if isempty(w)
    tw = n;
else
    if size(w, 1) > 1  % turn to row
        w = w.';
    end
    tw = sum(w);
end

% center data

if opts.centered 
    mu = 0;
else
    if isempty(w)
        mu = sum(X, 2) * (1/n);
    else
        mu = (X * w') * (1/tw);
    end
    X = bsxfun(@minus, X, mu);
end

% re-weight data

if ~isempty(w)
    X = bsxfun(@times, X, sqrt(w));
end


% perform analysis

switch lower(opts.method)
    case 'svd'
        [U, S, ~] = svd(X, 'econ');
        evs = diag(S).^2;
        k = decide_k(evs, k);
                
    case 'cov'
        C = X * X';
        [U, evs] = symeig(C);
        k = decide_k(evs, k);        
        
    case 'tcov'
        Q = X' * X;
        [V, evs] = symeig(Q);
        k = decide_k(evs, k);
        U = X * V(:, 1:k);
        U = bsxfun(@times, U, 1 ./ sqrt(sum(U.^2, 1)));  % normalize
        
    otherwise
        error('prin_comp:invalidarg', ...
            'Unknown method: %s', opts.method);
end

% produce outputs

if size(U, 2) > k
    P = U(:, 1:k);
else
    P = U;
end
evs = evs * (1/tw);

if nargout >= 3
    vars = evs(1:k);
end

if nargout >= 4
    res = sum(evs(k+1:end));
end


%% Auxiliary function

function k = decide_k(evs, k)

if k < 1
    p = k;
    cs = cumsum(evs);
    s = cs(end);
    k = find(cs >= p * s, 1);
end



function check_options(opts, n)

v = opts.centered;
if ~(isscalar(v) && (isnumeric(v) || islogical(v)))
    error('prin_comp:invalidarg', ...
        'The value for option centered should be either true or false.');
end

v = opts.weights;
if ~isempty(v)
    if ~(isreal(v) && isfloat(v) && isvector(v) && length(v) == n)
        error('prin_comp:invalidarg', ...
            'weights should be a real vector of length n.');
    end
end

v = opts.method;
if ~ischar(v)
    error('prin_comp:invalidarg', ...
        'The value for option method should be a string.');
end


    
    
    

