function [L, C, objv, min_ds, cnts] = kmeans_std(X, K, varargin)
%KMEANS_STD Standard K-means clustering
%
%   [L, C] = KMEANS_STD(X, K, ...);
%   [L, C] = KMEANS_STD(X, C0, ...);
%
%       Performs standard K-means on a set of vectors, and returns 
%       the resultant labels in L, and the centers in C.
%
%   [L, C, objv] = KMEANS_STD( ... );
%   [L, C, objv, min_ds] = KMEANS_STD( ... );
%   [L, C, objv, min_ds, cnts] = KMEANS_STD( ... );
%
%       With more output arguments, this function returns additional
%       information about the result.
%
%
%   Arguments
%   ---------
%   - X :       The matrix of samples, of size [d n].
%               Each column is a d-dimensional sample vector.
%       
%   - K :       The number of centers. K should be a positive integer
%               with 2 <= K < n.
%
%   - C0 :      The matrix of initial centers, of size [d K].
%
%   Returns
%   -------
%   - L :       The vector of result labels. (size is [1 n]).
%
%   - C :       The matrix of resultant centers. (size is [d K]).
%
%   - objv :    The objective value at final step, which equals
%               sum(min_ds).
%
%   - min_ds :  The vector of squared distances from each sample to the
%               closest center. (size is [1 n]).
%
%   - cnts :    The number of samples assigned to each cluster.
%               (size is [1 K]).
%
%   
%   One can customize the procedure using following options in form
%   of name/value pairs.
%
%   Options
%   -------
%   - maxiter :     The maximum number of iterations.
%                   (default = 200)
%
%   - tolfun :      The tolerance of obvjective function changes
%                   at convergence. (default = 1.0e-8)
%
%   - display :     The level of information display. 
%                   'off' | 'final' | {'iter'}.
%
%   - init :        The method to initialize the centers. It can take
%                   either of the following values
%                   - 'kmpp' : Using Kmeans++ method (call kmpp_seeds)
%                   - 'rand' : Randomly select K samples as seeds.
%                   default = 'kmpp'.
%
%   Remarks
%   -------
%       KMEANS_STD is not as heavy as the kmeans function in the 
%       statistics toolbox, and it is more efficient by using a 
%       faster vectorized method to calculate pairwise distances.
%       

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X) && ~isempty(X))
    error('kmeans_std:invalidarg', 'X should be a real matrix.');
end
[d, n] = size(X);

if ismatrix(K) && isnumeric(K) && isreal(K)
    if isscalar(K) 
        if ~(K > 1 && K == fix(K))
            error('kmeans_std:invalidarg', ...
                'K should be a positive integer with K > 1.');
        end
        C0 = [];
    else
        C0 = K;
        if size(C0, 1) ~= d
            error('kmeans_std:invalidarg', ...
                'The dimensions of X and C0 are inconsistent.');
        end
        K = size(C0, 2);
    end
    if K >= n
        error('kmeans_std:invalidarg', ...
            'The value of K should be less than the number of samples.');
    end
else
    error('kmeans_std:invalidarg', 'The second argument is invalid.');
end

% parse options

opts.maxiter = 200;
opts.tolfun = 1.0e-8;
opts.display = 'iter';
opts.init = 'kmpp';

if ~isempty(varargin)
    opts = parse_opts(opts, varargin);
end

displevel = check_options(opts);


%% main

% Initialize

% initialize centers

if isempty(C0)
    switch opts.init
        case 'kmpp'
            C = X(:, kmpp_seed(X, K));
        case 'rand'
            C = X(:, sample_wor(n, K));
        otherwise
            error('kmeans_std:invalidarg', ...
                'Invalid value for the option init.');
    end
else
    C = C0;
end

% initialize labels

D = pw_euclidean(C, X, 'sq');
[min_ds, L] = min(D, [], 1);
cnts = double(intcount(K, L).');

objv = sum(min_ds);


% Iterative update

maxiter = opts.maxiter;
tolfun = opts.tolfun;

t = 0;
converged = false;

% print header

if displevel >= 2
    fprintf('%-6s %15s %15s %12s %12s\n', ...
        'Iters', 'obj.value', 'obj.change', ...
        'labels.#ch', 'clus.#ch');
end

while ~converged && t < maxiter
    t = t + 1;
    
    pre_L = L;
    pre_objv = objv;
    
    % update centers and distances
    
    S = aggregx(K, X, L);
        
    if t == 1 || all(aff_c)
        if all(cnts > 0)
            u_all = 1;        
        else
            u_all = 0;
            ui = find(cnts > 0);
            ri = find(cnts == 0);
        end
    else
        u_all = 0;
        ui = find(cnts > 0 & aff_c);
        ri = find(cnts == 0 & aff_c);
    end        
    
    if u_all
        C = bsxfun(@times, S, 1 ./ cnts);
        D = pw_euclidean(C, X, 'sq');
    else
        Cu = bsxfun(@times, S(:, ui), 1 ./ cnts(ui));
        C(:, ui) = Cu;
        D(ui, :) = pw_euclidean(Cu, X, 'sq');
        
        if ~isempty(ri)
            Cr = X(:, sample_wor(n, numel(ri)));
            C(:, ri) = Cr;
            D(ri, :) = pw_euclidean(Cr, X, 'sq');
        end
    end
                    
    % update labels
        
    [min_ds, L] = min(D, [], 1);
    cnts = double(intcount(K, L).');
    
    objv = sum(min_ds);
    
    % determine affected clusters
    
    ch_l = find(L ~= pre_L);
    aff_c = false(1, K);
    if ~isempty(ch_l)        
        aff_c(L(ch_l)) = 1;
        aff_c(pre_L(ch_l)) = 1;
    end
    
    % determine convergence
    
    ch_l = numel(ch_l);
    ch_v = objv - pre_objv;
    
    converged = (ch_l == 0) || (abs(ch_v) <= tolfun);   
    
    % print iteration info
    
    if displevel >= 2
        fprintf('%-6d %15g %15g %12d %12d\n', ...
            t, objv, ch_v, ch_l, sum(aff_c));
    end
    
end
     
if displevel >= 1
    if converged
        fprintf('K-means converged at t = %d (objv = %g)\n', ...
            t, objv);
    else
        fprintf('K-means terminated at t = %d (objv = %g): NOT converged.\n', ...
            t, objv);
    end
end



%% Auxiliary functions

function displevel = check_options(s)

v = s.maxiter;
if ~(isscalar(v) && isnumeric(v) && isreal(v) && v >= 1)
    error('kmean_std:invalidarg', ...
        'The value for option maxiter should be a positive integer.');
end

v = s.tolfun;
if ~(isscalar(v) && isfloat(v) && isreal(v) && v >= 0)
    error('kmean_std:invalidarg', ...
        'The value for option tolfun should be a non-negative scalar.');
end

v = s.display;
if ~ischar(v)
    error('kmeans_std:invalidarg', ...
        'The value for option display should be a string.');
end

switch lower(v)
    case 'off'
        displevel = 0;
    case 'final'
        displevel = 1;
    case 'iter'
        displevel = 2;
    otherwise
        error('kmeans_std:invalidarg', ...
            'Invalid value for option display.');
end

v = s.init;
if ~ischar(v)
    error('kmeans_std:invalidarg', ...
        'The value for option init should be a string.');
end

        
        
        
        