function [L, C, objv, min_ds, cnts] = pli_kmeans(X, K, varargin)
%PLI_KMEANS Standard K-means clustering
%
%   [L, C] = PLI_KMEANS(X, K, ...);
%   [L, C] = PLI_KMEANS(X, C0, ...);
%
%       Performs standard K-means on a set of vectors, and returns 
%       the resultant labels in L, and the centers in C.
%
%   [L, C, objv] = PLI_KMEANS( ... );
%   [L, C, objv, min_ds] = PLI_KMEANS( ... );
%   [L, C, objv, min_ds, cnts] = PLI_KMEANS( ... );
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
%   - repeats:      The number of times repeating the procedure.
%                   The best solution is output eventually.
%                   (default = 1).
%
%   - weights:      The weights of data points, which should be
%                   a vector of length n.
%                   (default = [], indicating that all samples
%                   have the same unit weight).
%
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
%       PLI_KMEANS is not as heavy as the kmeans function in the 
%       statistics toolbox, and it is more efficient by using a 
%       faster vectorized method to calculate pairwise distances.
%       

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X) && ~isempty(X))
    error('pli_kmeans:invalidarg', 'X should be a real matrix.');
end
[d, n] = size(X);

if ismatrix(K) && isnumeric(K) && isreal(K)
    if isscalar(K) 
        if ~(K > 1 && K == fix(K))
            error('pli_kmeans:invalidarg', ...
                'K should be a positive integer with K > 1.');
        end
        C0 = [];
    else
        C0 = K;
        if size(C0, 1) ~= d
            error('pli_kmeans:invalidarg', ...
                'The dimensions of X and C0 are inconsistent.');
        end
        K = size(C0, 2);
    end
    if K >= n
        error('pli_kmeans:invalidarg', ...
            'The value of K should be less than the number of samples.');
    end
else
    error('pli_kmeans:invalidarg', 'The second argument is invalid.');
end

% parse options

opts.repeats = 1;
opts.weights = [];
opts.maxiter = 200;
opts.tolfun = 1.0e-8;
opts.display = 'iter';
opts.init = 'kmpp';

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
end

displevel = check_options(opts, n);

w = opts.weights;
if ~isempty(w)
    if size(w, 1) > 1 % turn to a row vector
        w = w.';
    end
end

if opts.repeats > 1
    if ~isempty(C0)
        error('pli_kmeans:invalidarg', ...
            'Cannot set more than one repeats when C0 is given.');
    end
end


%% main

[L, C, objv, min_ds, cnts] = do_kmeans(X, K, C0, w, displevel, opts);

if opts.repeats > 1
    
    for ri = 2 : opts.repeats
        [tL, tC, tobjv, tmin_ds, tcnts] = do_kmeans(X, K, C0, w, displevel, opts);
        
        if tobjv < objv
            L = tL;
            C = tC;
            objv = tobjv;
            min_ds = tmin_ds;
            cnts = tcnts;
        end
    end
end


%% Core function

function [L, C, objv, min_ds, cnts] = do_kmeans(X, K, C0, w, displevel, opts)

% Initialize

% initialize centers

if isempty(C0)
    switch opts.init
        case 'kmpp'
            C = X(:, pli_kmpp_seed(X, K));
        case 'rand'
            C = X(:, pli_samplewor(n, K));
        otherwise
            error('pli_kmeans:invalidarg', ...
                'Invalid value for the option init.');
    end
else
    C = C0;
end

% initialize labels

D = pli_pw_euclidean(C, X, 'sq');
[min_ds, L] = min(D, [], 1);
cnts = double(pli_intcount(K, L).');

if isempty(w)
    objv = sum(min_ds);
else
    objv = min_ds * w';
end


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
    
    if isempty(w)
        S = pli_aggregx(K, X, L);
    else
        S = pli_aggregx(K, bsxfun(@times, X, w), L);
    end
        
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
    
    if isempty(w)
        cw = cnts;
    else
        cw = pli_aggregx(K, w, L).';
    end
    
    if u_all
        C = bsxfun(@times, S, 1 ./ cw);
        D = pli_pw_euclidean(C, X, 'sq');
    else
        Cu = bsxfun(@times, S(:, ui), 1 ./ cw(ui));
        C(:, ui) = Cu;
        D(ui, :) = pli_pw_euclidean(Cu, X, 'sq');
        
        if ~isempty(ri)
            Cr = X(:, pli_samplewor(size(X,2), numel(ri)));
            C(:, ri) = Cr;
            D(ri, :) = pli_pw_euclidean(Cr, X, 'sq');
        end
    end
                    
    % update labels
        
    [min_ds, L] = min(D, [], 1);
    cnts = double(pli_intcount(K, L).');
    
    if isempty(w)
        objv = sum(min_ds);
    else
        objv = min_ds * w';
    end
    
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

function displevel = check_options(s, n)

v = s.repeats;
if ~(isscalar(v) && isreal(v) && v == fix(v) && v >= 1)
    error('pli_kmeans:invalidarg', ...
        'The value for repeats should be a positive integer.');
end

v = s.weights;
if ~isempty(v)
    if ~(isfloat(v) && isreal(v) && isvector(v) && numel(v) == n)
        error('pli_kmeans:invalidarg', ...
            'The weights should be a real vector of length n.');
    end
end

v = s.maxiter;
if ~(isscalar(v) && isnumeric(v) && isreal(v) && v >= 1)
    error('pli_kmeans:invalidarg', ...
        'The value for option maxiter should be a positive integer.');
end

v = s.tolfun;
if ~(isscalar(v) && isfloat(v) && isreal(v) && v >= 0)
    error('pli_kmeans:invalidarg', ...
        'The value for option tolfun should be a non-negative scalar.');
end

v = s.display;
if ~ischar(v)
    error('pli_kmeans:invalidarg', ...
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
        error('pli_kmeans:invalidarg', ...
            'Invalid value for option display.');
end

v = s.init;
if ~ischar(v)
    error('pli_kmeans:invalidarg', ...
        'The value for option init should be a string.');
end

        
        
        
        