function [L, s, objv, mincost, cnts] = kmedoid(X, K, varargin)
%KMEDOID k-Medoid clustering
%
%   [L, s] = KMEDOID(X, K, ...);
%   [L, s] = KMEDOID(x, s0, ...);
%
%       Performs K-medoid clustering on the samples given by X. 
%       By default the clustering is based on squared euclidean
%       distances.
%
%   [L, s, objv] = KMEDOID(X, K, ...);
%   [L, s, objv, minds] = KMEDOID(X, K, ...);
%   [L, s, objv, minds, cnts] = KMEDOID(X, K, ...);
%       
%       With more output arguments, this function returns additional
%       information about the result.
%
%   Arguments
%   ----------
%   - X :       The sample matrix (size [d n]). Each column is a 
%               d-dimensional vector.
%
%   - K :       The number of centers
%
%   - s0 :      The initial selection of centers. It is a vector 
%               of length K.
%
%   Returns
%   -------
%   - L :       The vector of resultant labels, size [1 n].
%
%   - s :       The list of the indices of the samples chosen as centers.
%               size [1 K]
%
%   - objv :    The total cost at the final step.
%
%   - mincost : The cost of assigning each sample to its closest center.
%               size [1 n]. 
%
%               By default the cost equals the squared distance. One can 
%               change the cost function by setting the option costfun.
%
%   - cnts :    The number of samples assigned to each center.
%               size [1 K].
%
%   Options
%   -------
%   - costfun :     The cost function. 
%
%                   costfun(X, Y) should return an m x n matrix when
%                   X and Y respectively have m and n columns.
%
%                   By default, the squared Euclidean distance is used
%                   as the cost.
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

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X) && ~isempty(X))
    error('kmedoid:invalidarg', 'X should be a real matrix.');
end
n = size(X, 2);

if isvector(K) && isnumeric(K) && isreal(K)
    if isscalar(K) 
        if ~(K > 1 && K == fix(K))
            error('kmedoid:invalidarg', ...
                'K should be a positive integer with K > 1.');
        end
        s0 = [];
    else
        s0 = K;
        K = numel(s0);
        
        if size(s0, 1) > 1  % make it a row
            s0 = s0.';
        end
    end
    if K >= n
        error('kmedoid:invalidarg', ...
            'The value of K should be less than the number of samples.');
    end
else
    error('kmedoid:invalidarg', 'The second argument is invalid.');
end

% parse options

opts.costfun = [];
opts.maxiter = 200;
opts.tolfun = 1.0e-8;
opts.display = 'iter';
opts.init = 'kmpp';

if ~isempty(varargin)
    opts = parse_opts(opts, varargin);
end

displevel = check_options(opts);
cfun = opts.costfun;

%% main

% Initialize

% initialize centers

if isempty(s0)
    switch opts.init
        case 'kmpp'
            s = kmpp_seed(X, K, cfun);
        case 'rand'
            s = sample_wor(n, K);
        otherwise
            error('kmedoid_std:invalidarg', ...
                'Invalid value for the option init.');
    end
else
    s = s0;
end

% initialize labels

if isempty(cfun)
    cfun = @(x, y) pw_euclidean(x, y, 'sq');
end

costs = cfun(X(:,s), X);
[mincost, L] = min(costs, [], 1);
[sL, gb, ge] = idxgroup(K, L);

objv = sum(mincost);


% Iterative update

maxiter = opts.maxiter;
tolfun = opts.tolfun;

t = 0;
converged = false;

% print header

if displevel >= 2
    fprintf('%-6s %15s %15s %12s %12s\n', ...
        'Iters', 'obj.value', 'obj.change', ...
        'centers.#ch', 'labels.#ch');
end

while ~converged && t < maxiter
    t = t + 1;
    
    pre_s = s;
    pre_L = L;
    pre_objv = objv;
    
    % update centers
    
    for k = 1 : K
        Ik = sL(gb(k):ge(k));
        Xk = X(:,Ik);
        Ck = cfun(Xk, Xk);
        [~, i] = min(sum(Ck, 2));
        s(k) = Ik(i);
    end
       
    ch_c = find(s ~= pre_s);
    if ~isempty(ch_c)
        
        % update costs
        
        costs(ch_c, :) = cfun(X(:, s(ch_c)), X);
        
        % update labels
                
        [mincost, L] = min(costs, [], 1);
        [sL, gb, ge] = idxgroup(K, L);
        
        objv = sum(mincost);
        
        ch_l = sum(L ~= pre_L);
        ch_v = objv - pre_objv;
    else
        ch_l = 0;
        ch_v = 0;
    end
                        
    % determine convergence
        
    converged = (ch_l == 0) || (abs(ch_v) <= tolfun);   
    
    % print iteration info
            
    if displevel >= 2
        fprintf('%-6d %15g %15g %12d %12d\n', ...
            t, objv, ch_v, numel(ch_c), ch_l);
    end
    
end

cnts = double((ge - gb + 1).');


%% Auxiliary functions

function displevel = check_options(s)

v = s.costfun;
if ~(isempty(v) || isa(v, 'function_handle'))
    error('kmedoid:invalidarg', ...
        'The value for option costfun should be a function handle.');
end

v = s.maxiter;
if ~(isscalar(v) && isnumeric(v) && isreal(v) && v >= 1)
    error('kmedoid:invalidarg', ...
        'The value for option maxiter should be a positive integer.');
end

v = s.tolfun;
if ~(isscalar(v) && isfloat(v) && isreal(v) && v >= 0)
    error('kmedoid:invalidarg', ...
        'The value for option tolfun should be a non-negative scalar.');
end

v = s.display;
if ~ischar(v)
    error('kmedoid:invalidarg', ...
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
        error('kmedoid:invalidarg', ...
            'Invalid value for option display.');
end

v = s.init;
if ~ischar(v)
    error('kmedoid:invalidarg', ...
        'The value for option init should be a string.');
end

