function [L, s, objv, mincost, cnts] = pli_kmedoid_c(C, K, varargin)
%PLI_KMEDOID_C K-medoid clustering based on pre-computed cost matrix
%
%   [L, s] = PLI_KMEDOID_C(C, K, ...);
%   [L, s] = PLI_KMEDOID_C(C, s0, ...);
%
%       Performs K-medoid clustering on a pre-computed cost table
%       given by an n x n matrix C.
%
%   [L, s, objv] = PLI_KMEDOID_C(C, K, ...);
%   [L, s, objv, minds] = PLI_KMEDOID_C(C, K, ...);
%   [L, s, objv, minds, cnts] = PLI_KMEDOID_C(C, K, ...);
%       
%       With more output arguments, this function returns additional
%       information about the result.
%
%   Arguments
%   ----------
%   - C :       The pre-computed matrix of size n x n. C(i, j) is the
%               cost of assigning sample j to a cluster with the i-th
%               sample being the center.
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

%% argument checking

if ~(ismatrix(C) && isfloat(C) && isreal(C) && size(C,1) == size(C,2))
    error('pli_kmedoid_c:invalidarg', 'C should be a real square matrix.');
end
n = size(C, 1);

if isvector(K) && isnumeric(K) && isreal(K)
    if isscalar(K) 
        if ~(K > 1 && K == fix(K))
            error('pli_kmedoid_c:invalidarg', ...
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
        error('pli_kmedoid_c:invalidarg', ...
            'The value of K should be less than the number of samples.');
    end
else
    error('pli_kmedoid_c:invalidarg', 'The second argument is invalid.');
end

% parse options

opts.maxiter = 200;
opts.tolfun = 1.0e-8;
opts.display = 'iter';
opts.init = 'kmpp';

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
end

displevel = check_options(opts);


%% main

% initialize centers

if isempty(s0)
    switch opts.init
        case 'kmpp'
            s = pli_kmpp_seed_c(C, K);
        case 'rand'
            s = pli_samplewor(n, K);
        otherwise
            error('pli_kmedoid_c:invalidarg', ...
                'Invalid value for the option init.');
    end
else
    s = s0;
end

[mincost, L] = min(C(s, :), [], 1);
[sL, gb, ge] = pli_idxgroup(K, L);
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
        Ck = C(Ik, Ik);
        [~, i] = min(sum(Ck, 2));
        s(k) = Ik(i);
    end
       
    ch_c = find(s ~= pre_s);
    if ~isempty(ch_c)        
        
        % update labels
                
        [mincost, L] = min(C(s, :), [], 1);
        [sL, gb, ge] = pli_idxgroup(K, L);
        
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

v = s.maxiter;
if ~(isscalar(v) && isnumeric(v) && isreal(v) && v >= 1)
    error('pli_kmedoid_c:invalidarg', ...
        'The value for option maxiter should be a positive integer.');
end

v = s.tolfun;
if ~(isscalar(v) && isfloat(v) && isreal(v) && v >= 0)
    error('pli_kmedoid_c:invalidarg', ...
        'The value for option tolfun should be a non-negative scalar.');
end

v = s.display;
if ~ischar(v)
    error('pli_kmedoid_c:invalidarg', ...
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
        error('pli_kmedoid_c:invalidarg', ...
            'Invalid value for option display.');
end

v = s.init;
if ~ischar(v)
    error('pli_kmedoid_c:invalidarg', ...
        'The value for option init should be a string.');
end

