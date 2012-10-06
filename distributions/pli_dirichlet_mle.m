function [a, fval, converged] = pli_dirichlet_mle(X, a0, varargin)
%PLI_DIRICHLET_MLE Maximum likelihood estimation of Dirichlet distribution
%
%   a = PLI_DIRICHLET_MLE(X, a0, ...);
%
%       Estimate the parameter of a Dirichlet distribution using 
%       maximum likelihood estimation.
%
%   [a, fval, converged] = PLI_DIRICHLET_MLE(X, a, ...);
%
%       Additionally returns information about the optimization.
%
%   Arguments
%   ---------
%   - X :       The input data.
%   - a0 :      An initial guess of the parameter.
%
%   Returns
%   -------
%   - a :           The estimated parameter.
%   - fval :        The objective value at a.
%   - converged :   Whether the optimization procedure converged.
%
%
%   One can specify options to control the estimation in the form of
%   name/value pairs.
%
%   Options
%   -------
%   - is_log :      When is_log is set to true, it indicates that X
%                   are in logarithm-scale. (default = false)
%
%   - weights :     The sample weights. (default = [])
%
%   - maxiter :     Maximum number of iterations. (default = 100)
%
%   - tolfun :      Tolerance of function value change. (default =1.0e-12)
%
%   - tolx :        Tolerance of soluton change. (default = 1.0e-10)
%
%   - verbose :     Whether to show iteration information. 
%                   (default = false).
%
%   Notes
%   -----
%       A simplified Newton method is used to optimize the parameter. 
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_dirichlet_mle:invalidarg', 'X should be a real matrix.');
end

d = size(X, 1);
n = size(X, 2);

if ~(isfloat(a0) && isreal(a0) && isequal(size(a0), [d 1]))
    error('pli_dirichlet_mle:invalidarg', ...
        'a0 should be a real column vector of length d.');
end

opts = struct('is_log', false, 'weights', [], ...
    'maxiter', 100, 'tolfun', 1.0e-12, 'tolx', 1.0e-10, ...
    'verbose', false);

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
end


%% main

% compute mean-log

if opts.is_log
    L = X;
else
    L = log(X);
end

if size(L, 2) == 1
    ml = L;
else
    w = opts.weights;        
    if isempty(w)
        ml = sum(L, 2) * (1 / size(L, 2));
    else
        if ~(isfloat(w) && isreal(w) && isequal(size(w), [n 1]))
            error('pli_dirichlet_mle:invalidarg', ...
                'weights should be either empty or an n x 1 real vector.');
        end        
        ml = (L * w) * (1 / sum(w));
    end
end
    
% optimize based on mean-log

maxiter = opts.maxiter;
tolfun = opts.tolfun;
tolx = opts.tolx;
verbose = opts.verbose;

t = 0;
converged = false;

a = a0;
[v, dir] = dmle_objfun(ml, a);

while ~converged && t < maxiter
    t = t + 1;
    
    bs = find(dir > 0);
    if isempty(bs)
        c = 1;
    else
        c = min(1, min(a(bs) ./ dir(bs)));
    end
    
    ta = a - c * dir;
    tv = dmle_objfun(ml, ta);
    
    while tv < v
        c = c * 0.6;
        
        if c > 1.0e-12
            ta = a - c * dir;        
            tv = dmle_objfun(ml, ta);
        else
            ta = a;
            tv = v;
        end
    end
    
    a_pre = a;
    a = ta;
    
    v_pre = v;
    [v, dir] = dmle_objfun(ml, a);
    
    if abs(v - v_pre) <= tolfun && max(abs(a - a_pre)) <= tolx
        converged = true;
    end
    
    % debug
    
    if verbose
        fprintf('Iter %3d:  objv = %12g (ch = %12g),  c = %g\n', ...
            t, v, v - v_pre, c);
    end
    
end

fval = v;


%% objective and direction

function [v, dir] = dmle_objfun(ml, a)

a0 = sum(a);

v = (a - 1)' * ml + gammaln(a0) - sum(gammaln(a));

if nargout == 2
    g = ml + psi(a0) - psi(a);

    q = - psi(1, a);
    z = psi(1, a0);
    b = sum(g ./ q) / (1 / z + sum(1 ./ q));
    dir = (g - b) ./ q;
end
    
