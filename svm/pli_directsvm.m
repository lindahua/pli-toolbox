function [w, w0, objv] = pli_directsvm(X, y, c, s0, varargin)
%PLI_DIRECTSVM Linear Support vector machine using direct optimization
%
%   The Linear SVM problem is formulated as an unconstrained optimization
%   problem as follows
%
%       minimize (1/2) * ||w||^2 + sum_i c * L(y_i * (w' * x_i + w0); h)
%
%   Here, L is the approximate hinge loss function defined by
%
%                   0                         (if v > 1 + h)
%       L(v; h) =   (1 + h - v)^2 / (4 h)     (if |1-v| <= h)
%                   1 - h                     (if v < 1 - h)
%   
%   Here, h is the width of the quadratic transition width.
%
%   [w, w0, objv] = PLI_DIRECTSVM(X, y, c);
%   [w, w0, objv] = PLI_DIRECTSVM(X, y, c, s0, ...);
%
%       Solves the coefficient w and offset w0 using for a linear
%       SVM for using direct optimization as described above.
%
%       A newton method is used to find the optimum.
%
%   Arguments
%   ---------
%   X :     The matrix of sample features, size [d n], where
%           d is the sample dimension (#features), and n is 
%           the number of observed samples.
%
%   y :     The response vector of length n. Each entry of y
%           should be either 1 or -1.
%
%   c :     The slack penalty coefficient, which can be either
%           a scalar, or a vector of length n (to specify
%           sample-dependent penalty).
%
%   s0 :    The initial guess of the solution, when should be 
%           a (d+1)-dimensional vector in the form of [w; w0].
%           By default, it is set to zeros(d+1, 1);
%
%
%   One can specify options to control the optimization procedure.
%
%   Options
%   -------
%   maxiter :       The maximum number of iterations. (default = 50)
%
%   tolx :          The tolerable change of solution at convergence.
%                   (default = 1.0e-9)
%
%   tolfun :        The tolerable change of objective value at convergence
%                   (default = 1.0e-8)
%
%   display :       The level of verbosity. 'off' | 'final' | {'iter'}.
%
%   h :             The quadratic transition width (default = 0.1)
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pli_directsvm:invalidarg', 'X should be a real matrix.');
end
[d, n] = size(X);

if ~(isvector(y) && isfloat(y) && isreal(y) && numel(y) == n)
    error('pli_directsvm:invalidarg', ...
        'y should be a real vector of length n.');
end

if ~(isfloat(c) && isreal(c) && ...
        (isscalar(c) || (isvector(c) && numel(c) == n)))
    error('pli_directsvm:invalidarg', ...
        'c should be either a scalar or a vector of length n.');
end

if nargin < 4 || isempty(s0)
    s0 = zeros(d+1, 1);
else
    if ~(isfloat(s0) && isreal(s0) && isequal(size(s0), [d+1, 1]))
        error('pli_directsvm:invalidarg', ...
            's0 should be a real vector of size [d+1, 1].'); 
    end
end

opts.maxiter = 50;
opts.tolx = 1.0e-9;
opts.tolfun = 1.0e-8;
opts.display = 'iter';
opts.h = 0.1;

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
    check_options(opts);
end

switch opts.display
    case 'off'
        displevel = 0;
    case 'final'
        displevel = 1;
    case 'iter'
        displevel = 2;
    otherwise
        error('pli_directsvm:invalidarg', ...
            'The value for the display option is invalid.');
end

%% main

maxiter = opts.maxiter;
tolx = opts.tolx;
tolf = opts.tolfun;

t = 0;
converged = false;

while ~converged && t < maxiter
    t = t + 1;
    
    
    
    
end


%% Evaluation

function [v, g, H] = objfun(X, y, c, h, s)

d = size(X, 1);
w = s(1:d);
w0 = s(d+1);

u = y .* (w' * X + w0);

sa = find(u < 1+h);

if isempty(sa)
    g = [w; 0];
else
    a = (u(sa) - (1+h)) * (1/(2*h));
    cay = a .* c(sa) .* y(sa);
    g = [w + X(:, sa) * cay'; sum(cay)];
end



%% Auxiliary functions

function check_options(opts)

v = opts.maxiter;
if ~(isreal(v) && isscalar(v) && v >= 1)
    error('pli_directsvm:invalidarg', ...
        'The value for maxiter should be a positive integer.');
end

v = opts.tolx;
if ~(isreal(v) && isscalar(v) && v > 0)
    error('pli_directsvm:invalidarg', ...
        'The value for tolx should be a positive real scalar.');
end

v = opts.tolfun;
if ~(isreal(v) && isscalar(v) && v > 0)
    error('pli_directsvm:invalidarg', ...
        'The value for tolfun should be a positive real scalar.');
end

v = opts.h;
if ~(isreal(v) && isscalar(v) && v > 0 && v < 1)
    error('pli_directsvm:invalidarg', ...
        'The value for h should be a real value in (0, 1).');
end

if ~ischar(opts.display)
    error('pli_directsvm:invalidarg', ...
        'The value for display should be a string.');
end

