function [theta, theta0] = pli_logireg(X, y, w, lambda, lambda0, s0, opts)
%PLI_LOGIREG Solves a logistic regression problem
%
%   Logistic regression is to minimize the following problem w.r.t.
%   w and w0, as
%
%       (lambda/2) * ||theta||^2 + (lambda0/2) * (theta0)^2 + 
%       sum_i loss(theta' * x_i + theta0, y_i).
%
%   Here, the loss function is defined to be
%
%       loss(u, y) = - (y log(p) + (1 - y) log(1 - p)), with
%
%                p = 1 / (1 + exp(-u)).
%
%   Here, y can take value 0 or 1.
%
%   
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda);
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0);
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0, s0);
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0, s0, opts);
%
%       Solves the logistic regression problem.
%
%   Arguments
%   ---------
%   - X :       The sample matrix, of size [d n].
%               Here, d is the sample dimension, and n is the number
%               of samples.
%
%   - y :       The vector of responses, of length n.
%
%   - w :       The sample weights, which can be either
%               - []: indicating all samples have unit weights, or
%               - a vector of length n, to specify one weight for each
%                 observed sample.
%
%   - lambda :  The regularization coefficient for theta.
%
%   - lambda0 : The regularization coefficient for theta0.
%               (It omitted, it is set to 1e-2 * lambda).
%
%               Note, lambda0 can also be set to inf, in which case,
%               theta0 is fixed to be zero.
%
%   - s0 :      The initial solution.
%
%               When lambda0 is finite, it should be a column vector
%               of length (d+1), in the form of [theta; theta0].
%
%               When lambda0 is inf, it should be a d-dimensional vector,
%               which is just for theta.
%
%   - opts :    The options for pli_fminunc to solve the optimization
%               problem. One can use pli_optimset to construct this
%               struct.
%
%
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0, s0, solver);
%
%       One can also supply his own solver to solve the problem instead
%       of using pli_fminunc.
%
%       Here, solver is a function handle for optimization that supports 
%       the following syntax.
%
%           x_opt = solver(f, x)
%

%% Argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_logireg:invalidarg', 'X should be a real matrix.');
end
[d, n] = size(X);

if ~(isfloat(y) && isreal(y) && isvector(y) && numel(y) == n)
    error('pli_logireg:invalidarg', ...
        'y should be a real vector of length n.');
end
if size(y, 1) > 1; y = y.'; end  % y: row

if isempty(w)
    w = [];
else
    if ~(isfloat(w) && isreal(w) && numel(w) == n)
        error('pli_logireg:invalidarg', ...
            'w should be a real vector of length n.');
    end
end
    

if ~(isfloat(lambda) && isreal(lambda) && isscalar(lambda) && lambda > 0)
    error('pli_logireg:invalidarg', ...
        'lambda should be a positive real scalar.');
end

if nargin < 5
    lambda0 = 0.01 * lambda;
else
    if ~(isfloat(lambda0) && isreal(lambda0) && isscalar(lambda0) && lambda0 > 0)
        error('pli_logireg:invalidarg', ...
            'lambda0 should be a positive real scalar.');
    end
end

use_bias = ~isinf(lambda0);
if use_bias
    ds = d + 1;
else
    ds = d;
end

if nargin < 6 || isempty(s0)
    s0 = zeros(ds, 1);
else    
    if ~(isfloat(s0) && isreal(s0) && isequal(size(s0), [ds 1]))
        error('pli_logireg:invalidarg', 'The value for s0 is invalid.e');
    end
end

if nargin < 7
    opts = [];
end


%% main

if use_bias
    lambda = [lambda * ones(d, 1); lambda0];
end

s = pli_genregress(@pli_logitloss, X, y, w, lambda, s0, opts);

if use_bias
    theta = s(1:d);
    theta0 = s(d+1);
else
    theta = s;
    theta0 = 0;
end


