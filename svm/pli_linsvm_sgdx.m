function [w, w0, objv] = pli_linsvm_sgdx(X, y, lambda, lambda0, varargin)
%PLI_LINSVM_SGDX Extended Stochastic Gradient Descent for Linear SVM
%
%   [w, w0, objv] = PLI_LINSVM_SGDX(X, y, lambda, lambda0, ...);
%
%       Trains a linear support vector machine using a stochastic
%       gradient descent (SGD) method. 
%
%       Several well-known methods are implemented in this function,
%       which include a Pegasos and SGD-QN. (More algorithms will be
%       incorporated in future versions).
%
%   Arguments
%   ---------
%   - X :           The sample feature matrix, size [d n].
%                   Here, d is the feature dimension, and 
%                   n is the number of samples.
%
%   - y :           The vector of sample outputs, of length n.
%                   Each value of y must be either 1 or -1).
%
%   - lambda :      The regularization coefficient for w.
%
%   - lambda0 :     The regularization coefficient for w0 (bias).
%                   lambda0 can be set to inf, in which case, bias
%                   is fixed to 0.
%   
%   Returns
%   -------
%   - w :           The coefficient vector of size [d 1].
%
%   - w0 :          The bias term.
%
%   - objv :        The objective value at the end of each epoch.
%                   If there are m epoches, objv will be a vector
%                   of length m.
%
%                   Note objective values will only be evaluated
%                   when the third output argument is desired.
%
%   One can specify other options in the form of name/value pairs.
%
%   Options
%   -------
%   - algorithm :   The algorithm to use. {'pegasos'} | 'sgd-qn'
%
%   - T :           The number of iterations to run. 
%                   (default = 100 / lambda).
%
%   - k :           The number of samples processed in each iteration.
%                   (default = 1).
%
%   - epoches :     The number of epoches. In each epoch, T iterations
%                   will be run, and objective value will be evaluated
%                   at the end of each epoch.
%                   (default = 1).
%
%   - skip :        The interval (in terms of the number of iterations)
%                   between evaluation of B. (default = 1).
%
%   - w_init :      Initial coefficient vector. (default = zeros(d,1)).
%
%   - w0_init :     Initial bias value (default = 0).
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pli_linsvm_sgdx:invalidarg', 'X should be a real matrix.');
end
[d, n] = size(X);

if ~(isvector(y) && isfloat(y) && isreal(y))
    error('pli_linsvm_sgdx:invalidarg', 'y should be a real vector.');
end

if length(y) ~= n
    error('pli_linsvm_sgdx:invalidarg', ...
        'The sizes of X and y are inconsistent.');
end

if ~(isfloat(lambda) && isreal(lambda) && lambda > 0 && isfinite(lambda))
    error('pli_linsvm_sgdx:invalidarg', ...
        'lambda should be a positive scalar.');
end

if ~(isfloat(lambda0) && isreal(lambda0) && lambda0 >= 0)
    error('pli_linsvm_sgdx:invalidarg', ...
        'lambda0 should be a non-negative scalar or inf.');
end

opts.algorithm = 'pegasos';
opts.T = 100 / lambda;
opts.k = 1;
opts.skip = 1;
opts.epoches = 1;
opts.w_init = [];
opts.w0_init = [];

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
    check_options(opts, d);
end

%% main

T = opts.T;
k = opts.k;
skip = opts.skip;
m = opts.epoches;

% initialize

if ~isa(X, 'double'); X = double(X); end
if ~isa(y, 'double'); y = double(y); end

w = opts.w_init;
if isempty(w)
    w = zeros(d, 1);
else
    if ~isa(w, 'double'); w = double(w); end
end

w0 = opts.w0_init;
if isempty(w0)
    w0 = 0;
else
    w0 = double(w0);
end

% select core algorithm

switch lower(opts.algorithm)
    case 'pegasos'
        corefun = ...
            @(x, y, t, w, w0) pli_pegasos(x, y, lambda, lambda0, t, k, w, w0);
                    
    case 'sgd-qn'
        if k ~= 1
            error('pli_linsvm_sgdx:invalidarg', ...
                'k must be 1 for algorithm sgd-qn.');
        end
        
        corefun = ...
            @(x, y, t, w, w0) pli_sgdqn(x, y, lambda, lambda0, t, skip, w, w0);
        
    otherwise
        error('pli_linsvm_sgdx:invalidarg', ...
            'Unknown algorithm name %s', opts.algorithm);
end

% run algorithm

t = 0;

if nargout >= 3
    calc_objv = true;
    objv = zeros(1, m);
else
    calc_objv = false;
end

for iep = 1 : m
    
    si = randi(n, [1, T * k]);
    Xs = X(:, si);
    ys = y(si);
    
    [w, w0] = corefun(Xs, ys, t, w, w0);
    t = t + T;
    
    if calc_objv
        objv(iep) = pli_linsvm_objv(X, y, lambda, lambda0, 0, w, w0);
    end
end
    

%% Auxiliary functions

function check_options(opts, d)

v = opts.algorithm;
if ~ischar(v)
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for algorithm option should be a string.');
end

v = opts.T;
if ~(isnumeric(v) && isreal(v) && isscalar(v) && v == fix(v) && v >= 1)
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for option T should be a positive integer.');
end

v = opts.k;
if ~(isnumeric(v) && isreal(v) && isscalar(v) && v == fix(v) && v >= 1)
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for option k should be a positive integer.');
end

v = opts.skip;
if ~(isnumeric(v) && isreal(v) && isscalar(v) && v == fix(v) && v >= 1)
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for option skip should be a positive integer.');
end

v = opts.epoches;
if ~(isnumeric(v) && isreal(v) && isscalar(v) && v >= 1)
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for epoches should be a positive integer.');
end

v = opts.w_init;
if ~(isempty(v) || (isfloat(v) && isreal(v) && isequal(size(v), [d 1])))
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for w_init should be empty or a d-by-1 real vector.');
end

v = opts.w0_init;
if ~(isempty(v) || (isfloat(v) && isreal(v) && isscalar(v)))
    error('pli_linsvm_sgdx:invalidarg', ...
        'The value for w_init should be empty or real scalar.');
end



