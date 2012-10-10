function [theta, theta0] = pli_logireg(X, y, w, lambda, lambda0, s0, solver)
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
%       loss(u, y) = log(1 + exp(-y * u))
%
%   
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda);
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0);
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0, s0);
%   [theta, theta0] = PLI_LOGIREG(X, y, w, lambda, lambda0, s0, solver);
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
%
%   - solver :  A function handle to solve the optimization problem,
%               which can be called as 
%
%                   x = solver(f, x0).
%
%               By default, pli_fminnewton is used.
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
    if size(w, 2) > 1; w = w.'; end % w: column
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

if nargin < 6 || isempty(s0)
    s0 = zeros(d+1, 1);
else
    if use_bias
        ds = d + 1;
    else
        ds = d;
    end
    
    if ~(isfloat(s0) && isreal(s0) && isequal(size(s0), [ds 1]))
        error('pli_logireg:invalidarg', 'The value for s0 is invalid.e');
    end
end

if nargin < 7
    solver = @pli_fmingd;
else
    if ~isa(solver, 'function_handle')
        error('pli_logireg:invalidarg', ...
            'solver should be a function handle.');
    end
end


%% main

s = solver(@objfun, s0);

if use_bias
    theta = s(1:d);
    theta0 = s(d+1);
else
    theta = s;
    theta0 = 0;
end


%% objective function

    function [v, g] = objfun(s)
        
        if use_bias
            t = s(1:d);
            t0 = s(d+1);            
            u = (t' * X) + t0;
        else
            t = s;
            u = t' * X;
        end
        
        % evaluate objective value
        
        yu = y .* u;
        
        sp = find(yu >= 0);
        sn = find(yu < 0);
        
        loss = zeros(1, n);
        loss(sp) = log(1 + exp(- yu(sp)));
        loss(sn) = -yu(sn) + log(1 + exp(yu(sn)));
        
        if isempty(w)
            tloss = sum(loss);
        else
            tloss = loss * w;
        end
        
        v = (lambda/2) * (t'*t) + tloss;
        if use_bias && lambda0 > 0
            v = v + (lambda0/2) * (t0^2);
        end
        
        % evaluate gradient
        
        if nargout >= 2
            
            yq = y ./ (1 + exp(yu));
            
            if isempty(w)
                g = - (X * yq');
            else
                g = - (X * (yq' .* w));
            end
            
            g = g + lambda * t;
            
            if use_bias
                if isempty(w)
                    g0 = - sum(yq);
                else
                    g0 = - ((yq) * w);
                end
                
                g0 = g0 + lambda0 * t0;
                g = [g; g0];
            end
        end 
    end

end




