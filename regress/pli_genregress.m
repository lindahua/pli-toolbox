function theta = pli_genregress(lossfun, X, y, w, lambda, s0, opts)
%PLI_LOGIREG Solves a generic regression problem
%
%   Generally, a regression problem is to minimize the following 
%   problem w.r.t. theta and theta0, as
%
%       (lambda/2) * ||theta||^2 + (lambda0/2) * (theta0)^2 + 
%       sum_i loss(x_i, y_i; theta).
%
%   theta = PLI_GENREGRESS(X, y, w, lambda, s0);
%   theta = PLI_GENREGRESS(X, y, w, lambda, s0, opts);
%
%       Solves the logistic regression problem.
%
%   Arguments
%   ---------
%   - lossfun:  The loss function, which should support the following
%               syntax:
%
%                   v = loss(param, X, y, w);
%                       returns the total loss on samples X and y.
%
%                       w is the sample vector, which is either empty
%                       or a vector of length n.
%
%                   [v, g] = loss(param, X, y, w);
%                       returns the gradient of total loss w.r.t. 
%                       the parameters as well.
%
%               If newton method is to be used, it should also support:
%
%                   [v, g, H] = loss(param, X, y, w);
%                       additionally returns the Hessian matrix.
%
%
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
%   - lambda :  The regularization coefficients, which can be either
%               a scalar or a vector of the same size as theta.
%
%   - s0 :      The initial solution.
%
%   - opts :    The options for pli_fminunc to solve the optimization
%               problem. One can use pli_optimset to construct this
%               struct.
%
%
%   theta = PLI_GENREGRESS(X, y, w, lambda, s0, solver);
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
    error('pli_genregress:invalidarg', 'X should be a real matrix.');
end
n = size(X, 2);

if ~(isfloat(y) && isreal(y) && isvector(y) && numel(y) == n)
    error('pli_genregress:invalidarg', ...
        'y should be a real vector of length n.');
end

if isempty(w)
    w = [];
else
    if ~(isfloat(w) && isreal(w) && numel(w) == n)
        error('pli_genregress:invalidarg', ...
            'w should be a real vector of length n.');
    end
    if size(w, 2) > 1; w = w.'; end % w: column
end
    
if ~(isfloat(s0) && isreal(s0) && isvector(s0) && size(s0, 2) == 1)
    error('pli_genregress:invalidarg', 's0 should be a real vector.');
end


if ~(isfloat(lambda) && isreal(lambda) && ...
        (isscalar(lambda) || isequal(size(lambda), size(s0))) )
    error('pli_genregress:invalidarg', ...
        'lambda should be either a scalor or a vector with the same size as s0.');
end

if nargin < 7 || isempty(opts)
    solver = @pli_fminunc;
else
    if isstruct(opts)
        solver = @(f, x) pli_fminunc(f, x, opts);
    elseif isa(opts, 'function_handle')
        solver = opts;
    else
        error('pli_genregress:invalidarg', ...
            'The last argument is invalid.');
    end
end


%% main

theta = solver(@objfun, s0);


%% objective function

    function [v, g, H] = objfun(s)

        if nargout <= 1
            v = lossfun(s, X, y, w);
        elseif nargout == 2
            [v, g] = lossfun(s, X, y, w);
        else
            [v, g, H] = lossfun(s, X, y, w);
        end    
              
        if isscalar(lambda)
            rv = (0.5 * lambda) * (s'*s);
        else
            rv = 0.5 * (lambda' * (s.^2));
        end
        v = v + rv;
                
        if nargout >= 2
            g = g + (lambda .* s);
        end
        
        if nargout >= 3
            d = size(s, 1);
            didx = (1 + (0:d-1) * (d+1)).';
            H(didx) = H(didx) + lambda;  
        end        

    end

end




