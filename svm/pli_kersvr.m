function [alpha, b, objv] = pli_kersvr(K, y, e, C, solver, opts)
%PLI_KERSVR Solves Kernel Support Vector Regression 
%
%   [alpha, b] = PLI_KERSVR(K, y, e, C);
%   [alpha, b] = PLI_KERSVR(K, y, e, C, solver);
%   [alpha, b] = PLI_KERSVR(K, y, e, C, solver, opts);
%
%       Solves Kernel support vector regression based on a given kernel
%       matrix.
%
%   [alpha, b, objv] = PLI_KERSVR( ... );
%
%       Additionally returns the objetive function value.
%
%
%   Arguments
%   ---------
%   - K :       The kernel matrix, size [n, n].
%
%   - y :       The vector of responses, length n.
%
%   - e :       The tolerance of deviation (error).
%
%   - C :       The slack penalty
%
%   - solver :  The QP solver employed to solve the problem, which
%               can be a string with either the following value:
%               - 'ip':     Using MATLAB's interior-point algorithm
%               - 'gurobi': Using Gurobi's QP solver
%               It can also be a function handle. 
%
%               The default solver is 'ip'.
%
%   - opts :    The struct of options/params input to the solver.
%
%   Returns
%   -------
%   - alpha :   The coefficients on training samples
%
%   - b :       The bias constant
%
%   - objv :    The objective function value.
%
%
%   With these quantities, the prediction can be written as
%
%       f(x) = sum_i alpha_i k(x_i, x) + b
%

%% argument checking

if ~(isfloat(C) && isreal(C) && isscalar(C) && C > 0)
    error('pli_kersvr:invalidarg', 'C should be a positive scalar.');
end

if nargin < 5
    solver = 'ip';
end

if nargin < 6
    opts = [];
end

%% main

n = size(K, 1);
[H, f, Aeq, beq] = pli_kersvr_prob(K, y, e);

lb = zeros(2*n, 1);
ub = C .* ones(2*n, 1);

[sol, objv] = pli_qpsolve(H, f, [], [], Aeq, beq, lb, ub, solver, opts);

ap = sol(1:n);
an = sol(n+1:2*n);
alpha = ap - an;

b = find_offset(K, y, e, C, ap, an, alpha);


function b = find_offset(K, y, e, C, ap, an, alpha)

if size(y, 2) > 1
    y = y.';
end

ep = max(1e-12, C * 1e-9);
s1 = find(ap < C - ep | an > ep);
s2 = find(ap > ep | an < C - ep);

if ~isempty(s1) && ~isempty(s2)   
    
    u = K * alpha;
    
    m1 = max(y(s1) - u(s1) - e);
    m2 = min(y(s2) - u(s2) - e);    
    b0 = (m1 + m2) / 2;   
               
    b = fminsearch(@(x) fun_r(x, u, y, C), b0);    
else
    b = 0;
    warning('svm_dual_offset:nosuppvec', ...
        'No proper training samples to bound of value of b.');
end


function v = fun_r(b, u, y, c)

loss = max(abs(u + b - y), 0);
if isscalar(c)
    v = c * sum(loss);
else
    v = sum(loss .* c);
end




