function [alpha, bias, objv] = pli_kersvm(K, y, C, solver, opts)
%PLI_KERSVM Solve Kernel Support Vector Machine
%
%   [alpha, bias, objv] = PLI_KERSVM(K, y, C);
%   [alpha, bias, objv] = PLI_KERSVM(K, y, C, solver);
%   [alpha, bias, objv] = PLI_KERSVM(K, y, C, solver, opts);
%
%       Solves Kernel SVM. Given alpha and bias, prediction on a sample x
%       can be made using the following formula:
%
%           sum_i alpha_i y_i k(x_i, x) + b
%
%
%   Arguments
%   ---------
%   - K :       A pre-computed kernel matrix on training set
%
%   - y :       the binary labels (+1/-1)
%
%   - C :       the slack coefficient.
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
%   - alpha :   The coefficients on training samples, a n-by-1 vector.
%   - bias :    The bias term.
%   - objv :    The objective value.
%

%% argument checking

if ~(isfloat(C) && isreal(C) && isscalar(C) && C > 0)
    error('pli_kersvm:invalidarg', 'C should be a positive scalar.');
end

if nargin < 4
    solver = 'ip';
end

if nargin < 5
    opts = [];
end

%% main

if size(y, 1) > 1
    y = y.';
end

[H, f, Aeq, beq] = pli_kersvm_prob(K, y);

n = size(K, 1);
lb = zeros(n, 1);
ub = C * ones(n, 1);

didx = 1 : (n+1) : n^2;
H(didx) = H(didx) + 1.0e-10 * max(H(didx));

[alpha, objv] = pli_qpsolve(H, f, [], [], Aeq, beq, lb, ub, solver, opts);

% solve bias

u = (alpha.' .* y) * K;

ep = 1.0e-9 * C;
sv = find(alpha > ep & alpha < C - ep);

if isempty(sv)
    b = 0;
else
    b = mean(y(sv) - u(sv));
end

objf = @(b) objv_findb(b, y, u);
bias = fminsearch(objf, b);


function v = objv_findb(b, y, u)

v = sum(max(1 - (u + b) .* y, 0));

