function [theta, bias, xi, objv] = pli_linsvm(X, y, lambda, solver, opts)
%PLI_LINSVM Solve a linear SVM problem
%
%   [theta, bias, xi, objv] = PLI_LINSVM(X, y, lambda);
%   [theta, bias, xi, objv] = PLI_LINSVM(X, y, lambda, solver);
%   [theta, bias, xi, objv] = PLI_LINSVM(X, y, lambda, solver, opts);
%
%       Solves a linear SVM problem.
%
%   Arguments
%   ---------
%   X :         The feature matrix, of size [d n]. Each column represents
%               an observed sample.
%
%   y :         The response vector, of length n. Each value in y 
%               should be either 1 or -1.
%
%   lambda :    The regularization coefficient for theta.
%
%   solver :    The QP solver employed to solve the problem, which
%               can be a string with either the following value:
%               - 'ip':     Using MATLAB's interior-point algorithm
%               - 'gurobi': Using Gurobi's QP solver
%               It can also be a function handle. 
%
%               The default solver is 'ip'.
%
%   opts :  The struct of options/params input to the solver.
%
%
%   Returns
%   -------
%   - theta :   The feature weight vector
%   - bias :    The (bias) offset value
%   - xi :      The values of slack variables
%   - objv :    The objective function value
%

%% argument checking

if nargin < 4
    solver = 'ip';
end

if nargin < 5
    opts = [];
end


%% main

[H, f, A, b, lb] = pli_linsvm_prob(X, y, lambda);

[sol, objv] = pli_qpsolve(H, f, A, b, [], [], lb, [], solver, opts);

d = size(X, 1);
theta = sol(1:d);
bias = sol(d+1);
xi = sol(d+2:end);



