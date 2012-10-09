function [w, w0, xi, objv] = pli_linsvm(X, y, c, solver, opts)
%PLI_LINSVM Solve a linear SVM problem
%
%   [w, w0, xi, objv] = PLI_LINSVM(X, y, c);
%   [w, w0, xi, objv] = PLI_LINSVM(X, y, c, solver);
%   [w, w0, xi, objv] = PLI_LINSVM(X, y, c, solver, opts);
%
%       Solves a linear SVM problem.
%
%   Arguments
%   ---------
%   X :     The feature matrix, of size [d n]. Each column represents
%           an observed sample.
%
%   y :     The response vector, of length n. Each value in y should be
%           either 1 or -1.
%
%   c :     The slack coefficients, which can be either a scalar or
%           a vector of length n.
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
%   - w :       The feature weight vector
%   - w0 :      The offset value
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

[H, f, A, b, lb] = pli_linsvm_prob(X, y, c);

[sol, objv] = pli_qpsolve(H, f, A, b, [], [], lb, [], solver, opts);

d = size(X, 1);
w = sol(1:d);
w0 = sol(d+1);
xi = sol(d+2:end);



