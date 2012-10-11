function [H, f, A, b, lb] = pli_linsvm_prob(X, y, lambda)
%PLI_LINSVM_PROB Constructs a QP problem for linear SVM
%
%   [H, f, A, b] = PLI_LINSVM_PROB(X, y, lambda);
%
%       Constructs a QP problem for a linear SVM as
%
%           minimize (lambda/2) * ||theta|| + (1/n) * sum_i xi_i
%
%               s.t. y_i (theta' * x_i + bias) >= 1 - xi_i
%
%       The solution s is a (d+n+1)-dimensional vector, where 
%       - s(1:d) is theta,
%       - s(d+1) is bias,
%       - s(d+2:d+n+1) is xi (slack variables)
%
%       This problem can be written as
%
%           minimize 0.5 * s' * H * s + f' * s
%               s.t. A * x <= b, x >= lb
%
%       This function returns H, f, A, b, and lb, 
%       which can then be fed to a QP solver to get the solution. 
%       Both H and A are sparse.
%
%   Arguments
%   ---------
%   X :     The feature matrix, size [d n]
%   y :     The response vector (length = n)
%   c :     The slack coefficients, a scalar or a vector of length n.
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_linsvm_prob:invalidarg', ...
        'X should be a real matrix.');
end
[d, n] = size(X);

if ~(isfloat(y) && isreal(y) && isvector(y) && length(y) == n)
    error('pli_linsvm_prob:invalidarg', ...
        'y should be a real vector of length n.');
end

if ~(isfloat(lambda) && isreal(lambda) && isscalar(lambda) && lambda > 0)
    error('pli_linsvm_prob:invalidarg', ...
        'lambda should be a positive real scalar.');
end

%% main

if ~isa(X, 'double'); X = double(X); end
if ~isa(y, 'double'); y = double(y); end
if size(y, 2) > 1; y = y.'; end % turn y into a column

sd = d + n + 1;

H = sparse(1:d, 1:d, double(lambda), sd, sd);

f = zeros(sd, 1);
f(d+2:sd) = (1/n);

A = [bsxfun(@times, X.', -y), -y,  sparse(1:n, 1:n, -1, n, n)];
b = - ones(n, 1);

lb = zeros(sd, 1);
lb(1:d+1) = -inf;

