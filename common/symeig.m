function [U, evs] = symeig(C)
%SYMEIG Eigen-analysis on symmetric matrix
%
%   [U, evs] = SYMEIG(C);
%
%       This is a simple wrapper of the builtin function eig. 
%       It returns the outputs in a form that is convenient for 
%       further analysis.
%
%   Arguments
%   ----------
%   - C :       The input matrix, which should be symmetric.
%
%   Returns
%   --------
%   - U :       The matrix of eigenvectors. Each column is an eigenvector.
%   - evs :     The vector of eigenvalues, arranged in descending order.
%

%% main

[U, D] = eig(C);
evs = diag(D);

if ~issorted(evs)
    [evs, si] = sort(evs);
    U = U(:, si);
end

