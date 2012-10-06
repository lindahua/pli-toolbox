function v = pli_logdet(A)
%PLI_LOGDET Log-determinant of a positive definite matrix
%
%   v = PLI_LOGDET(A)
%       
%       Evaluates the log-determinant of A (using Cholesky
%       decomposition). Here, A must be a positive definite
%       matrix.
%

L = chol(A);
v = 2.0 * sum(log(diag(L)));

