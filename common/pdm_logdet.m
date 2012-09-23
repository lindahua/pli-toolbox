function v = pdm_logdet(A)
%PDM_LOGDET Log-determinant of a positive definite matrix
%
%   v = PDM_LOGDET(A)
%       
%       Evaluates the log-determinant of A (using Cholesky
%       decomposition). Here, A must be a positive definite
%       matrix.
%

L = chol(A);
v = 2.0 * sum(log(diag(L)));

