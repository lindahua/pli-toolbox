function v = pli_logdet(A)
%PLI_LOGDET Log-determinant of a positive definite matrix
%
%   v = PLI_LOGDET(A)
%       
%       Evaluates the log-determinant of A (using Cholesky
%       decomposition). Here, A must be a positive definite
%       matrix.
%

%% argument checking

if ~(isfloat(A) && isreal(A) && ismatrix(A) && size(A,1) == size(A,2))
    error('pli_logdet:invalidarg', ...
        'A should be a real square matrix.');
end


%% main

R = chol(A);

diagR = diag(R);
if issparse(diagR)
    diagR = full(diagR);
end

v = 2.0 * sum(log(diagR));


