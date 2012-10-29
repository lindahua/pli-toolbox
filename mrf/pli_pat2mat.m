function A = pli_pat2mat(pat, x)
%PLI_PAT2MAT Sparse matrix from a specific pattern
%
%   A = PLI_PAT2MAT(pat, phi);
%
%       Constructs a sparse matrix A from a given pattern pat and a 
%       value vector phi, in the way below:
%
%       The size of A equals the size of pat.
%       Given i and j, if pat(i, j) == 0, then A(i, j) is 0.
%       If pat(i, j) > 0, then A(i, j) = x(pat(i, j)).
%

%% main

[m, n] = size(pat);
[I, J, IDX] = find(pat);

A = sparse(I, J, x(IDX), m, n);


