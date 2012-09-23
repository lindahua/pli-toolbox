function A = add_diagonals(A, v)
%ADD_DIAGONALS Add values to diagonal elements
%
%   B = ADD_DIAGONALS(A, v)
%
%       Add the value(s) of v to the diagonal elements of A.
%       
%   Parameters
%   ----------
%   - A :   A square matrix.
%   - v :   The value(s) to be added to the diagonal of A.
%           v can be either a scalar or a vector.
%
%   Returns
%   -------
%   - B :   The resultant matrix.
%

%% argument checking

if ~ismatrix(A)
    error('add_diagonals:invalidarg', 'A must be a matrix.');
end

%% main

siz = size(A);
inds = 1 + (0:min(siz)-1) * (siz(1)+1);
A(inds) = A(inds) + v;

