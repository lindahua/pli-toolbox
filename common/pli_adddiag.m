function A = pli_adddiag(A, v)
%PLI_ADDDIAG Add values to diagonal elements
%
%   B = PLI_ADDDIAG(A, v)
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
    error('pli_adddiag:invalidarg', 'A must be a matrix.');
end

%% main

siz = size(A);
inds = 1 + (0:min(siz)-1) * (siz(1)+1);

if size(v, 1) > 1
    v = v.';
end
A(inds) = A(inds) + v;

