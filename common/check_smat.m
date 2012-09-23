function [d, f] = check_smat(A)
%CHECK_SMAT Checks a compact representation of a symmetric matrix
%
%   [d, f] = CHECK_SMAT(A);
%
%       Verifies the validity of A as a compact representation of 
%       a symmetric matrix, and returns its dimensio and form code.
%
%   Arguments
%   ---------
%   - A :  A compact representation of a symmetric matrix, which
%          can be in either of the following forms.
%
%          - a scalar: representing a matrix as A * eye(d) with
%                      arbitrary dimension d.
%                       
%          - a vector: representing a diagonal matrix as diag(A).
%
%          - a square matrix: the full representation.
%
%   Returns
%   -------
%   - d :   The matrix dimension, i.e. the full matrix is of size
%           d x d. When A is a scalar, d is set to 1, which, however,
%           can be used to represent a matrix of any dimension.
%
%   - f :   An integer indicating the representation form.
%
%           - f == 0:   scalar form
%           - f == 1:   column vector form
%           - f == 2:   full matrix form
%

%% argument checking

if ~(isfloat(A) && isreal(A) && ismatrix(A))
    error('check_smat:invalidarg', 'A must be a real matrix.');
end

%% main

if isscalar(A)
    d = 1;
    f = 0;
elseif size(A, 2) == 1
    d = numel(A);
    f = 1;
else
    d = size(A, 1);
    if size(A, 2) ~= d
        error('check_smat:invalidarg', ...
            'A is neither a vector nor a square matrix.');
    end
    f = 2;
end

