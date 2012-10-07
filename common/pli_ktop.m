function [I, Y] = pli_ktop(X, K, dir, dim)
%PLI_KTOP Finds K smallest or largest elements
%   
%   I = PLI_KTOP(X, K, 'smallest');
%   I = PLI_KTOP(X, K, 'largest');
%
%       Finds K smallest or largest (depending on the value of the
%       third argument) elements in X along the first non-singleton
%       dimension, and returns their indices.
%
%       In particular, if X is a vector, it returns a vector I of 
%       length K, such that X(I) contains the K smallest or largest 
%       elements of X, in ascending or descending order.
%
%       If X is an m-by-n matrix (m > 1 and n >1), then it returns
%       a K-by-n matrix, where I(:,j) contains the indices of the
%       smallest or largest elements of X(:,j).
%
%   I = PLI_KTOP(X, K, 'smallest', dim);
%   I = PLI_KTOP(X, K, 'largest', dim);
%
%       One can also specify the dimension along which the selection
%       is performed. 
%
%       Here, dim can be either 1 or 2. If dim == 1, it returns a 
%       matrix of size [K, n], where I(:,j) contains the indices of
%       the top elements of X(:,j). If dim == 2, it returns a matrix
%       of size [m, K], where I(i, :) contains the indices of the top
%       elements of X(i,:).
%
%   [I, Y] = PLI_KTOP(X, K, ...);
%
%       Additionally returns the values of the selected elements in
%       the matrix Y.
%

%% argument checking

if ~(isnumeric(X) && isreal(X) && ismatrix(X) && ~issparse(X))
    error('pli_ktop:invalidarg', ...
        'X should be a non-sparse numeric matrix.');
end

if ~(isnumeric(K) && isreal(K) && isscalar(K) && K >= 1)
    error('pli_ktop:invalidarg', 'K should be a positive scalar.');
end
K = int32(K);

if strcmp(dir, 'smallest')
    is_asc = true;
elseif strcmp(dir, 'largest')
    is_asc = false;
else
    error('pli_ktop:invalidarg', ...
        'The value of the 3rd argument is invalid.');
end

if nargin < 4
    if size(X, 1) > 1
        dim = 1;
    else
        dim = 2;
    end
else
    if ~(isscalar(dim) && (dim == 1 || dim == 2))
        error('pli_ktop:invalidarg', ...
            'The value of dim should be either 1 or 2.');
    end
end

if K > size(X, dim)
    error('pli_ktop:invalidarg', ...
        'The value of K exceeds the working dimension.');
end


%% main

if dim == 1
    if nargout <= 1
        I = ktop_cimp(X, K, is_asc, 1); 
    else
        [I, Y] = ktop_cimp(X, K, is_asc, 1);
    end
else
    if size(X, 1) == 1
        if nargout <= 1
            I = ktop_cimp(X, K, is_asc, 2);
        else
            [I, Y] = ktop_cimp(X, K, is_asc, 2);
        end
    else
        if nargout <= 1
            I = ktop_cimp(X.', K, is_asc, 1).';
        else
            [I, Y] = ktop_cimp(X.', K, is_asc, 1);
            I = I.';
            Y = Y.';
        end
    end
end


