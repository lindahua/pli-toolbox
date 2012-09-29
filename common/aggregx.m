function A = aggregx(K, vals, I, op)
%AGGREGX Aggregation according to subscripts
%
%   A = AGGREGX(K, vals, I)
%
%       Accumulates the scalars/vectors in vals according to corresponding
%       indices given in I.
%
%       There are three ways to use this function.
%
%       - scalar accumulation: 
%         Let I be a matrix of size [m, n], then A will be a matrix
%         of size [K, 1], as
%         
%           A(k) = sum(vals(I == k)).
%
%       - row accumulation:
%         Let I be a column of size [m 1], then A will be a matrix of
%         size [K, n], as
%
%           A(k,:) = sum(vals(I == k, :), 1).
%
%       - column accumulation:
%         Let I be a row of size [1 n], then A will be a matrix of 
%         size [m, K], as
%
%           A(:,k) = sum(vals(:, I == k), 2).
%
%   A = AGGREGX(K, vals, I, op)
%
%       Uses the argument op to specify the type of accumulation to 
%       perform. Here, op can be 'sum', 'max', or 'min'. When op
%       is not given, it is set to 'sum' by default.
%

%% argument checking

if nargin < 4
    op_code = 0;
else
    switch op
        case 'sum'
            op_code = 0;
        case 'max'
            op_code = 1;
        case 'min'
            op_code = 2;
        otherwise
            error('aggregx:invalidarg', 'Invalid value for op.');
    end
end

if ~(isfloat(vals) && isreal(vals) && ismatrix(vals) && ~issparse(vals))
    error('aggregx:invalidarg', ...
        'vals should be a non-sparse real matrix.');
end

K = int32(K);    
if K <= 0
    error('aggregx:invalidarg', 'K should be a positive integer.');
end

if ~(isnumeric(I) && isreal(I) && ismatrix(I) && ~issparse(I))
    error('aggregx:invalidarg', ...
        'I should be a non-sparse numeric matrix.');
end


%% main

A = aggregx_cimp(K, vals, int32(I)-1, op_code);




