function A = pli_aggreg2(siz, vals, I, J, op)
%PLI_AGGREG2 Aggregation according to row and column indices
%
%   A = PLI_AGGREG2(siz, vals, I, J)
%
%       Accumulates the values in vals to the entries of A, according 
%       to the subscripts given in I and J.
%
%       The output A is a matrix, whose size is specified by siz, and
%       it is of the same class as vals, which can be either double
%       or single.
%
%       Suppose vals is a matrix of size [m, n], then I and J can also
%       be matrices of size [m, n]. Then, the values in A are defined
%       to be
%
%           A(k, l) = sum(vals(I == k & J == l));
%
%       AGGREGX supports broadcasting, and therefore the size of I and 
%       J can also be [m, 1] or [1, n]. Note that I and J need not be
%       of the same size. 
%
%   A = PLI_AGGREG2(siz, vals, I, J, op)
%
%       Uses the argument op to specify the type of accumulation to 
%       perform. Here, op can be 'sum', 'max', or 'min'. When op
%       is not given, it is set to 'sum' by default.
%


%% argument checking

am = int32(siz(1));
an = int32(siz(2));

if ~(am >= 1 || an >= 1)
    error('pli_aggreg2:invalidarg', 'The value of siz is invalid.');
end

if ~(isfloat(vals) && isreal(vals) && ~issparse(vals))
    error('pli_aggreg2:invalidarg', ...
        'vals should be a non-sparse real matrix.');
end

if ~(isnumeric(I) && isreal(I) && ismatrix(I))
    error('pli_aggreg2:invalidarg', 'I should be a numeric matrix.');
end

if ~(isnumeric(J) && isreal(J) && ismatrix(J))
    error('pli_aggreg2:invalidarg', 'J should be a numeric matrix.');
end

if nargin < 5
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
            error('pli_aggreg2:invalidarg', 'Invalid value for op.');
    end
end


%% main

A = aggreg2_cimp(am, an, vals, int32(I)-1, int32(J)-1, op_code);


