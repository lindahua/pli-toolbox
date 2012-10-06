function C = pli_intcount(rgn, I, J)
%PLI_INTCOUNT Integer counting
%
%   C = PLI_INTCOUNT(m, I);
%
%       Counts the number of times each integer in [1, m] appear in I.
%
%       The result C is a column vector of size [m, 1], as
%
%           C(i) = sum(I(:) == i).
%
%   C = PLI_INTCOUNT([m, n], I, J);
%
%       Counts the number of times each pair in [1, m] x [1, n] appear
%       in I and J.
%
%       The result C is a matrix of size [m, n], as
%
%           C(i, j) = sum(I(:) == i & J(:) == j).
%
%       Here, I and J must be of the same shape.
%
%   Note, the class of C is int32.
%

%% argument checking

if ~(isnumeric(rgn) && isreal(rgn))
    error('pli_intcount:invalidarg', ...
        'The first argument should be numeric.');
end

if isscalar(rgn)
    nd = 1;
    m = int32(rgn(1));
    
elseif numel(rgn) == 2
    nd = 2;
    m = int32(rgn(1));
    n = int32(rgn(2));
    
else
    error('pli_intcount:invalidarg', ...
        'The first argument is invalid.');
end    

if ~(isnumeric(I) && isreal(I) && ~issparse(I))
    error('pli_intcount:invalidarg', ...
        'I should be a non-sparse real array.');
end

if nd == 2
    if ~(isnumeric(J) && isreal(J) && ~issparse(J))            
        error('pli_intcount:invalidarg', ...
            'J should be a non-sparse real array.');
    end

    if numel(I) ~= numel(J)
        error('pli_intcount:invalidarg', ...
            'I and J should be of the same size.');
    end
    
    if ~isa(J, class(I))
        J = cast(J, class(I));
    end
end

%% main

if nd == 1
    C = intcount1_cimp(m, I);
else
    C = intcount2_cimp(m, n, I, J);
end


