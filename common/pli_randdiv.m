function R = pli_randdiv(n, d)
%PLI_RANDDIV Random division of indices
%
%   R = PLI_RANDDIV(n, d);
%
%       Randomly divides 1:n into d disjoint sets, each containing
%       approximately (n/d) numbers.
%
%   R = PLI_RANDDIV(n, [m1, m2, ..., md])
%
%       Randomly divides 1:n into d disjoint sets, such that the i-th set 
%       contains mi numbers.
%

%% argument checking

if ~(isnumeric(n) && isscalar(n) && n == fix(n) && n >= 1)
    error('pli_randdiv:invalidarg', 'n should be a positive integer.');
end

if isscalar(d)
    if ~(isnumeric(d) && d == fix(d) && d >= 1 && d <= n)
        error('pli_randdiv:invalidarg', ...
            'd should be a positive integer with d <= n.');
    end
    ns = [];
elseif isvector(d)
    ns = d;
    if sum(ns) ~= n
        error('pli_randdiv:invalidarg', 'Inconsistent total number.');
    end
else
    error('pli_randdiv:invalidarg', 'The second argument is invalid.');
end

%% main

% calculate ns

if isempty(ns)
    nb = floor(n / d);
    ns = nb * ones(1, d);        
    rn = n - nb * d;
    
    if rn > 0
        a = pli_samplewor(d, rn);
        ns(a) = ns(a) + 1;
    end
end

% make R

p = randperm(n);
R = cell(1, d);

ei = 0;
for k = 1 : d
    bi = ei + 1;
    ei = ei + ns(k);    
    R{k} = sort(p(bi:ei));    
end


