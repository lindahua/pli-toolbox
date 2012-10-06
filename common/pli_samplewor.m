function x = pli_samplewor(m, n)
%PLI_SAMPLEWOR Random sampling without replacement
%
%   x = PLI_SAMPLEWOR(m, n)
%
%       Draws n integers from 1:m without replacement. 
%       
%       Here, both m and n are positive integers, with n <= m.
%
%       The size of x will be [n 1]
%

%% argument checking

if ~(isreal(m) && isscalar(m) && m > 0)
    error('pli_samplewor:invalidarg', 'm should be a positive scalar.');
end

if ~(isreal(n) && isscalar(n) && n > 0 && n <= m)
    error('pli_samplewor:invalidarg', ...
        'n should be a positive scalar with n <= m.');
end

%% main

if n == 1
    x = randi(m, 1);
    
elseif n == 2
    x = randi(m, [2 1]);
    while x(2) == x(1)
        x(2) = randi(m, 1);
    end
    
else
    if n * 5 < m
        b = false(m, 1);
        b(randi(m, [n 1])) = 1;
        
        s = sum(b);
        while s < n
            b(randi(m, [n - s, 1])) = 1;
            s = sum(b);
        end
        
        x = find(b);
        x = x(randperm(n));
        
    else
        x = randperm(m);
        if n < m
            x = x(1:n);
        end
    end
end


