function g = pli_approxgrad(f, x, h)
%PLI_APPROXGRAD Computes approximated gradient of a function
%
%   g = PLI_APPROXGRAD(f, x, h);
%
%       Computes the approximated gradient of f at x, using finite
%       difference with interval h.
%

%% main

d = length(x);

g = zeros(d, 1);

hh = h / 2;
s = 1 / h;

for i = 1 : d
    xp = x;
    xn = x;
    
    xp(i) = xp(i) + hh;
    xn(i) = xn(i) - hh;
    
    g(i) = s * (f(xp) - f(xn));
end

