function [y, g] = pli_approxl1(x, w)
%PLI_APPROX1 Approximate L1 function 
%
%   The approximate L1 function is defined as below:
%
%       y(i) = (1/2) * (x^2/w + w)   when |x| < w
%            = |x|                   when |x| >= w
%
%
%   y = PLI_APPROXL1(x, w);
%
%       Evaluates approximate L1 function at x, with band-width w.
%
%   [y, g] = PLI_APPROXL1(x, w);
%
%       Additionally returns the derivatives.


%% main

y = abs(x);

if w > 0
    is_small = find(y < w);
    if ~isempty(is_small)
        y(is_small) = (1 / (2*w)) * y(is_small).^2 + (w/2);
    end
end

if nargin >= 2
    g = sign(x);
    if w > 0 && ~isempty(is_small)        
        g(is_small) = x(is_small) * (1/w);
    end
end

