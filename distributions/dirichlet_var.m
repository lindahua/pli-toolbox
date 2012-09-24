function v = dirichlet_var(alpha)
%DIRICHLET_VAR Component variances of Dirichlet distribution
%
%   v = dirichlet_var(alpha)
%
%       Evaluates the variances of a Dirichlet distribution or
%       multiple Dirichet distributions with parameter alpha.
%
%       The size of v is the same as that of alpha.
%

%% main

m = size(alpha, 2);

a0 = sum(alpha, 1);
s = (a0 .^ 2) .* (a0 + 1);

if m == 1
    v = alpha .* (a0 - alpha) * (1 ./ s);
else
    v = alpha .* bsxfun(@minus, a0, alpha);
    v = bsxfun(@times, v, (1 ./ s));
end

