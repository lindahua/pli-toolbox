function v = ddentropy(P, dim)
%DDENTROPY Entropy of discrete distribution
%
%   v = DDENTROPY(P);
%   v = DDENTROPY(P, dim);
%
%       Evaluates the entropy of the probability distribution along
%       the specified dimension. If dim is not given, it is along
%       the first non-singleon dimension.
%

%% main

if nargin < 2
    v = - nansum(P .* log(P));
else
    v = - nansum(P .* log(P), dim);
end
    

