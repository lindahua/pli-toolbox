function [H, f, Aeq, beq] = pli_kersvr_prob(K, y, e)
%PLI_KERSVR_PROB QP problem for kernel SVR
%
%   [H, f, Aeq, beq] = PLI_KERSVR_PROB(K, y, e);
%
%       Constructs a QP problem for kernelized support vector regression
%       as follows
%
%           minimize (1/2) (ap - an)' K (ap - an) 
%                   + e * sum(ap + an) - y' * (ap - an)
%
%           s.t. sum(ap - an) = 0, and 0 <= ap, an <= C
%
%       The dimension of the solution space is 2 * n. The solution is
%       in the form of [ap an].
%
%   Arguments
%   ---------
%   - K :       The kernel matrix, size [n, n]
%   - y :       The response values, vector of length n
%   - e :       The allowable deviation range
%
%   Returns
%   -------
%   - H :       The quadratic coefficient matrix
%   - f :       The linear coefficient vector
%   - Aeq :     The equality constraint coefficient matrix
%   - beq :     The rhs of equality constraints
%

%% argument checking

if ~(isfloat(K) && isreal(K) && ismatrix(K) && size(K,1) == size(K,2))
    error('pli_kersvr_prob:invalidarg', ...
        'K should be a real square matrix.');
end
n = size(K, 1);

if ~(isfloat(y) && isreal(y) && isvector(y) && length(y) == n)
    error('pli_kersvr_prob:invalidarg', ...
        'y should be a real vector of length n.');
end

if size(y, 2) > 1
    y = y.';
end

if ~(isfloat(e) && isreal(e) && isscalar(e) && e > 0)
    error('pli_kersvr_prob:invalidarg', 'e should be a positive scalar.');
end
e = double(e);

%% main

if ~isa(K, 'double'); K = double(K); end
if ~isa(y, 'double'); y = double(y); end

H = [K -K; -K K];

didx = 1 + (0:2*n-1) * (2*n+1);
H(didx) = H(didx) + 1.0e-8;

f = [e - y; e + y];
Aeq = [ones(1, n), -ones(1, n)];
beq = 0;


