function [H, f, Aeq, beq] = pli_kersvm_prob(K, y)
%PLI_KERSVM_PROB The QP problem for Kernel SVM
%
%   [H, f, Aeq, beq] = PLI_KERSVM_PROB(K, y);
%
%       Constructs a QP problem for kernel SVM. This problem is formulated
%       as follows
%
%           minimize   (1/2) * sum_{i,j} a_i a_j y_i y_j K(i, j) 
%                    - sum_i a_i
%
%           s.t. sum a_i y_i = 0, 0 <= a_i <= C
%
%       This function returns the matrices/vectors related to this problem.
%
%   
%   Arguments
%   ---------
%   - K :           The pre-computed kernel matrix
%   - y :           The vector of binary labels (+1/-1).
%  

%% argument checking

if ~(isfloat(K) && isreal(K) && ismatrix(K) && size(K,1) == size(K,2))
    error('pli_kersvm_prob:invalidarg', ...
        'K should be a real square matrix.');
end
n = size(K, 1);

if ~(isnumeric(y) && isreal(y) && isvector(y) && length(y) == n)
    error('pli_kersvm_prob:invalidarg', ...
        'y should be a real vector of length n.');
end

if ~isa(y, 'double'); y = double(y); end
if size(y, 1) > 1; y = y.'; end  % turn y to a row

%% main

H = K .* (y' * y);
f =  - ones(n, 1);

Aeq = y;
beq = 0;

