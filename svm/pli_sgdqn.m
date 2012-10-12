function [w, w0] = pli_sgdqn(X, y, lambda, lambda0, t0, skip, w, w0)
%PLI_SGDQN SVM estimation using SGD-QN algorithm
%
%   [w, w0] = PLI_SGDQN(X, y, lambda, lambda0, t0, K, w, w0)
%
%       Solves the coefficient w and offset w0 using for a linear
%       SVM for using SGD-QN algorithm.
%
%   Arguments
%   ---------
%   X :         The stream of sample features
%
%   y :         The stream of sample outputs
%
%   lambda :    The regularization coefficient for theta.
%
%   lambda0 :   The regularization coefficient for theta0.
%               (lambda0 == 0 indicates w0 is fixed to zero)
%
%   t0 :        Start time
%
%   skip :      Interval between evaluation of B
%
%   Updates
%   -------
%   w :         The coefficient vector.
%
%   w0 :        The bias (offset) value.
%
%
%   Note: every argument should have double value type. 
%
%   Remarks
%   -------
%   - This is NOT a function aimed for end users. It is supposed to be
%     called by other functions as a computation core 
%     (e.g. pli_linsvm_sgdx). No argument checking is performed.
%
%   - X and y should have been randomly permutated. This function 
%     runs a pass of X column by column.
%

%% main

aug = sqrt(lambda / lambda0);

if aug > 0
    w0_ = w0 / aug;
else
    w0_ = 0;
end

[w, w0_] = sgdqn_cimp(X, y, lambda, aug, t0, skip, w, w0_);

w0 = w0_ * aug;


