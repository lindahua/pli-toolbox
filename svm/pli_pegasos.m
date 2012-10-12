function [w, w0] = pli_pegasos(X, y, lambda, lambda0, t0, K, w, w0)
%PLI_PEGASOS SVM estimation using Pegasos algorithm
%
%   [w, w0] = PLI_PEGASOS(X, y, lambda, lambda0, t0, K, w, w0)
%
%       Solves the coefficient w and offset w0 using for a linear
%       SVM for using Pegasos algorithm.
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
%   k :         Block size (#samples for each iteration)
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

[w, w0_] = pegasos_cimp(X, y, lambda, aug, t0, K, w, w0_);

w0 = w0_ * aug;

function [w, w0] = ref_pegasos(X, y, t, lambda, aug, w, w0) 

% make predictions

u = w' * X + w0 * aug;

% udpate w

eta = 1 / (lambda * t);
rp = (1 - eta * lambda);

w = rp * w;
w0 = rp * w0;

k = size(X, 2);
r = eta / k;
for i = 1 : size(X, 2)    
    if y(i) * u(i) < 1            
        w = w + (r * y(i)) * X(:,i);
        w0 = w0 + (r * y(i)) * aug;
    end    
end

% rescale w (projection)

w_sca = 1 / (sqrt(lambda) * norm([w; w0]));
if w_sca < 1
    w = w * w_sca;
    w0 = w0 * w_sca;
end


function [w, w0] = ref_pegasos_run(X, y, lambda, aug, t0, K, w, w0)  %#ok<DEFNU>
%A reference (pure-matlab) implmentation for debugging

disp('Running ref-implementation ...');

fun = @(X, y, t, w, w0) ref_pegasos(X, y, t, lambda, aug, w, w0);
[w, w0] = svm_sgdx_refrun(fun, X, y, t0, K, w, w0);





