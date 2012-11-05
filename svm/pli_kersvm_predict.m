function u = pli_kersvm_predict(alpha, bias, ytr, K)
%PLI_KERSVM_PREDICT Prediction based on Kernel SVM
%
%   u = PLI_KERSVM_PREDICT(alpha, bias, ytr, K);    
%
%       Evaluates the prediction based on a kernel SVM.
%
%       Suppose a kernel SVM is trained on a data-set with m samples,
%       and there are n testing samples.
%
%       Then, alpha should be an m-by-1 vector, ytr is the binary
%       labels of the training samples, which should be a vector 
%       of length m. K is the kernel matrix between training and
%       testing samples. 
%       

%% main

if size(ytr, 1) > 1
    u = (alpha .* ytr)' * K;
else
    u = (alpha' .* ytr) * K;
end

if bias ~= 0 
    u = u + bias;
end

