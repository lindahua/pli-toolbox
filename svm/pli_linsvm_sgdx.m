function [w, w0] = pli_linsvm_sgdx(X, y, lambda, lambda0, varargin)
%PLI_LINSVM_SGDX Extended Stochastic Gradient Descent for Linear SVM
%
%   [w, w0] = PLI_LINSVM_SGDX(X, y, lambda, lambda0, ...);
%
%       Trains a linear support vector machine using a stochastic
%       gradient descent (SGD) method. 
%
%       Several well-known methods are implemented in this function,
%       which include a Pegasos and SGD-QN. (More algorithms will be
%       incorporated in future versions).
%
%   Arguments
%   ---------
%   - X :       The sample feature matrix, size [d n].
%               Here, d is the feature dimension, and n is the number
%               of samples.
%
%   - y :       The vector of sample outputs, of length n.
%               (Each value of y must be either 1 or -1).
%
%   - lambda :  The regularization coefficient for w.
%
%   - lambda0 : The regularization coefficient for w0 (the bias term).
%               