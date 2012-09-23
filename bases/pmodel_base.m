classdef pmodel_base < handle
    % The base class of simple probabilistic models
    %
    
    methods(Abstract)
        
        [obs, n] = check_observations(obj, obs)
        % Verify the validity of input observations
        
        params = estimate_param(obj, obs, weights, hints)
        % Estimate model parameters from observations
        
        L = evaluate_loglik(obj, params, obs)
        % Evaluate log-likelihoods w.r.t. given parameters
        
        L = evaluate_logpri(obj, params)
        % Evaluate log prior values of the input parameters
        
    end
    
end
