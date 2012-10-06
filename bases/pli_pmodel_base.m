classdef pli_pmodel_base < handle
    % The base class of simple probabilistic models
    %
    
    methods(Abstract)
        
        n = check_observations(self, obs)
        % Verify the validity of input observations
        
        params = estimate_param(self, obs, weights, hints)
        % Estimate model parameters from observations
        
        L = evaluate_loglik(self, params, obs)
        % Evaluate log-likelihoods w.r.t. given parameters
        
        L = evaluate_logpri(self, params)
        % Evaluate log prior values of the input parameters
        
    end
    
end
