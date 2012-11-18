classdef pli_pmodel_base < handle
    % The base class of simple probabilistic models
    %
    
    methods(Abstract)
        
        n = check_observations(self, obs)
        % Verify the validity of input observations
        %
        %   It returns the number of samples contained in obs when
        %   obs is a valid sample set. Otherwise, it raises an error.
        %
        
        params = update_params(self, obs, weights, sidx, params)
        % Update model parameters from re-weighted observations
        %
        %   It optimizes the parameters based on MAP criterion.
        %
        %   It should be able to support several usage as follows:
        %
        %   params = self.update_map(obs, [], [], params);
        %
        %       updates parameters based on unweighted observations.
        %
        %   params = self.update_map(obs, weights, [], params);
        %
        %       updates parameters based on a weighted set of 
        %       observations. weights can be an m-by-n matrix, 
        %       for estimating m sets of parameters.
        %
        %   param = self.update_map(obs, [], sidx, param);
        %
        %       updates parameters based on a subset of non-weighted
        %       observations. sidx is a vector of indices.
        %
        %   param = self.update_map(obs, weights, sidx, param);
        %
        %       updates parameters based on a subset of weighted 
        %       observations. weights and sidx should be vectors
        %       of the same length.
        %        
                
        L = evaluate_loglik(self, params, obs)
        % Evaluate log-likelihoods w.r.t. given parameters
        %
        %   It evaluates the log-likelihood w.r.t a given set
        %   of parameters.
        %
        %   The resultant matrix L should be an m-by-n matrix,
        %   where m is the number of components parameters, and
        %   n is the number of observed samples.
        %
        
        L = evaluate_logpri(self, params)
        % Evaluate log prior values of the input parameters
        %
        %   It evaluates the log-prior over a set of parameters
        %
        %   If params contains m components, it should return
        %   a vector of length m in general. If no prior is associated,
        %   this function can simply return 0.
        %
        
    end
    
end
