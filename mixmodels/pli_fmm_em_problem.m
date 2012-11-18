classdef pli_fmm_em_problem < handle
    %PLI_FMM_EM_PROBLEM The problem to estimate finite mixtures using EM
    %
    %   A solution to a FMM EM problem is a struct with the following
    %   fields:
    %   - K :           The number of components.
    %
    %   - n :           The number of observations.
    %
    %   - components :  The component parameters.
    %
    %   - pi :          The prior probabilities over components
    %                   The size of pi is [K 1].
    %
    %   - logliks :     The log-likelihood matrix, of size [K, n].
    %
    %   - Q :           The soft assignment matrix.
    %                   The size of Q should be [K, n].
    %   
    
    %% Properties
    
    properties(GetAccess='public', SetAccess='private')
        model;      % the underlying generative model
        pricount;   % prior counting of each component
        
        nobs = 0;   % the number of observations
        obs;        % the observations
        weights;    % the observation weights        
    end
    
    
    %% Problem construction and settings
    
    methods
        
        function obj = pli_fmm_em_problem(model, pric)
            % Constructs a problem object
            %
            %   obj = FMM_EM_PROBLEM(model);
            %   obj = FMM_EM_PROBLEM(model, pric);
            %
            %   Arguments
            %   ---------
            %   - model :   The underlying generative model, which
            %               should be an instance of pmodel_base.
            %
            %   - pric :    Prior counts for each component. 
            %               (default = 0).
            %
            
            if ~isa(model, 'pli_pmodel_base')
                error('pli_fmm_em_problem:invalidarg', ...
                    'model should be an instance of pli_pmodel_base.');
            end
            
            if nargin < 2
                pric = 0;
            else
                if ~(isreal(pric) && isscalar(pric) && pric >= 0)
                    error('pli_fmm_em_problem:invalidarg', ...
                        'pric should be non-negative scalar.');
                end
                pric = double(pric);
            end
            
            obj.model = model;
            obj.pricount = pric; 
        end
        
        
        function set_obs(self, obs, w)
            % Set observation to the problem.
            %
            %   self.set_obs(obs, w);
            %
            %   Arguments
            %   ---------
            %   obs :   The observations.
            %
            %   w :     The observaton weights.
            %           (default = []).
            %
            
            n = self.model.check_observations(obs);
            
            if nargin < 3 || isempty(w)
                w = [];
            else
                if ~(isreal(w) && isfloat(w) && isequal(size(w), [n 1]))
                    error('pli_fmm_em_problem:invalidarg', ...
                        'The size of w is invalid.');
                end
            end
            
            self.nobs = n;
            self.obs = obs;
            self.weights = w;            
        end
        
    end
       
    
    %% Methods to support EM estimation
    
    methods

        function sol = init_solution(self, K)
            % Initializes a solution for EM estimation
            %
            %   sol = self.init_solution(K);
            %       Randomly initializes a solution with K components.
            %
            %   sol = self.init_solution(Q);
            %       Initializes a solution, given an initial guess of
            %       the soft assignment matrix Q.
            %
            %       Q should be a matrix of size [K, n], and each column
            %       of Q sums to one.
            %
            %   Note: the components and logliks are set to empty.
            %
            
            n = self.nobs;
            
            if isscalar(K)
                Q = rand(K, n);
                Q = bsxfun(@times, Q, 1 ./ sum(Q, 1));
                
            elseif ismatrix(K)
                Q = K;
                if size(Q, 2) ~= n
                    error('fmm_init:invalidarg', ...
                        'The size of Q0 is inconsistent with obs.');
                end
                K = size(Q, 1);
            else
                error('fmm_init:invalidarg', 'The third argument is invalid.');
            end
            
            pi = ones(K, 1) * (1 ./ double(K));
            
            sol.K = K;
            sol.n = self.nobs;
            sol.components = [];
            sol.pi = pi;
            sol.logliks = [];
            sol.Q = Q;
            
        end
                
        function v = eval_objv(self, sol)
            % Evaluates the objective at a solution
            %
            %   v = self.eval_objv(sol);
            %
            %       The input solution should be completed. 
            %       
            %   Note: the struct produced by the init_solution method
            %   is not completed, as components are initially empty.
            %   It would become completed after one update iteration.
            %
            
            v = fmm_em_evalobjv(self, sol);                                    
        end
        

        function sol = update(self, sol)
            % Updates a solution a single E-M iteration
            %
            %   sol = self.update_solution(sol);
            %
            %       This function first updates the component params 
            %       and pi with an M-step, and then the Q matrix with
            %       an E-step.
            %
            
            sol = fmm_em_update(self, sol);
        end
        
    end
    
    
end



