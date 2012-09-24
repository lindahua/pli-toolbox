classdef fmm_em_problem < handle
    %FMM_EM_PROBLEM The problem to estimate finite mixtures using EM
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
        
        function obj = fmm_em_problem(model, pric)
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
            
            if ~isa(model, 'pmodel_base')
                error('fmm_em_problem:invalidarg', ...
                    'model should be an instance of pmodel_base.');
            end
            
            if nargin < 2
                pric = 0;
            else
                if ~(isreal(pric) && isscalar(pric) && pric >= 0)
                    error('fmm_em_problem:invalidarg', ...
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
                    error('fmm_em_problem:invalidarg', ...
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
            
            mdl = self.model;
            w = self.weights;
            pric = self.pricount;
            
            Q = sol.Q;
            n = size(Q, 2);
            
            if isempty(w)
                w = ones(n, 1);
            end
            
            % sum of observation log-likelihoods
            
            L = sol.logliks;
            if isempty(L)
                L = mdl.evaluate_loglik(sol.components, self.obs);
            end
            
            sll_obs = sum(L .* Q, 1) * w;
            
            % sum of assignment log-likelihoods
            
            log_pi = log(sol.pi);
            sll_q = log_pi' * (Q * w);
            
            % sum of component log-priors
            
            lpri_comp = mdl.evaluate_logpri(sol.components);
            lpri_comp = sum(lpri_comp);

            % the log-prior of pi
            
            if pric > 0
                lpri_pi = pric * sum(log_pi);
            else
                lpri_pi = 0;
            end
                        
            % entropy of Q
            
            ent_q = - (nansum(Q .* log(Q), 1) * w);
            
            % overall
            
            v = sll_obs + sll_q + lpri_comp + lpri_pi + ent_q;
                                    
        end
        
        
        
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
        

        function sol = update(self, sol)
            % Updates a solution a single E-M iteration
            %
            %   sol = self.update_solution(sol);
            %
            %       This function first updates the component params 
            %       and pi with an M-step, and then the Q matrix with
            %       an E-step.
            %
            
            mdl = self.model;
            n = self.nobs;
            O = self.obs;
            w = self.weights;
            pric = self.pricount;
            
            % M-step
            
            if isempty(w)
                pi = sol.Q * ones(n, 1);
                W = sol.Q.';
            else
                pi = sol.Q * w;
                W = bsxfun(@times, sol.Q.', w);
            end
            
            if pric > 0
                pi = pi + pric;
            end            
            pi = pi / sum(pi);
            
            params = mdl.estimate_param(O, W);
            L = mdl.evaluate_loglik(params, O);
            
            % E-step
            
            Q = normalize_exp(bsxfun(@plus, L, log(pi)), 1);
            
            % write update to sol
            
            sol.pi = pi;
            sol.components = params;
            sol.logliks = L;
            sol.Q = Q;                        
        end
        
    end
    
    
end



