classdef pli_lda_vinfer_problem < handle
    % Latent Dirichlet Allocation variational inference
    %
    %   A solution for this problem is a struct with following fields:
    %
    %   - h :       The document histogram
    %   
    %   - gamma :   The variational Dirichlet parameter, size = [K 1]
    %   - phi :     The per-word topic distribution, size = [V K]
    %
    %   - elog_theta:   expectation of log(theta) w.r.t. Dir(gamma), [K 1]
    %   - toc_weights:  expectation of topic weights w.r.t. phi, [K 1]
    %
    
    
    properties(GetAccess='public', SetAccess='private')
        ntopics;    % The number of topics (K)
        vsize;      % The size of vocabulary (V)
        
        alpha;      % The Dirichlet prior parameter, size = [K, 1]
        unigrams;   % The topic unigrams, size = [V, K]
        
        lnB;            % log-Beta value for the Dirichlet prior
        log_unigrams;   % log of the unigrams, size = [V, K]
    end
    
    
    %% Problem construction
    
    methods
        
        function obj = pli_lda_vinfer_problem(a, U)
            % Constructs a problem instance
            %
            %   obj = pli_lda_vinfer_problem(a, U);
            %
            %   Arguments
            %   ---------
            %   - a :       The Dirichlet prior parameter
            %   - U :       The topic unigrams
            %
            
            % check arguments
            
            if ~(isfloat(a) && isreal(a) && ismatrix(a) && size(a,2) == 1)
                error('pli_lda_vinfer_problem:invalidarg', ...
                    'a should be a real column vector.');
            end
            
            if ~(isfloat(U) && isreal(U) && ismatrix(U))
                error('pli_lda_vinfer_problem:invalidarg', ...
                    'U should be a real matrix.');
            end
            
            V = size(U, 1);
            K = size(U, 2);
            
            if length(a) ~= K
                error('pli_lda_vinfer_problem:invalidarg', ...
                    'The dimensions of a and U are inconsistent.');
            end
            
            % set fields
            
            obj.ntopics = K;
            obj.vsize = V;
            
            obj.alpha = a;
            obj.unigrams = U;
            
            obj.lnB = pli_mvbetaln(a);
            obj.log_unigrams = log(U);                               
        end

    end
    
    
    %% Methods for variational inference
    
    methods
        
        function v = eval_objv(self, sol)
            % Evaluate the objective value of a solution
            %
            %   v = self.eval_obj(sol);
            %
            
            h = sol.h;
            
            gam = sol.gamma;
            phi = sol.phi;
            elog_theta = sol.elog_theta;
                        
            % E log(theta | alpha)
                                    
            ell_theta = (self.alpha - 1)' * elog_theta - self.lnB;
            
            % E log(z | theta)
            
            ell_z = sol.toc_weights' * elog_theta;
            
            % E log(w | z; U)
            
            ell_w = sum(h' * (phi .* self.log_unigrams));
            
            % entropy of gamma
            
            ent_gam = pli_dirichlet_entropy(gam, elog_theta);
            
            % entropy of phi
            
            ent_phi = h' * pli_ddentropy(phi, 2);
            
            % overall
            
            v = ell_theta + ell_z + ell_w + ent_gam + ent_phi;            
        end
        
        
        function sol = init_solution(self, h)
            % Initialize an LDA inference solution
            %
            %   sol = self.init_solution(h);
            %      
            %       Initializes a solution for a document. 
            %
            %   Arguments
            %   ---------
            %   - h :   The word histogram of the document.
            %
            %   This method initializes gamma while leaving phi empty.
            %
            
            V = self.vsize;
            K = self.ntopics;
            
            if ~(isfloat(h) && isreal(h) && isequal(size(h), [V 1]))
                error('pli_lda_vinfer_problem:invalidarg', ...
                    'h should be a real vector of size [V 1]');
            end
            
            sol.h = h; 
            
            gam = self.alpha + sum(h) / K;
            sol.gamma = gam;            
            sol.elog_theta = psi(gam) - psi(sum(gam));
            
            sol.phi = [];
            sol.toc_weights = [];
        end
        
        
        function sol = update(self, sol)
            % Updates a solution
            %
            %   sol = self.update(sol);
            %
            %   This method first computes phi based on gamma, and
            %   then updates gamma based on phi.
            %
            
            % Perform update
            
            [gam, phi, toc_w, elog_theta] = pli_lda_vinfer_update( ...
                self.alpha, self.log_unigrams, sol.h, sol.gamma, 1);            
            
            % write to sol
            
            sol.gamma = gam;
            sol.elog_theta = elog_theta;
            
            sol.phi = phi;
            sol.toc_weights = toc_w;            
        end

    end
    
    
end

