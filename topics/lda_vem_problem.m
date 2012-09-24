classdef lda_vem_problem < handle
    % Estimating of an LDA model using Variational EM
    %
    %   A solution is a struct with following fields:
    %
    %   - K :           The number of topics
    %   - V :           The size of vocabulary
    %   - n :           The number of documents
    %
    %   - alpha :       The Dirichlet prior parameter, size=[K, 1]
    %   - U :           The estimated topic unigrams, size=[V, K]
    %   - logU :        Logarithm of U, size=[V, K]
    %   
    %   - Gam :         The matrix of per-doc gamma params, size=[K, n]
    %   - Elog_theta :  The matrix of per-doc E[log(theta)], size=[K, n]
    %   - accumPhi :    Accumulated phi matrix, size=[V, K]
    %   - tocW :        The matrix of per-doc topic weights, size=[K, n]
    %
    %   - gamma_ents :  The per-doc entropy of gamma, size = [1, n]
    %   - phi_ents :    The per-doc entropy of phi, size = [1, n]
    %   
    
    properties(GetAccess='public', SetAccess='private')
        vsize;      % The vocabulary size        
        pricount;   % The prior count of each word in a topic        
        doc_hists;  % The matrix of per-doc histograms, size = [V, n]
    end
    
    
    %% Problem construction and setting
    
    methods
        
        function obj = lda_vem_problem(V, pric)
            % Constructs a problem object
            %
            %   obj = lda_vem_problem(V);
            %   obj = lda_vem_problem(V, pric);
            %
            %   Arguments
            %   ---------
            %   - V :       The size of vocabulary
            %   - pric :    The prior count of each topic (default = 0)
            %
            
            if ~(isscalar(V) && V == fix(V) && V > 1)
                error('lda_vem_problem:invalidarg', ...
                    'V should be a positive integer scalar with V > 1.')
            end
            
            if nargin < 2
                pric = 0;
            else
                if ~(isscalar(pric) && isreal(pric) && pric >= 0)
                    error('lda_vem_problem:invalidarg', ...
                        'pric should be a non-negative real scalar.');
                end
                pric = double(pric);
            end
            
            obj.vsize = V;
            obj.pricount = pric;
        end
                
        
        function set_docs(self, H)
            % Set the training documents
            %
            %   self.set_docs(H);
            %
            %   Arguments
            %   ---------
            %   - H :   The matrix of word histograms, size = [V, n].
            %           Here, n is the number of documents.
            %
            %           In particular, H(:,i) is the histogram for the
            %           i-th document.
            %
            
            if ~(isfloat(H) && isreal(H) && ismatrix(H))
                error('lda_vem_problem:invalidarg', ...
                    'H should be a real matrix.');
            end
            
            if size(H, 1) ~= self.vsize
                error('lda_vem_problem:invalidarg', ...
                    'The dimension of H is invalid.');
            end
            
            self.doc_hists = H;            
        end
        
    end
    
    
    %% Methods for Variational EM
    
    methods
                    
        function v = eval_objv(self, sol)
            % Evaluates the objective of a solution
            %
            %   v = self.eval_objv(sol);
            %           
            
            Elog_theta = sol.Elog_theta;
            logU = sol.logU;
            
            % E log(theta | alpha)
            
            lnB = mvbetaln(sol.alpha);
            ell_theta = (sol.alpha - 1)' * Elog_theta - lnB;
            
            % E log(z | theta)
            
            ell_z = sum(sol.tocW .* Elog_theta, 1);
            
            % E log(w | z; U)
            
            sum_ell_w = sum(sum(sol.accumPhi .* logU, 1));
            
            % log(U | pri_c)
            
            pric = self.pricount;
            if pric > 0
                lpri_U = sum(sum(logU, 1)) * pric;
            else
                lpri_U = 0;
            end
            
            % overall
            
            v = sum(ell_theta) + ...
                sum(ell_z) + ...
                sum_ell_w + ...
                sum(sol.gamma_ents) + ...
                sum(sol.phi_ents) + ...
                lpri_U;
        end
        
        
        function sol = init_solution(self, U, alpha)
            % Initializes a solution 
            %
            %   sol = self.init_solution(U);
            %   sol = self.init_solution(U, alpha);
            %
            %   Arguments
            %   ---------
            %   - U :       An initial guess or pre-fixed set of topic
            %               unigrams, size = [V, K]
            %
            %   - alpha :   An initial guess of the Dirichlet param.
            %
            
            if ~(isfloat(U) && isreal(U) && ismatrix(U))
                error('lda_vem_problem:invalidarg', ...
                    'U should be a real matrix.');
            end
            
            V = self.vsize;
            
            if size(U, 1) ~= V
                error('lda_vem_problem:invalidarg', ...
                    'The dimension of U is incorrect.');
            end
            
            K = size(U, 2);
            n = size(self.doc_hists, 2);
            
            if n == 0
                error('lda_vem_problem:rterror', ...
                    'Documents have not been set to the problem.');
            end
            
            % set dimensions
                        
            sol.K = K; 
            sol.V = V;
            sol.n = n;
            
            % set alpha and U
            
            if nargin < 3
                alpha = ones(K, 1) * 2;
            else
                if ~(isfloat(alpha) && isreal(alpha) && isvector(alpha))
                    error('lda_vem_problem:invalidarg', ...
                        'alpha should be a real vector.');
                end
                
                if numel(alpha) ~= K
                    error('lda_vem_problem:invalidarg', ...
                        'The dimension of alpha is invalid.');
                end
                
                if size(alpha, 2) > 1
                    alpha = alpha.';
                end
            end
            
            sol.alpha = alpha;
            
            sol.U = U;
            sol.logU = log(U);
            
            % per-doc results empty
            
            sol.Gam = [];
            sol.Elog_theta = [];
            sol.accumPhi = [];
            sol.tocW = [];
            
            sol.gamma_ents = [];
            sol.phi_ents = [];            
        end
        
        
        function sol = update(self, sol, nv, uu)
            % Updates a solution
            %
            %   sol = self.update(sol);
            %   sol = self.update(sol, nv, uu);
            %
            %       Updates a solution according to a specified rule.
            %
            %   Arguments
            %   ---------
            %   - sol :     The solution struct.
            %
            %   - nv :      The number of variational iterations 
            %               at each cyle (before a model update).
            %               (default = 5).
            %
            %   - uu :      Whether to update unigrams. (default = true).
            %
            
            % argument checking
            
            if nargin < 3
                nv = 5;
            else
                if nv < 1
                    error('ldm_vem_problem:invalidarg', ...
                        'Argument nv for update should be at least 1.');
                end
            end
            
            if nargin < 4
                uu = true;
            end
            
            % useful variables
            
            H = self.doc_hists;
            K = sol.K;
            n = sol.n;
            
            logU = sol.logU;
            alpha = sol.alpha;
            
            % prepare gamma
            
            Gam = sol.Gam;
            if isempty(Gam)
                Gam = bsxfun(@plus, ones(K, 1) * (sum(H, 1) / K), alpha);
                Elog_theta = zeros(K, n);
            end
            
            % per-doc inference
            
            accumPhi = zeros(sol.V, K);
            tocW = zeros(K, n);
            
            gam_ents = zeros(1, n);
            phi_ents = zeros(1, n);
            
            for i = 1 : n
                h = H(:, i);
                
                % update phi and gam
                
                [gam, phi, toc_w, elog_t] = lda_vinfer_update( ...
                    alpha, logU, h, Gam(:, i), nv);
                
                % store/accumulate results
                
                Gam(:, i) = gam;
                Elog_theta(:, i) = elog_t;
                
                tocW(:, i) = toc_w;
                accumPhi = accumPhi + bsxfun(@times, h, phi);
                
                gam_ents(i) = dirichlet_entropy(gam);
                phi_ents(i) = h' * ddentropy(phi, 2);
            end
            
            sol.Gam = Gam;
            sol.Elog_theta = Elog_theta;
            sol.accumPhi = accumPhi;
            sol.tocW = tocW;
            
            sol.gamma_ents = gam_ents;
            sol.phi_ents = phi_ents;            
            
            % estimate alpha
            
            alpha = dirichlet_mle(Elog_theta, alpha, ...
                'is_log', true, 'verbose', false);
            
            sol.alpha = alpha;
            
            % estimate U
            
            if uu
                pric = self.pricount;
                if pric == 0
                    A = accumPhi;
                else
                    A = accumPhi + pric;
                end
                
                U = bsxfun(@times, A, 1 ./ sum(A, 1));
                
                sol.U = U;
                sol.logU = log(U);
            end
            
        end
            
    end
    
end



