classdef pli_gauss_model < pli_pmodel_base
    % Gaussian generative model
    
    properties
        cform;
        dim;  
        tie_cov; 
    end
    
    methods
        
        function obj = pli_gauss_model(d, cf)
            
            % verify arguments
            
            if ~(isscalar(d) && isreal(d) && d == fix(d) && d >= 1)
                error('pli_gauss_model:invalidarg', ...
                    'd should be a positive integer scalar.');
            end
            
            cform_ok = 0;
            if ischar(cf) && isscalar(cf)                
                if cf == 's' || cf == 'd' || cf == 'f'
                    cform_ok = 1;
                    tiec = false;
                end                
            elseif strcmp(cf(2:end), '-tied')
                cf = cf(1);
                if cf == 's' || cf == 'd' || cf == 'f'
                    cform_ok = 1;
                    tiec = true;
                end
            end
            
            if ~cform_ok
                error('pli_gauss_model:invalidarg', ...
                    'The value of cf is invalid.');
            end
            
            % set properties
            
            obj.cform = cf;
            obj.dim = d;
            obj.tie_cov = tiec;
        end
        
    end    
    
    
    methods                
        
        function n = check_observations(self, X)
            
            if isfloat(X) && isreal(X) && ismatrix(X)
                if size(X, 1) == self.dim
                    n = size(X, 2);
                else
                    error('pli_gauss_model:invalidarg', ...
                        'The size of X is invalid.');
                end
            else
                error('pli_gauss_model:invalidarg', ...
                    'X should be a real matrix.');
            end
        end
        
        
        function G = update_params(self, X, weights, sidx, ~)
            
            if ~isempty(sidx)
                X = X(:, sidx);
            end            
            
            if self.tie_cov
                G = pli_gauss_mle(X, weights, self.cform, 'tie-cov');
            else
                G = pli_gauss_mle(X, weights, self.cform);
            end
        end
        
        function L = evaluate_loglik(~, G, X)
            
            L = pli_gauss_logpdf(G, X);
        end
        
        function L = evaluate_logpri(~, ~)
            
            L = 0;
        end
        
    end
    
end