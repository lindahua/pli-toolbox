classdef gauss_model < pmodel_base
    % Gaussian generative model
    
    properties
        cform;
        dim;
    end
    
    methods
        
        function obj = gauss_model(d, cf)
            
            % verify arguments
            
            if ~(isscalar(d) && isreal(d) && d == fix(d) && d >= 1)
                error('gauss_model:invalidarg', ...
                    'd should be a positive integer scalar.');
            end
            
            cform_ok = 0;
            if ischar(cf) && ...
                    (isscalar(cf) || strcmp(cf(2:end), '-tied'))
                
                if cf(1) == 's' || cf(1) == 'd' || cf(1) == 'f'
                    cform_ok = 1;
                end                
            end
            
            if ~cform_ok
                error('gauss_model:invalidarg', ...
                    'The value of cf is invalid.');
            end
            
            % set properties
            
            obj.cform = cf;
            obj.dim = d;
        end
        
    end    
    
    
    methods                
        
        function n = check_observations(self, X)
            
            if isfloat(X) && isreal(X) && ismatrix(X)
                if size(X, 1) == self.dim
                    n = size(X, 2);
                else
                    error('gauss_model:invalidarg', ...
                        'The size of X is invalid.');
                end
            else
                error('gauss_model:invalidarg', ...
                    'X should be a real matrix.');
            end
        end
        
        
        function G = estimate_param(self, X, weights, ~)
            
            G = gauss_mle(X, weights, self.cform);
        end
        
        function L = evaluate_loglik(~, G, X)
            
            L = gauss_logpdf(G, X);
        end
        
        function L = evaluate_logpri(~, ~)
            
            L = 0;
        end
        
    end
    
end