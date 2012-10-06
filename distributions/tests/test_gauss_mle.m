classdef test_gauss_mle < mtest_case
    % Unit testing of pli_gauss_mle
    %
    
    properties
        cform
        dim
        num
    end
    
    
    methods
    
        function obj = test_gauss_mle(cf, d, m)
            obj.cform = cf;
            obj.dim = d;
            obj.num = m;
        end
        
        
        function s = name(self)
            s = sprintf('%s [cf=%s, d=%d, m=%d]', ...
                class(self), self.cform, self.dim, self.num);
        end
        
        
        function test_mle(self)
            
            cf = self.cform;
            d = self.dim;
            m = self.num;
            
            n = 100;
            X = bsxfun(@times, randn(d, d) * randn(d, n), randn(d, 1));
            
            % test non-weighted case
            if m == 1
                G0 = test_gauss_mle.safe_mle(X, ones(n,1), cf);
                G = pli_gauss_mle(X, [], cf);
            
                assert(G.dim == G0.dim);
                assert(G.num == G0.num);
                assert(G.cform == G0.cform);
                assert(mtest_is_approx(G.mu, G0.mu));
                assert(mtest_is_approx(G.cov, G0.cov));
            end
            
            w = rand(n, m);
            G0 = test_gauss_mle.safe_mle(X, w, cf);
            G = pli_gauss_mle(X, w, cf);
            
            assert(G.dim == G0.dim);
            assert(G.num == G0.num);
            assert(G.cform == G0.cform);
            assert(mtest_is_approx(G.mu, G0.mu));
            
            assert(mtest_is_approx(G.cov, G0.cov));                                    
        end
        
        
        function test_mle_tiecov(self)
                                    
            cf = self.cform;
            d = self.dim;
            m = self.num;
            
            n = 100;
            X = bsxfun(@times, randn(d, d) * randn(d, n), randn(d, 1));
            
            cf_t = [cf '-tied'];
            
            w = rand(n, m);
            G0 = test_gauss_mle.safe_mle(X, w, cf_t);
            G = pli_gauss_mle(X, w, cf_t);
            
            assert(G.dim == G0.dim);
            assert(G.num == G0.num);
            assert(G.cform == G0.cform);
            assert(mtest_is_approx(G.mu, G0.mu));
            
            assert(mtest_is_approx(G.cov, G0.cov)); 
        end
                
    end
    
    
    methods(Static)
        function G = safe_mle(X, w, cform)
            
            if isscalar(cform)
                tie = 0;
            elseif length(cform) == 6
                tie = 1;
            end
            
            switch cform(1)
                case 's'
                    cf = 0;
                case 'd'
                    cf = 1;
                case 'f'
                    cf = 2;
            end
            
            d = size(X, 1);
            m = size(w, 2);
            
            if tie
                mc = 1;
            else
                mc = m;
            end
            
            sw = sum(w, 1);
            
            % estimate mu
            
            mu = zeros(d, m);
            for k = 1 : m
                mu(:,k) = X * (w(:,k) / sw(k));
            end
                        
            % estimate cov
            
            if ~tie
                C = zeros(d, d, m);
                for k = 1 : m
                    Z = bsxfun(@minus, X, mu(:,k));
                    Ck = (Z * bsxfun(@times, w(:,k), Z')) / sw(k);
                    C(:,:,k) = Ck;
                end
                
            else
                Z = cell(1, m);
                for k = 1 : m
                    Z{k} = bsxfun(@minus, X, mu(:,k));                    
                end
                Z = [Z{:}];
                
                C = (Z * bsxfun(@times, w(:), Z')) / sum(sw);
            end
            
            if cf == 0
                cov = zeros(mc, 1);
                for k = 1 : mc
                    cov(k) = mean(diag(C(:,:,k)));
                end
            elseif cf == 1
                cov = zeros(d, mc);
                for k = 1 : mc
                    cov(:,k) = diag(C(:,:,k));
                end
            else
                cov = C;
            end
            
            % make struct
                    
            G.dim = d;
            G.num = m;
            G.cform = cf;
            G.mu = mu;
            G.cov = cov;                        
        end
        
    end
    
    
end
