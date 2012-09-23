classdef test_gauss < mtest_case
    
    properties
        cform;
        dim;
        num;
        zero_mean;
        tie_cov;
        
        gauss;
        cov_mat;        
    end
    
    methods
        
        function obj = test_gauss(cf, d, m, zmean, tiecov)
            
            if m > 1 && zmean
                error('test_gauss:invalidarg', ...
                    'Can not set zmean when m > 1.');
            end
            
            obj.dim = d;
            obj.num = m;  
            obj.zero_mean = zmean;
            obj.tie_cov = tiecov;
            
            if tiecov
                mc = 1;
            else
                mc = m;
            end
            
            if zmean
                mu = 0;
            else
                mu = randn(d, m);
            end
            
            switch cf
                case 's'
                    cov = 1.0 + rand(mc, 1);
                    
                    Cmat = zeros(d, d, mc);
                    for k = 1 : mc
                        Cmat(:,:,k) = diag(ones(d,1) * cov(k));
                    end
                    
                    obj.cform = 0;
                    
                case 'd'
                    cov = 1.0 + rand(d, mc);
                    
                    Cmat = zeros(d, d, mc);
                    for k = 1 : mc
                        Cmat(:,:,k) = diag(cov(:,k));
                    end          
                    
                    obj.cform = 1;
                    
                case 'f'
                    cov = zeros(d, d, mc);                    
                    for k = 1 : mc
                        T = randn(d, d);
                        C = add_diagonals(T * T', 1.0);
                        cov(:,:,k) = C;
                    end
                    
                    Cmat = cov;
                    
                    obj.cform = 2;
            end
            
            if tiecov
                obj.gauss = make_gauss(d, mu, cov, 'tie_cov');
            else
                obj.gauss = make_gauss(d, mu, cov);
            end
            
            obj.cov_mat = Cmat;
            
        end
        
        
        function s = name(self)
            
            cf_syms = ['s', 'd', 'f'];
            cf_sym = cf_syms(self.cform + 1);                
            
            s = sprintf('%s [f=%c, d=%d, m=%d, zm=%d, tc=%d]', ...
                class(self), cf_sym, self.dim, self.num, ...
                self.zero_mean, self.tie_cov);
            
        end        
        
        
        function test_basics(self)
            
            G = self.gauss;
                      
            assert(isstruct(G));
            assert(isscalar(G));
            assert(strcmp(G.tag, 'gauss'));
            assert(G.dim == self.dim);
            assert(G.num == self.num);
            assert(G.cform == self.cform);
            
        end    
        
        
        function test_mahdist(self)
            
            G = self.gauss;
            Cmat = self.cov_mat;
            
            m = self.num;
            tc = self.tie_cov;
            
            n = 200;
            X = randn(self.dim, n) * 2.0;                        
            
            if m == 1
                sqD0 = test_gauss.safe_sqmahdist(X, G.mu, Cmat);
            else
                sqD0 = zeros(m, n);
                if tc
                    for k = 1 : m
                        sqD0(k, :) = ...
                            test_gauss.safe_sqmahdist(X, G.mu(:,k), Cmat);
                    end
                else
                    for k = 1 : m
                        sqD0(k, :) = ...
                            test_gauss.safe_sqmahdist(X, G.mu(:,k), Cmat(:,:,k));
                    end
                end
            end

            D0 = sqrt(sqD0);
            
            sqD = gauss_mahdist(G, X, 'sq');
            assert(mtest_has_size(sqD, [m n]));
            
            D = gauss_mahdist(G, X);
            assert(mtest_has_size(D, [m n]));           
            
            assert(mtest_is_approx(sqD, sqD0, 1.0e-10));
            assert(mtest_is_approx(D, D0, 1.0e-10));            
        end
        
        
        function test_logdet(self)
            
            G = self.gauss;
            Cmat = self.cov_mat;
            
            if self.tie_cov
                mc = 1;
            else
                mc = self.num;
            end
            
            if mc == 1
                v0 = pdm_logdet(Cmat);
            else
                v0 = zeros(mc, 1);
                for k = 1 : mc
                    v0(k) = pdm_logdet(Cmat(:,:,k));
                end
            end
            
            v = gauss_logdet(G);
            
            assert(mtest_has_size(v, [mc 1]));
            assert(mtest_is_approx(v, v0, 1.0e-10));
        
        end
        
        
        function test_logpdf(self)
            
            G = self.gauss;
            Cmat = self.cov_mat;
            
            m = self.num;
            tc = self.tie_cov;
            
            n = 200;
            X = randn(self.dim, n) * 2.0;                        
            
            if m == 1
                L0 = test_gauss.safe_logpdf(X, G.mu, Cmat);
            else
                L0 = zeros(m, n);
                if tc
                    for k = 1 : m
                        L0(k, :) = ...
                            test_gauss.safe_logpdf(X, G.mu(:,k), Cmat);
                    end
                else
                    for k = 1 : m
                        L0(k, :) = ...
                            test_gauss.safe_logpdf(X, G.mu(:,k), Cmat(:,:,k));
                    end
                end
            end   
            
            L = gauss_logpdf(G, X);
            assert(mtest_has_size(L, [m n]));
            
            assert(mtest_is_approx(L, L0, 1.0e-10));
        end
    
    end
    
    
    methods(Static)
        
        function v = safe_sqmahdist(X, mu, C)
            
            d = size(X, 1);
            if isequal(mu, 0)
                mu = zeros(d, 1);
            end
            
            invC = inv(C);
                        
            n = size(X, 2);
            v = zeros(1, n);
            for i = 1 : n
                z = X(:,i) - mu;
                v(i) = z' * (invC * z); %#ok<MINV>
            end
        end               
        
        
        function v = safe_logpdf(X, mu, C)
            
            d = size(X, 1);
            sqD = test_gauss.safe_sqmahdist(X, mu, C);
            ldcov = pdm_logdet(C);
            
            v = (-0.5) * (sqD + (d * log(2*pi) + ldcov));            
        end        
        
    end
        
        
end




