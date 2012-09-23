classdef test_gauss_sample < mtest_case
    
    properties
        cform
        dim 
        
        gauss
        cov_mat
    end
        
    methods
        
        function obj = test_gauss_sample(cf, d)
            obj.cform = cf;
            obj.dim = d;
            
            mu = randn(d, 1);
            
            switch cf
                case 's'
                    cov = 1.0 + rand();
                    C = diag(ones(d, 1) * cov);
                case 'd'
                    cov = 1.0 + rand(d, 1);
                    C = diag(cov);
                case 'f'
                    T = randn(d, d);
                    cov = add_diagonals(T * T', 1.0);
                    C = cov;
            end
            
            obj.gauss = make_gauss(d, mu, cov);
            assert(obj.gauss.num == 1);
            
            obj.cov_mat = C;
        end
        
        
        function s = name(self)
            
            s = sprintf('%s [cf=%s, d=%d]', ...
                class(self), self.cform, self.dim);            
        end
        
        
        function test_sample(self)
            
            G = self.gauss;
            n = 1e6;
            
            X = gauss_sample(G, n);
            assert( mtest_has_size(X, [G.dim n]) );
            
            sample_mean = mean(X, 2);
            
            Z = bsxfun(@minus, X, sample_mean);
            sample_cov = (Z * Z') * (1 / n);
           
            assert( mtest_is_approx(sample_mean, G.mu, 0.02) );
            assert( mtest_is_approx(sample_cov, self.cov_mat, 0.02) ); 
        end

    end
    
end
