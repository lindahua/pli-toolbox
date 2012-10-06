classdef test_pw_mahdist < mtest_case
    % Unit testing of pli_pw_mahdist
    
    properties
        dim
        form
        C
        Qmat
        X
        Y
    end
    
    methods
                        
        function obj = test_pw_mahdist(f, d, m, n)
            
            obj.dim = d;
            obj.form = f;
            
            if f == 0
                obj.C = 0.5 + rand();
                obj.Qmat = (1.0 / obj.C) * eye(d);
            elseif f == 1
                obj.C = 0.5 + rand(d, 1);
                obj.Qmat = diag(1.0 ./ obj.C);
            else
                T = randn(d, d);
                obj.C = pli_adddiag(T * T', 0.5);
                obj.Qmat = inv(obj.C);
            end
                                    
            obj.X = rand(d, m);
            obj.Y = rand(d, n);            
        end
        
        function s = name(self)
            s = sprintf('%s [f = %d, d = %d]', ...
                class(self), self.form, self.dim);
        end        
        
        function test_qdist_sq(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.Y);
            D = pli_pw_mahdist(self.X, self.Y, self.C, 'sq');
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_qdist_sq_selfpw(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.X);
            D = pli_pw_mahdist(self.X, [], self.C, 'sq');
            assert( mtest_is_approx(D, D0) );          
        end
                
        function test_qdist(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.Y);
            D0 = sqrt(D0);
            D = pli_pw_mahdist(self.X, self.Y, self.C);
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_qdist_selfpw(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.X);
            D0 = sqrt(D0);
            D = pli_pw_mahdist(self.X, [], self.C);
            assert( mtest_is_approx(D, D0) );          
        end
    end
       
    
    methods(Static)
        
        function D = safe_sqdist(Qmat, X, Y)
            
            m = size(X, 2);
            n = size(Y, 2);           
            
            D = zeros(m, n);
            for j = 1 : n
                for i = 1 : m
                    v = X(:,i) - Y(:,j);
                    D(i, j) = v' * Qmat * v;
                end
            end            
        end
        
    end
    
end