classdef test_pw_qdist < mtest_case
    % Unit testing of pli_pw_qdist
    
    properties
        dim
        form
        Q
        Qmat
        X
        Y
    end
    
    methods
                        
        function obj = test_pw_qdist(f, d, m, n)
            
            obj.dim = d;
            obj.form = f;
            
            if f == 0
                obj.Q = 0.5 + rand();
                obj.Qmat = obj.Q * eye(d);
            elseif f == 1
                obj.Q = 0.5 + rand(d, 1);
                obj.Qmat = diag(obj.Q);
            else
                T = randn(d, d);
                obj.Q = pli_adddiag(T * T', 0.5);
                obj.Qmat = obj.Q;
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
            D = pli_pw_qdist(self.X, self.Y, self.Q, 'sq');
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_qdist_sq_selfpw(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.X);
            D = pli_pw_qdist(self.X, [], self.Q, 'sq');
            assert( mtest_is_approx(D, D0) );          
        end
                
        function test_qdist(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.Y);
            D0 = sqrt(D0);
            D = pli_pw_qdist(self.X, self.Y, self.Q);
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_qdist_selfpw(self)
            D0 = self.safe_sqdist(self.Qmat, self.X, self.X);
            D0 = sqrt(D0);
            D = pli_pw_qdist(self.X, [], self.Q);
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