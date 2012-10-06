classdef test_pw_cosine < mtest_case
    % Unit testing of pw_cosine
    
    properties
        dim
        X
        Y
    end
    
    methods
        
        function obj = test_pw_cosine(d, m, n)
            obj.dim = d;
            obj.X = rand(d, m);
            obj.Y = rand(d, n);            
        end
        
        function test_cosine(self)
            D0 = self.safe_dist(self.X, self.Y);
            D = pli_pw_cosine(self.X, self.Y);
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_cosine_selfpw(self)
            D0 = self.safe_dist(self.X, self.X);
            D = pli_pw_cosine(self.X);
            assert( mtest_is_approx(D, D0) );          
        end
    end
       
    
    methods(Static)
        
        function D = safe_dist(X, Y)
            
            m = size(X, 2);
            n = size(Y, 2);
            
            D = zeros(m, n);
            for j = 1 : n
                for i = 1 : m
                    xi = X(:,i);
                    yj = Y(:,j);
                    D(i, j) = 1 - (xi' * yj) / (norm(xi) * norm(yj));
                end
            end            
        end
        
    end
    
end