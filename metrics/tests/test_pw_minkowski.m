classdef test_pw_minkowski < mtest_case
    % Unit testing of pw_minkowski
    
    properties
        p
        dim
        X
        Y
    end
    
    methods
        
        function obj = test_pw_minkowski(p, d, m, n)
            obj.p = p;
            obj.dim = d;
            obj.X = rand(d, m);
            obj.Y = rand(d, n);            
        end
        
        function s = name(self)
            s = sprintf('%s (p = %g)', class(self), self.p);
        end        
        
        function test_minkowski(self)
            D0 = self.safe_dist(self.X, self.Y, self.p);
            D = pw_minkowski(self.X, self.Y, self.p);
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_minkowski_selfpw(self)
            D0 = self.safe_dist(self.X, self.X, self.p);
            D = pw_minkowski(self.X, [], self.p);
            assert( mtest_is_approx(D, D0) );          
        end
    end
       
    
    methods(Static)
        
        function D = safe_dist(X, Y, p)
            
            m = size(X, 2);
            n = size(Y, 2);
            
            D = zeros(m, n);
            
            if isfinite(p)
                for j = 1 : n
                    for i = 1 : m
                        v = abs(X(:,i) - Y(:,j));
                        D(i, j) = sum(v .^ p) .^ (1/p);
                    end
                end
            else
                for j = 1 : n
                    for i = 1 : m
                        v = abs(X(:,i) - Y(:,j));
                        D(i, j) = max(v);
                    end
                end
            end
                
        end
        
    end
    
end