classdef test_pw_euclidean < mtest_case
    % Unit testing of pw_euclidean
    
    properties
        dim
        X
        Y
    end
    
    methods
        
        function obj = test_pw_euclidean(d, m, n)
            obj.dim = d;
            obj.X = rand(d, m);
            obj.Y = rand(d, n);            
        end
        
        function test_sqeucdist(self)
            D0 = self.safe_sqdist(self.X, self.Y);
            D = pli_pw_euclidean(self.X, self.Y, 'sq');
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_sqeucdist_selfpw(self)
            D0 = self.safe_sqdist(self.X, self.X);
            D = pli_pw_euclidean(self.X, [], 'sq');
            assert( mtest_is_approx(D, D0) );          
        end
                
        function test_eucdist(self)
            D0 = self.safe_sqdist(self.X, self.Y);
            D0 = sqrt(D0);
            D = pli_pw_euclidean(self.X, self.Y);
            assert( mtest_is_approx(D, D0) );          
        end
        
        function test_eucdist_selfpw(self)
            D0 = self.safe_sqdist(self.X, self.X);
            D0 = sqrt(D0);
            D = pli_pw_euclidean(self.X);
            assert( mtest_is_approx(D, D0) );          
        end
    end
       
    
    methods(Static)
        
        function D = safe_sqdist(X, Y)
            
            m = size(X, 2);
            n = size(Y, 2);
            
            D = zeros(m, n);
            for j = 1 : n
                for i = 1 : m
                    D(i, j) = sum((X(:,i) - Y(:,j)).^2);
                end
            end            
        end
        
    end
    
end
