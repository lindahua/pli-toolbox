classdef test_pw_hamming < mtest_case
    % Unit testing of pw_hamming
    
    properties
        dim
        X
        Y
    end
    
    methods
        
        function obj = test_pw_hamming(d, m, n)
            K = 3;
            obj.dim = d;
            obj.X = randi(K, [d, m]);
            obj.Y = randi(K, [d, n]);            
        end
        
        function test_hamming(self)
            D0 = self.safe_dist(self.X, self.Y);
            D = pw_hamming(self.X, self.Y);
            assert( isequal(D, D0) );          
        end
        
        function test_hamming_selfpw(self)
            D0 = self.safe_dist(self.X, self.X);
            D = pw_hamming(self.X);
            assert( isequal(D, D0) );          
        end
    end
       
    
    methods(Static)
        
        function D = safe_dist(X, Y)
            
            m = size(X, 2);
            n = size(Y, 2);
            
            D = zeros(m, n);
            for j = 1 : n
                for i = 1 : m
                    D(i, j) = sum(X(:,i) ~= Y(:,j));
                end
            end            
        end
        
    end
    
end