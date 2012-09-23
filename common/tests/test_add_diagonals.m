classdef test_add_diagonals < mtest_case
    % Unit-testing of add_diagonals
    %
    
    properties
        A
    end
    
    methods
        
        function obj = test_add_diagonals(m, n)
            obj.A = reshape(1:m*n, m, n);
        end
        
        function s = name(self)
            c = class(self);
            A_ = self.A;
            s = sprintf('%s(%d, %d)', c, size(A_, 1), size(A_, 2));
        end        
        
        function test_add_scalar(self)
            v = 2;
            R = add_diagonals(self.A, v);
            R0 = self.safe_result(v);
            
            assert( isequal(R, R0) );
        end
        
        function test_add_vector(self)
            n = min(size(self.A));
            v = 1 : n;
            
            R = add_diagonals(self.A, v);
            R0 = self.safe_result(v);
            
            assert( isequal(R, R0) );
        end
    end
    
    
    methods(Access='private')
        
        function R = safe_result(self, v)
            n = min(size(self.A));
            if isscalar(v)
                v = v * ones(n, 1);
            end
            
            R = self.A;
            for i = 1 : n
                R(i,i) = R(i,i) + v(i);
            end
        end

    end
    
end