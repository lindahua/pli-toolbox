classdef test_aggregx < mtest_case
    
    properties
        X
    end
    
    
    methods
        function obj = test_aggregx(m, n)
            obj.X = randi(100, [m, n]);
        end
        
        function s = name(self)
            [m, n] = size(self.X);
            s = sprintf('%s(%d, %d)', class(self), m, n);
        end
        
        
        function test_sum_by_scalars(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [m, n]);
            
            A0 = zeros(K, 1);
            for k = 1 : K
                A0(k) = sum(X_(I == k));
            end
            
            A = aggregx(K, X_, I);
            
            assert( isequal(A, A0) );
        end        
                
        function test_max_by_scalars(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [m, n]);
            
            A0 = zeros(K, 1);
            for k = 1 : K
                v = max(X_(I == k));
                if isempty(v)
                    v = -inf;
                end
                A0(k) = v;
            end
            
            A = aggregx(K, X_, I, 'max');
            assert( isequal(A, A0) );
        end
                        
        function test_min_by_scalars(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [m, n]);
            
            A0 = zeros(K, 1);
            for k = 1 : K
                v = min(X_(I == k));
                if isempty(v)
                    v = inf;
                end
                A0(k) = v;
            end
            
            A = aggregx(K, X_, I, 'min');
            
            assert( isequal(A, A0) );
        end
        
                
        function test_sum_by_rows(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [m, 1]);
            
            A0 = zeros(K, n);
            for k = 1 : K
                A0(k, :) = sum(X_(I == k, :), 1);
            end
            
            A = aggregx(K, X_, I);
            
            assert( isequal(A, A0) );
        end
                
        function test_max_by_rows(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [m, 1]);
            
            A0 = zeros(K, n);
            for k = 1 : K
                v = max(X_(I == k, :), [], 1);
                if isempty(v)
                    v = -inf;
                end
                A0(k, :) = v;
            end
            
            A = aggregx(K, X_, I, 'max');
            
            assert( isequal(A, A0) );
        end
                        
        function test_min_by_rows(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [m, 1]);
            
            A0 = zeros(K, n);
            for k = 1 : K
                v = min(X_(I == k, :), [], 1);
                if isempty(v)
                    v = inf;
                end
                A0(k, :) = v;
            end
            
            A = aggregx(K, X_, I, 'min');
            
            assert( isequal(A, A0) );
        end
        
                
        function test_sum_by_columns(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [1, n]);
            
            A0 = zeros(m, K);
            for k = 1 : K
                A0(:, k) = sum(X_(:, I==k), 2);
            end
            
            A = aggregx(K, X_, I);            
            assert( isequal(A, A0) );
        end
                
        function test_max_by_columns(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [1, n]);
            
            A0 = zeros(m, K);
            for k = 1 : K                
                v = max(X_(:, I==k), [], 2);
                if isempty(v)
                    v = -inf;
                end
                A0(:, k) = v;
            end
            
            A = aggregx(K, X_, I, 'max');
            
            assert( isequal(A, A0) );
        end
        
        function test_min_by_columns(self)
            X_ = self.X;
            [m, n] = size(X_);
            
            K = 10;
            I = randi(K + 2, [1, n]);
            
            A0 = zeros(m, K);
            for k = 1 : K                
                v = min(X_(:, I==k), [], 2);
                if isempty(v)
                    v = inf;
                end
                A0(:, k) = v;
            end
            
            A = aggregx(K, X_, I, 'min');
            
            assert( isequal(A, A0) );
        end        
        
    end
    
end