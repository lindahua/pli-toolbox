classdef test_ktop < mtest_case
    
    properties
        m = 28;
        n = 32;
        k = 9;
    end
    
    methods
        
        function test_on_column(self)
            
            x = rand(self.m, 1);
            
            [y0, I0] = sort(x, 1, 'ascend');
            y0 = y0(1:self.k);
            I0 = I0(1:self.k);
            
            I = pli_ktop(x, self.k, 'smallest');
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(x, self.k, 'smallest');
            assert( isequal(I, I0) );
            assert( isequal(y, y0) );
                                    
            [y0, I0] = sort(x, 1, 'descend');
            y0 = y0(1:self.k);
            I0 = I0(1:self.k);
            
            I = pli_ktop(x, self.k, 'largest');
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(x, self.k, 'largest');
            assert( isequal(I, I0) );
            assert( isequal(y, y0) );
            
        end
        
                
        function test_on_row(self)
            
            x = rand(1, self.n);
                                    
            [y0, I0] = sort(x, 2, 'ascend');
            y0 = y0(1:self.k);
            I0 = I0(1:self.k);
                                    
            I = pli_ktop(x, self.k, 'smallest');
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(x, self.k, 'smallest');
            assert( isequal(I, I0) );
            assert( isequal(y, y0) );
            
            [y0, I0] = sort(x, 2, 'descend');
            y0 = y0(1:self.k);
            I0 = I0(1:self.k);
            
            I = pli_ktop(x, self.k, 'largest');
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(x, self.k, 'largest');
            assert( isequal(I, I0) );
            assert( isequal(y, y0) );            
            
        end
        
                
        function test_on_dim1(self)
            
            X = rand(self.m, self.n);
            
            [Y0, I0] = sort(X, 1, 'ascend');
            Y0 = Y0(1:self.k, :);
            I0 = I0(1:self.k, :);
            
            I = pli_ktop(X, self.k, 'smallest', 1);
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(X, self.k, 'smallest', 1);
            assert( isequal(I, I0) );
            assert( isequal(y, Y0) );
                                    
            [Y0, I0] = sort(X, 1, 'descend');
            Y0 = Y0(1:self.k, :);
            I0 = I0(1:self.k, :);
            
            I = pli_ktop(X, self.k, 'largest', 1);
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(X, self.k, 'largest', 1);
            assert( isequal(I, I0) );
            assert( isequal(y, Y0) );
            
        end
        
                
        function test_on_dim2(self)
            
            X = rand(self.m, self.n);
            
            [Y0, I0] = sort(X, 2, 'ascend');
            Y0 = Y0(:, 1:self.k);
            I0 = I0(:, 1:self.k);
            
            I = pli_ktop(X, self.k, 'smallest', 2);
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(X, self.k, 'smallest', 2);
            assert( isequal(I, I0) );
            assert( isequal(y, Y0) );
                                    
            [Y0, I0] = sort(X, 2, 'descend');
            Y0 = Y0(:, 1:self.k);
            I0 = I0(:, 1:self.k);
            
            I = pli_ktop(X, self.k, 'largest', 2);
            assert( isequal(I, I0) );
            
            [I, y] = pli_ktop(X, self.k, 'largest', 2);
            assert( isequal(I, I0) );
            assert( isequal(y, Y0) );
            
        end
        
    
    end
    
end
