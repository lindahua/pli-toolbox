classdef test_intcount < mtest_case
    % Unit testinf for intcount
    %
    
    methods
       
        function test_intcount1d(~)
            
            % prepare data
            
            m = 8;            
            len = 1000;            
            I = randi(m, [len, 1]);
                        
            % ground-truth
            
            C0 = zeros(m, 1);
            for i = 1 : m
                C0(i) = sum(I == i);
            end
            
            % exact m
            
            C = intcount(m, I);
            assert( isequal(C, C0) );  
            
            % larger m
            
            mp = m + 2;
            
            Cp0 = zeros(mp, 1);
            Cp0(1:m) = C0;
            
            Cp = intcount(mp, I);
            assert( isequal(Cp, Cp0) );
            
            % smaller m
            
            mn = m - 2;
            
            Cn0 = C0(1:mn);
            
            Cn = intcount(mn, I);
            assert( isequal(Cn, Cn0) );
                        
        end
        
        
        function test_intcount2d(~)
            
            % prepare data
            
            m = 8;
            n = 9;
            len = 2000;
            
            I = randi(m, [len, 1]);
            J = randi(n, [len, 1]);
            
            % ground-truth
            
            C0 = zeros(m, n);
            for j = 1 : n
                for i = 1 : m
                    C0(i, j) = sum(I == i & J == j);
                end
            end
            
            % exact m, n
            
            C = intcount([m, n], I, J);
            assert( isequal(C, C0) );            
            
            % larger m, n
            
            mp = m + 2;
            np = n + 2;
            
            Cp0 = zeros(mp, np);
            Cp0(1:m, 1:n) = C0;
            
            Cp = intcount([mp, np], I, J);
            assert( isequal(Cp, Cp0) );
            
            % smaller m, n
            
            mn = m - 2;
            nn = n - 2;
            
            Cn0 = C0(1:mn, 1:nn);
            
            Cn = intcount([mn, nn], I, J);
            assert( isequal(Cn, Cn0) );            
        end

    end
    
end


