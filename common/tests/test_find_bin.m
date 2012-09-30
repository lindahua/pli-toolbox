classdef test_find_bin < mtest_case
    
    methods
        
        function test_nonsorted(~)
            n = 1000;
            e = [5, 13, 25];
            x = randi(30, [n 1]);
            
            r = find_bin(e, x);
            r0 = test_find_bin.find_bin_safe(e, x);
            
            assert( isequal(r, r0) );
            
        end
        
        function test_sorted(~)
            n = 1000;
            e = [5, 13, 25];
            x = randi(30, [n 1]);
            x = sort(x);
            
            r = find_bin(e, x, true);
            r0 = test_find_bin.find_bin_safe(e, x);
            
            assert( isequal(r, r0) );
        end
    
    end
    
    
    methods(Static)
        
        function r = find_bin_safe(e, x)
            r = zeros(size(x), 'int32');
            n = numel(x);
            
            for i = 1 : n
                xi = x(i);
                if xi < e(1)
                    r(i) = 0;
                else
                    r(i) = find(xi >= e, 1, 'last');
                end                    
            end            
        end
    end
    
end

