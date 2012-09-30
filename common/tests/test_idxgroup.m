classdef test_idxgroup < mtest_case
    
    methods
    
        function test_idxgroup_normal(~)
           
            K = 5;
            n = 1000;
            L = randi(5, [1 n]);
            
            G1_t = test_idxgroup.idxgroup_safe(K, L);
            [~,~,~,G1] = idxgroup(K, L);
            assert( isequal(G1_t, G1) );
            
            
            G2_t = test_idxgroup.idxgroup_safe(K+1, L);
            [~,~,~,G2] = idxgroup(K+1, L);
            assert( isequal(G2_t, G2) );
            
            G3_t = test_idxgroup.idxgroup_safe(K-1, L);
            [~,~,~,G3] = idxgroup(K-1, L);
            assert( isequal(G3_t, G3) );            
        end                         
        
    end
    
    
    methods(Static)
        
        function gs = idxgroup_safe(K, L)
            gs = cell(K, 1);
            for k = 1 : K
                gs{k} = find(L == k);
            end
        end
        
    end
end

