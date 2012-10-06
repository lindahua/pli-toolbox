classdef test_aggreg2 < mtest_case
    
    properties
        K, L
        X
    end
    
    
    methods
        function obj = test_aggreg2()
            m = 20;
            n = 25;
            obj.K = 3;
            obj.L = 4;
            
            obj.X = randi(100, [m n]);
        end
        
        
        function test_sum_mm(self)
            self.dotest('m', 'm', 'sum');
        end
        
        function test_sum_mc(self)
            self.dotest('m', 'c', 'sum');
        end
        
        function test_sum_mr(self)
            self.dotest('m', 'r', 'sum');
        end   
        
        function test_sum_cm(self)
            self.dotest('c', 'm', 'sum');
        end
        
        function test_sum_cc(self)
            self.dotest('c', 'c', 'sum');
        end
        
        function test_sum_cr(self)
            self.dotest('c', 'r', 'sum');
        end
        
        function test_sum_rm(self)
            self.dotest('r', 'm', 'sum');
        end
        
        function test_sum_rc(self)
            self.dotest('r', 'c', 'sum');
        end
        
        function test_sum_rr(self)
            self.dotest('r', 'r', 'sum');
        end   
        
        
        function test_max_mm(self)
            self.dotest('m', 'm', 'max');
        end
        
        function test_max_mc(self)
            self.dotest('m', 'c', 'max');
        end
       
        function test_max_mr(self)
            self.dotest('m', 'r', 'max');
        end   
        
        function test_max_cm(self)
            self.dotest('c', 'm', 'max');
        end
        
        function test_max_cc(self)
            self.dotest('c', 'c', 'max');
        end
        
        function test_max_cr(self)
            self.dotest('c', 'r', 'max');
        end
        
        function test_max_rm(self)
            self.dotest('r', 'm', 'max');
        end
        
        function test_max_rc(self)
            self.dotest('r', 'c', 'max');
        end
        
        function test_max_rr(self)
            self.dotest('r', 'r', 'max');
        end 
        
        
        function test_min_mm(self)
            self.dotest('m', 'm', 'min');
        end
        
        function test_min_mc(self)
            self.dotest('m', 'c', 'min');
        end
       
        function test_min_mr(self)
            self.dotest('m', 'r', 'min');
        end   
        
        function test_min_cm(self)
            self.dotest('c', 'm', 'min');
        end
        
        function test_min_cc(self)
            self.dotest('c', 'c', 'min');
        end
        
        function test_min_cr(self)
            self.dotest('c', 'r', 'min');
        end
        
        function test_min_rm(self)
            self.dotest('r', 'm', 'min');
        end
        
        function test_min_rc(self)
            self.dotest('r', 'c', 'min');
        end
        
        function test_min_rr(self)
            self.dotest('r', 'r', 'min');
        end         
        
        
        function dotest(self, sI, sJ, op)
            
            X_ = self.X;
            [m, n] = size(X_);
            
            K_ = self.K;
            L_ = self.L;
            
            % prepare indicues
            
            [I, I0] = test_aggreg2.gen_inds(sI, K_, m, n);
            [J, J0] = test_aggreg2.gen_inds(sJ, L_, m, n);
            
            A0 = zeros(K_, L_);
            
            % make ground-truth
            
            switch op
                case 'sum'
                    for l = 1 : L_
                        for k = 1 : K_
                            A0(k, l) = sum(X_(I0 == k & J0 == l));
                        end
                    end
                    
                case 'max'
                    for l = 1 : L_
                        for k = 1 : K_
                            v = max(X_(I0 == k & J0 == l));
                            if isempty(v)
                                v = -inf;
                            end
                            A0(k, l) = v;
                        end
                    end
                    
                case 'min'
                    for l = 1 : L_
                        for k = 1 : K_
                            v = min(X_(I0 == k & J0 == l));
                            if isempty(v)
                                v = inf;
                            end
                            A0(k, l) = v;
                        end
                    end
            end
            
            % run function
            
            A = pli_aggreg2([K_, L_], X_, I, J, op);
            
            % verify
            
            assert( isequal(A, A0) );
        end
        
        
    end
    
    
    methods(Static)
        
        function [I, I0] = gen_inds(s, K, m, n)
            
            if s == 'm'
                I = randi(K, [m n]);
                I0 = I;
            elseif s == 'c'
                I = randi(K, [m 1]);
                I0 = I(:, ones(1, n));
            elseif s == 'r'
                I = randi(K, [1 n]);
                I0 = I(ones(m, 1), :);
            end
            
        end
        
    end
    
end

