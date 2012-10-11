function [w, w0] = svm_sgdx_refrun(fun, X, y, t0, K, w, w0)
%A skeleton function to run a SGD svm learning algorithm

n = size(X, 2);

if K == 1
    t = t0;
    for i = 1 : n
        t = t + 1;
        Xi = X(:,i);
        yi = y(i);
        [w, w0] = fun(Xi, yi, t, w, w0); 
    end
else
    t = t0;
    rn = n;
    i = 0;
    
    while rn >= K
        t = t + 1;
        
        Xi = X(:, i+1:i+K);
        yi = y(i+1:i+K);
        
        [w, w0] = fun(Xi, yi, t, w, w0);
                
        i = i + K;
        rn = rn - K;
    end
    
    if rn > 0
        t = t + 1;
        
        Xi = X(:, i+1:i+rn);
        yi = y(i+1:i+rn);
        
        [w, w0] = fun(Xi, yi, t, w, w0);
    end
end
    