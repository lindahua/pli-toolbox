function y = log1p_exp(x)
%LOG1P_EXP Calculates log(1 + exp(x)) in a robust way
%

y = zeros(size(x));

is_p = x > 0;
sp = find(is_p);
sn = find(~is_p);

y(sp) = x(sp) + log1p(exp(-x(sp)));
y(sn) = log1p(exp(x(sn)));
