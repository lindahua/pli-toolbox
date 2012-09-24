function lda_vem_demo()
%LDA_VEM_DEMO Demonstrate the use of LDA with variational EM
%
%   LDA_VEM_DEMO;
%

%% Experiment configuration

V = 8;
K = 3;
doc_len = 200;
ndocs = 5;

%% Generate data

disp('Underlying model');
disp('========================');

% topics

U = rand(V, K);
U = bsxfun(@times, U, 1 ./ sum(U, 1));

for k = 1 : K
    print_vec(sprintf('Topic [%d]', k), '%.4f ', U(:,k));
end

disp(' ');

% Dirichlet prior

alpha = 1.2 + rand(K, 1) * 3.0;
print_vec('alpha', '%.4f ', alpha);

disp(' ');

% documents

disp('Documents');
disp('========================');

Theta = dirichlet_sample(alpha, ndocs);

for i = 1 : ndocs
    
    theta = Theta(:, i);
    p = U * theta;    
    h = mnrnd(doc_len, p);
    
    H(:,i) = h;
    print_vec(sprintf('Doc [%d].hist', i), '%3d ', h);
end
disp(' ');


%% Inference for individual documents

disp('Document inference');
disp('========================');

vi_problem = lda_vinfer_problem(alpha, U);

for i = 1 : ndocs
    
    h = H(:, i);
    sol = vi_problem.init_solution(h);
    
    sol = iter_optimize('maximize', ...
        @vi_problem.eval_objv, @vi_problem.update, sol, ...
        'display', 'off');
        
    print_vec(sprintf('Doc [%d].gamma', i), '%.4f ', sol.gamma); 
end

disp(' ');


function print_vec(title, spec, v)

fprintf('%s: ', title);

n = numel(v);
for j = 1 : n
    fprintf(spec, v(j));
end

fprintf('\n');




