clc; clear; close all;

project;

% Sensitivity params
num_tests = 10;  
perturbation_scale = 0.2; % 20% purturb
cost_results = zeros(num_tests, 1);

% Vary R Matrix
for i = 1:num_tests
    R_perturbed = R .* (1 + perturbation_scale * (2 * rand(size(R)) - 1)); % Random variation
    x = rand(N,1) * 5;  % Reset initial flow
    H = eye(N); % Reset Hessian
    max_iter = 100; 
    tol = 1e-6; 

    % Re-run optimization with perturbed R
    for iter = 1:max_iter
        g = 2 * (R_perturbed' * (R_perturbed * x));  

        if norm(g) < tol
            break;
        end

        p_dir = -H * g;
        alpha = 0.1;
        x_new = x + alpha * p_dir;
        g_new = 2 * (R_perturbed' * (R_perturbed * x_new));
        s_vec = x_new - x;
        y_vec = g_new - g;
        rho = 1 / (y_vec' * s_vec);
        
        if rho > 0
            H = (eye(N) - rho * (s_vec * y_vec')) * H * (eye(N) - rho * (y_vec * s_vec')) + rho * (s_vec * s_vec');
        end
        
        x = x_new;
    end

    cost_results(i) = sum((R_perturbed * x) .^ 2);
end

% Sensitivity Results
figure;
plot(1:num_tests, cost_results, '-o', 'LineWidth', 1.5);
xlabel('Test Case'); ylabel('Final Pumping Cost');
title('Sensitivity Analysis on R');
grid on;

