clc; clear; close all;

project;

%% Game-Theoretic Setup
max_iter = 100;  % Maximum iterations for best response dynamics
tol = 1e-5;      % Convergence tolerance
alpha = 0.02;     % Learning rate 

% inti flows randomly
x = randn(N,1) * 5; 
cost_history = zeros(max_iter,1);

for iter = 1:max_iter
    x_prev = x;
    
    % every node updates flow based on best response
    for i = 1:N
        % local cost gradient
        neighbors = find(A(i, :) == 1); % connected nodes
        if isempty(neighbors)
            continue; 
        end
        
        grad_i = sum(2 * R(i, neighbors)' .* (x(i) - x(neighbors)));
        
        % gradient descent (selfish behavior)
        x(i) = x(i) - alpha * grad_i;
    end
    
    % total system cost
    cost_history(iter) = sum(sum(R .* (x * x'))); 
    
    % Check convergence
    if norm(x - x_prev) < tol
        disp(['Game converged at iteration ', num2str(iter)]);
        cost_history = cost_history(1:iter);
        break;
    end
end

%% Results
fprintf('Final Flow Distribution:\n');
for i = 1:N
    fprintf('Node %d (%s): Flow = %.4f\n', i, node_type(i), x(i));
end

fprintf('Final Total Cost: %.5f\n', sum(sum(R .* (x * x'))));

% Plot cost evolution
figure;
plot(1:length(cost_history), cost_history, '-o', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Total Cost'); title('Cost Evolution in Game-Theoretic Optimization'); grid on;
