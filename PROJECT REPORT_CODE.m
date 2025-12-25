function chemical_reaction_simulation()
    % Numerical Simulation of Reversible and Competitive Reaction System
    % with Catalyst Deactivation
    % 
    % Reactions:
    % A <-> B (k1 = 2, k-1 = 0.4)
    % B -> C (k2 = 0.6)
    % B -> D (k3 = 0.2)
    % Catalyst deactivation: d(Cat)/dt = -kd * Cat (kd = 0.1)
    clc; clear; close all;
    k1 = 2;      % Forward rate constant A -> B
    k_1 = 0.4;   % Reverse rate constant B -> A
    k2 = 0.6;    % Rate constant B -> C
    k3 = 0.2;    % Rate constant B -> D
    kd = 0.1;    % Catalyst deactivation rate constant
    
    % Initial conditions: [A, B, C, D, Cat]
    y0 = [1, 0, 0, 0, 1];
    
    % Time parameters
    t_start = 0;
    t_end = 10;
    h = 0.01;    % Time step
    
    % Create time vector
    t = t_start:h:t_end;
    n = length(t);
    
    % Initialize solution matrices for different methods
    y_euler = zeros(n, 5);
    y_rk4 = zeros(n, 5);
    y_gauss_seidel = zeros(n, 5);
    y_gauss_elim = zeros(n, 5);
    
    % Set initial conditions
    y_euler(1, :) = y0;
    y_rk4(1, :) = y0;
    y_gauss_seidel(1, :) = y0;
    y_gauss_elim(1, :) = y0;
    
    % Define the system of ODEs
    function dydt = reaction_system(t_val, y)
        A = y(1); B = y(2); C = y(3); D = y(4); Cat = y(5);
        
        dydt = zeros(5, 1);
        dydt(1) = -k1 * A * Cat + k_1 * B * Cat;           % dA/dt
        dydt(2) = k1 * A * Cat - k_1 * B * Cat - k2 * B * Cat - k3 * B * Cat; % dB/dt
        dydt(3) = k2 * B * Cat;                            % dC/dt
        dydt(4) = k3 * B * Cat;                            % dD/dt
        dydt(5) = -kd * Cat;                               % dCat/dt
    end
    
    % Solve using different methods
    fprintf('Solving using different numerical methods...\n');
    
    %% 1. Euler's Method
    fprintf('1. Euler Method...\n');
    for i = 1:n-1
        dydt = reaction_system(t(i), y_euler(i, :));
        y_euler(i+1, :) = y_euler(i, :) + h * dydt';
    end
    
    %% 2. Runge-Kutta 4th Order Method
    fprintf('2. Runge-Kutta 4th Order Method...\n');
    for i = 1:n-1
        k1_rk = h * reaction_system(t(i), y_rk4(i, :))';
        k2_rk = h * reaction_system(t(i) + h/2, y_rk4(i, :) + k1_rk/2)';
        k3_rk = h * reaction_system(t(i) + h/2, y_rk4(i, :) + k2_rk/2)';
        k4_rk = h * reaction_system(t(i) + h, y_rk4(i, :) + k3_rk)';
        
        y_rk4(i+1, :) = y_rk4(i, :) + (k1_rk + 2*k2_rk + 2*k3_rk + k4_rk)/6;
    end
    
    %% 3. Gauss-Seidel Method (Implicit Euler with iterative solver)
    fprintf('3. Gauss-Seidel Method (Implicit Euler)...\n');
    max_iter = 100;
    tolerance = 1e-6;
    
    for i = 1:n-1
        % Initial guess for next step
        y_guess = y_gauss_seidel(i, :);
        
        for iter = 1:max_iter
            y_old = y_guess;
            
            % Gauss-Seidel iterations for implicit Euler
            for j = 1:5
                % Compute residual for component j
                dydt = reaction_system(t(i+1), y_guess);
                residual = y_guess(j) - y_gauss_seidel(i, j) - h * dydt(j);
                
                % Update component j (simplified Newton-like update)
                y_guess(j) = y_guess(j) - 0.5 * residual;
            end
            
            % Check convergence
            if norm(y_guess - y_old) < tolerance
                break;
            end
        end
        
        y_gauss_seidel(i+1, :) = y_guess;
    end
    
    %% 4. Gauss Elimination Method (Implicit Euler with direct solver)
    fprintf('4. Gauss Elimination Method (Implicit Euler)...\n');
    
    for i = 1:n-1
        % For demonstration, using a simplified approach
        % In practice, this would involve forming and solving the Jacobian system
        
        % Use Newton's method with Gauss elimination for solving the implicit system
        y_guess = y_gauss_elim(i, :);
        
        for newton_iter = 1:10
            % Evaluate function and Jacobian
            F = y_guess - y_gauss_elim(i, :) - h * reaction_system(t(i+1), y_guess)';
            
            % Approximate Jacobian using finite differences
            J = eye(5);
            epsilon = 1e-8;
            for j = 1:5
                y_perturb = y_guess;
                y_perturb(j) = y_perturb(j) + epsilon;
                F_perturb = y_perturb - y_gauss_elim(i, :) - h * reaction_system(t(i+1), y_perturb)';
                J(:, j) = J(:, j) - h * (F_perturb' - F') / epsilon;
            end
            
            % Solve linear system using Gauss elimination (MATLAB's backslash)
            delta_y = -J \ F';
            y_guess = y_guess + delta_y';
            
            % Check convergence
            if norm(delta_y) < tolerance
                break;
            end
        end
        
        y_gauss_elim(i+1, :) = y_guess;
    end
    
    %% Plot Results
    fprintf('Generating plots...\n');
    
    % Create figure with subplots
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot concentrations vs time for each method
    methods = {'Euler', 'RK4', 'Gauss-Seidel', 'Gauss Elimination'};
    solutions = {y_euler, y_rk4, y_gauss_seidel, y_gauss_elim};
    species = {'A', 'B', 'C', 'D', 'Cat'};
    colors = {'b', 'r', 'g', 'm', 'c'};
    
    for method_idx = 1:4
        subplot(2, 2, method_idx);
        hold on;
        
        for species_idx = 1:5
            plot(t, solutions{method_idx}(:, species_idx), ...
                 'Color', colors{species_idx}, 'LineWidth', 2, ...
                 'DisplayName', species{species_idx});
        end
        
        xlabel('Time');
        ylabel('Concentration');
        title(sprintf('%s Method', methods{method_idx}));
        legend('Location', 'best');
        grid on;
        hold off;
    end
    
    % Comparison plot
    figure('Position', [100, 100, 1200, 600]);
    
    subplot(1, 2, 1);
    hold on;
    plot(t, y_euler(:, 1), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Euler');
    plot(t, y_rk4(:, 1), 'r-', 'LineWidth', 2, 'DisplayName', 'RK4');
    plot(t, y_gauss_seidel(:, 1), 'g:', 'LineWidth', 1.5, 'DisplayName', 'Gauss-Seidel');
    plot(t, y_gauss_elim(:, 1), 'm-.', 'LineWidth', 1.5, 'DisplayName', 'Gauss Elimination');
    xlabel('Time');
    ylabel('Concentration of A');
    title('Comparison of Methods - Species A');
    legend('Location', 'best');
    grid on;
    
    subplot(1, 2, 2);
    hold on;
    plot(t, y_euler(:, 2), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Euler');
    plot(t, y_rk4(:, 2), 'r-', 'LineWidth', 2, 'DisplayName', 'RK4');
    plot(t, y_gauss_seidel(:, 2), 'g:', 'LineWidth', 1.5, 'DisplayName', 'Gauss-Seidel');
    plot(t, y_gauss_elim(:, 2), 'm-.', 'LineWidth', 1.5, 'DisplayName', 'Gauss Elimination');
    xlabel('Time');
    ylabel('Concentration of B');
    title('Comparison of Methods - Species B');
    legend('Location', 'best');
    grid on;
    
    %% Performance Analysis
    fprintf('\n=== Performance Analysis ===\n');
    
    % Calculate mass balance error (should be conserved for A+B+C+D)
    for method_idx = 1:4
        total_mass = sum(solutions{method_idx}(:, 1:4), 2);
        mass_error = abs(total_mass - total_mass(1));
        max_mass_error = max(mass_error);
        fprintf('%s Method - Max Mass Balance Error: %.6f\n', ...
                methods{method_idx}, max_mass_error);
    end
    
    % Final concentrations
    fprintf('\n=== Final Concentrations (t = %.1f) ===\n', t_end);
    fprintf('Method\t\tA\t\tB\t\tC\t\tD\t\tCat\n');
    fprintf('------\t\t----\t\t----\t\t----\t\t----\t\t----\n');
    for method_idx = 1:4
        final_conc = solutions{method_idx}(end, :);
        fprintf('%s\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
                methods{method_idx}, final_conc(1), final_conc(2), ...
                final_conc(3), final_conc(4), final_conc(5));
    end
    
    fprintf('\nSimulation completed successfully!\n');
end

% Additional function for stability analysis
function stability_analysis()
    % Analyze stability of different methods with varying step sizes
    fprintf('\n=== Stability Analysis ===\n');
    
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001];
    
    for h = step_sizes
        fprintf('Testing step size h = %.3f\n', h);
        
        % Test with Euler method
        try
            % Run a short simulation to check stability
            t = 0:h:2;
            y = [1, 0, 0, 0, 1];  % Initial conditions
            
            for i = 1:length(t)-1
                dydt = reaction_system_simple(t(i), y);
                y_new = y + h * dydt;
                
                % Check for instability (negative concentrations or explosion)
                if any(y_new < -0.1) || any(y_new > 10)
                    fprintf('  Euler method: UNSTABLE\n');
                    break;
                end
                y = y_new;
            end
            
            if i == length(t)-1
                fprintf('  Euler method: STABLE\n');
            end
            
        catch
            fprintf('  Euler method: ERROR\n');
        end
    end
end

function dydt = reaction_system_simple(t, y)
    % Simplified reaction system for stability analysis
    k1 = 2; k_1 = 0.4; k2 = 0.6; k3 = 0.2; kd = 0.1;
    
    A = y(1); B = y(2); C = y(3); D = y(4); Cat = y(5);
    
    dydt = zeros(1, 5);
    dydt(1) = -k1 * A * Cat + k_1 * B * Cat;
    dydt(2) = k1 * A * Cat - k_1 * B * Cat - k2 * B * Cat - k3 * B * Cat;
    dydt(3) = k2 * B * Cat;
    dydt(4) = k3 * B * Cat;
    dydt(5) = -kd * Cat;
end