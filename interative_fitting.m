    % Main execution function to keep workspace clean
    % experimentalData: 2D matrix of observed values
    
    clc;close all;

    [d2,d2ri,d2ir,d2ii,dataFreq]=readpipe_2Di('alh_zf3.ft2');
    
    observed = d2(281:2:340,2360:2:2460);


    noise_floor=1.84e5;



    noise_floor = noise_floor / max(observed(:));
    observed = observed/max(observed(:)); % Normalize for stability and better visualization
    
    
    % Grid size is always 1
    [rows, cols] = size(observed);
    [x, y] = meshgrid(1:cols, 1:rows);

    % ---------------------------------------------------------
    % 2. INITIALIZATION
    % ---------------------------------------------------------
    % Modified: Dynamic target error based on signal max
    target_error = max(max(observed(:)) * 0.003 * 6, noise_floor);
    current_error = inf; 
    
    % Storage for optimized parameters.
    % NEW FORMAT: [Amp, XCen, XWid, XDiff, YCen, YWid, YDiff, ... next peak ...]
    % (7 parameters per peak)
    current_params = []; 
    
    % Regularization Steps per round
    alphas = [1, 3, 5];
    lambdas = [0, 10, 100];
    
    round_counter = 0;
    
    % ---------------------------------------------------------
    % 3. ITERATIVE FITTING LOOP
    % ---------------------------------------------------------
    while current_error > target_error
        round_counter = round_counter + 1;
        fprintf('\n=== Round %d ===\n', round_counter);
        
        % Setup Figure
        fig_pos = [50 + (round_counter*30), 50, 1000, 700];
        hFig = figure('Name', sprintf('Round %d (2D Independent)', round_counter), 'Color', 'w', 'Position', fig_pos);
        
        % A. Calculate Residual
        if isempty(current_params)
            current_model = zeros(size(x));
        else
            current_model = get_total_model_2d(x, y, current_params);
        end
        residual = observed - current_model;
        
        % B. Guess New Peak Properties based on Residual Max
        [max_res_val, max_idx_linear] = max(residual(:));
        [row, col] = ind2sub(size(residual), max_idx_linear);
        
        guess_amp = max(0, max_res_val);
        guess_cen_x = x(row, col); % Extract X location
        guess_cen_y = y(row, col); % Extract Y location
        
        % Defaults
        guess_wid_x = 3.0; guess_diff_x = 0.0;
        guess_wid_y = 3.0; guess_diff_y = 0.0; % Y defaults
        
        % New Peak Block (7 params)
        new_peak_guess = [guess_amp, guess_cen_x, guess_wid_x, guess_diff_x, ...
                                     guess_cen_y, guess_wid_y, guess_diff_y];
        
        % C. Append new params to vector
        params_working = [current_params, new_peak_guess];
        num_peaks = length(params_working) / 7;
        
        % D. Define Bounds (7 params per peak)
        % Bounds must be dynamic to data scale now
        max_dim_x = max(x(:));
        max_dim_y = max(y(:));
        max_amp = max(observed(:)) * 1.5;
        if max_amp == 0, max_amp = 1; end
        
        % [Amp, XCen, XWid, XDiff, YCen, YWid, YDiff]
        lb_peak = [0,          min(x(:)),       1.0, -3,   min(y(:)),       1.0, -3];
        ub_peak = [max_amp,    max(x(:)),        10,  3,   max(y(:)),        10,  3];
        
        lb = repmat(lb_peak, 1, num_peaks);
        ub = repmat(ub_peak, 1, num_peaks);
        
        % Use default options (removed MaxFunctionEvaluations, MaxIterations, StepTolerance)
        options = optimoptions('lsqnonlin', 'Display', 'off');
        
        % E. Multi-Step Optimization Loop
        for step = 1:3
            alpha_val = alphas(step);
            lambda_val = lambdas(step);
            
            fprintf('  Step %d: Alpha=%d, Lambda=%d... ', step, alpha_val, lambda_val);
            
            obj_fun = @(p) objective_function_2d(p, x, y, observed, noise_floor, alpha_val, lambda_val);
            
            [params_working, resnorm] = lsqnonlin(obj_fun, params_working, lb, ub, options);
            
            % SPECIAL LOGIC: Capture Exit Condition Metric at Step 1
            if step == 1
                final_model_step1 = get_total_model_2d(x, y, params_working);
                res_step1 = observed - final_model_step1;
                % Root Mean Quartic Error on the 2D data
                current_error = (mean(res_step1(:).^4))^(1/4); 
                fprintf('Step 1 Error (RM4E): %.5f ', current_error);
                
                if current_error <= target_error
                    fprintf('\n  Target met at Step 1. Skipping Steps 2 & 3.\n');
                    break; 
                end
            end
            fprintf('\n');
        end
        
        % F. Update Globals
        current_params = params_working;
        
        fprintf('  Round %d Complete. Check Metric: %.5f (Target: %.5f)\n', ...
            round_counter, current_error, target_error);
        
        % G. Visualization 
        final_model = get_total_model_2d(x, y, current_params);
        final_residual = observed - final_model;
        visualize_fit_2d(hFig, x, y, observed, final_model, final_residual, current_params, round_counter, current_error);
        
        if num_peaks >= 12
            warning('Max peak limit reached.');
            break;
        end
    end
    
    fprintf('\nConverged! Final Check Error: %.5f using %d peaks.\n', current_error, length(current_params)/7);
    disp('Final Parameters [Amp, XCen, XWid, XDiff, YCen, YWid, YDiff]:');
    disp(reshape(current_params, 7, [])');


% ---------------------------------------------------------
% 4. MODEL FUNCTIONS (2D)
% ---------------------------------------------------------

function z_total = get_total_model_2d(x, y, params)
    z_total = zeros(size(x));
    num_peaks = length(params) / 7;
    
    for i = 1:num_peaks
        idx = (i-1)*7 + 1;
        p_peak = params(idx : idx+6);
        z_total = z_total + calculate_2d_peak_fitting(x, y, p_peak);
    end
end

% Main calculator for fitting (uses unified 7-param block)
function z = calculate_2d_peak_fitting(x, y, p)
    % p = [Amp, XCen, XWid, XDiff, YCen, YWid, YDiff]
    
    amp = p(1);
    
    % X Components
    xc = p(2);
    xw = abs(p(3));
    xd = abs(p(4));
    sig_x = xw/2 + xd;
    gam_x = xw/2 - xd;
    
    % Y Components
    yc = p(5);
    yw = abs(p(6));
    yd = abs(p(7));
    sig_y = yw/2 + yd;
    gam_y = yw/2 - yd;
    
    % Profiles
    val_x = voigt(x - xc, sig_x, gam_x);
    norm_x = voigt(0, sig_x, gam_x);
    
    val_y = voigt(y - yc, sig_y, gam_y);
    norm_y = voigt(0, sig_y, gam_y);
    
    % Combine
    z = amp * (val_x ./ norm_x) .* (val_y ./ norm_y);
end

% ---------------------------------------------------------
% 5. OBJECTIVE FUNCTION (2D)
% ---------------------------------------------------------

function r = objective_function_2d(params, x, y, observed, noise_floor, alpha, lambda)
    pred = get_total_model_2d(x, y, params);
    e = pred - observed; 
    
    % 1. Asymmetric Weighting
    w_asym = ones(size(e));
    w_asym(e > 0) = alpha; 
    
    % Residuals for lsqnonlin (Vectorized)
    r_data = sqrt(w_asym(:)) .* e(:);

    % 2. Concentration Regularizer
    r_conc = 0;
    if lambda > 0
        % Modified: Dynamic threshold based on signal max
        threshold = max(0.01 * max(observed(:)) * 6, noise_floor);
        q = 1.6;
        k = 100; 
        
        sq_err = e(:).^2; 
        w_mask = 1 ./ (1 + exp(-k*(abs(e(:)) - threshold))); 
        
        s_masked = sq_err .* w_mask;
        S = sum(s_masked);
        
        if S > 1e-16
            p = s_masked ./ S;
            N = numel(p);
            grid_bias = linspace(-1, 1, N)';
            p = p .* (1 + 1e-6 * grid_bias);
            p = p / (sum(p) + eps);
            
            Cq = sum(p.^q);
            conc_penalty = 1 - Cq;
            
            r_conc = sqrt(lambda * conc_penalty);
        end
    end
    
    r = [r_data; r_conc];
end

% ---------------------------------------------------------
% 6. VISUALIZATION (2D)
% ---------------------------------------------------------
function visualize_fit_2d(hFig, x, y, observed, fitted, residual, params, round_num, error_val)
    set(0, 'CurrentFigure', hFig);
    
    % --- Subplot 1: Contour Plot (Overlay) ---
    subplot(2, 2, 1);
    
    % Generate Log Levels
    max_val = max(observed(:));
    noise_floor = 0.02 * max_val; 
    levels = [];
    if max_val > noise_floor
        curr = noise_floor;
        while curr < max_val
            levels(end+1) = curr;
            curr = curr * 1.4;
        end
        if isempty(levels), levels = [noise_floor, max_val]; end
    else
        levels = linspace(min(observed(:)), max(observed(:)), 5);
    end
    
    contour(x, y, observed, levels, 'k', 'LineWidth', 1.0, 'DisplayName', 'Observed'); hold on;
    contour(x, y, fitted, levels, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Fit');
    hold off;
    title(sprintf('Round %d: Obs vs Fit', round_num));
    xlabel('X'); ylabel('Y');
    axis equal tight; grid on;
    
     % --- Subplot 2: Residual Map ---
     subplot(2, 2, 2);
     imagesc(x(1,:), y(:,1), residual);
     set(gca, 'YDir', 'normal'); % Fix flipped Y axis
     colorbar;
     title('Residual Map');
     xlabel('X'); ylabel('Y');
     axis equal tight; 
     
    % --- Subplot 3: Cross Sections (Newest Peak) ---
    num_peaks = length(params)/7;
    last_idx = (num_peaks-1)*7 + 1;
    last_p = params(last_idx : last_idx+6);
    x_cen_last = last_p(2);
    y_cen_last = last_p(5);
    
    % Find indices closest to peak centers
    [~, row_idx] = min(abs(y(:,1) - y_cen_last)); % For X-slice (constant Y)
    [~, col_idx] = min(abs(x(1,:) - x_cen_last)); % For Y-slice (constant X)
    
    subplot(2, 2, 3);
    
    % 1. X-Slice (Horizontal cut)
    plot(x(row_idx,:), observed(row_idx,:), 'k.', 'DisplayName', 'Obs X-Slice'); hold on;
    plot(x(row_idx,:), fitted(row_idx,:), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Fit X-Slice');
    
    % 2. Y-Slice (Vertical cut)
    % Note: x-axis of plot is "coordinate value" (shared by X and Y dims if square, else similar scale)
    plot(y(:,col_idx), observed(:,col_idx), 'b^', 'MarkerSize', 3, 'DisplayName', 'Obs Y-Slice');
    plot(y(:,col_idx), fitted(:,col_idx), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Fit Y-Slice');
    
    hold off;
    title(sprintf('Cross Sections @ X=%.1f, Y=%.1f', x(1,col_idx), y(row_idx,1)));
    xlabel('Coordinate'); ylabel('Intensity');
    grid on; legend('Location', 'best');
    
    % --- Subplot 4: Fit Contour with Peak Markers ---
    subplot(2, 2, 4);
    contour(x, y, fitted, levels, 'r-', 'LineWidth', 1.2); hold on;
    
    % Loop to plot markers
    for i = 1:num_peaks
        idx = (i-1)*7 + 1;
        px = params(idx+1); % X Center
        py = params(idx+4); % Y Center
        
        if i == num_peaks
            % LATEST PEAK -> Blue Star
            plot(px, py, 'b*', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'New Peak');
        else
            % PREVIOUS PEAKS -> Blue Diamond
            plot(px, py, 'bd', 'MarkerSize', 6, 'MarkerFaceColor', 'b', 'DisplayName', 'Prev Peaks');
        end
    end
    hold off;
    title(sprintf('Fit Model + Peak Locs (%d Total)', num_peaks));
    xlabel('X'); ylabel('Y');
    axis equal tight; grid on;
    
    drawnow;
end