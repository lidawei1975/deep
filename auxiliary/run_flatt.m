function [pos_signal, neg_signal, region_mask] = run_flatt(signal, window_size, var_threshold)
% RUN_FLATT_SLOPE_VARIANCE Labels signal using linear regression parameters.
%
% Inputs:
%   signal        - 1D vector containing the input data[
%   window_size   - Size of the overlapping window (e.g., 16)
%   var_threshold - Variance threshold. Windows with residual variance below
%                   this are labeled as empty regions (0).
%
% Outputs:
%   pos_signal    - Logical mask: 1 where signal is in a positive-trending window
%   neg_signal    - Logical mask: 1 where signal is in a negative-trending window
%   region_mask   - Mask mapping the entire signal:
%                    1  = Positive Peak / Trend Region
%                   -1  = Negative Peak / Trend Region
%                    0  = Empty Region (Low fluctuation / Linear baseline)

    signal = signal(:)'; % Force row vector
    signal_max = max(signal(:));
    N = length(signal);
    
    % Force window_size to be an odd number
    if mod(window_size, 2) == 0
        window_size = window_size + 1;
    end
    
    % Initialize region mask
    region_mask = zeros(1, N);
    
    % Define sliding window step size to 1
    step = 1;
    
    % Center offset for the segment
    half_win = (window_size - 1) / 2;
    
    % Independent variable (time/index vector) for the linear fit
    x = (1:window_size)';
    
    % --- Loop through overlapping windows ---
    for start_idx = 1:step:(N - window_size + 1)
        end_idx = start_idx + window_size - 1;
        
        % Extract current window data
        y = signal(start_idx:end_idx)';
        
        % 1. Perform Linear Fitting (y = m*x + c)
        % Polyfit(x, y, 1) returns [slope, intercept]
        p = polyfit(x, y, 1);
        slope = p(1);
        
        % 2. Calculate Residuals and Variance after Linear Fitting
        y_fitted = polyval(p, x);
        residuals = y - y_fitted;
        res_variance = std(residuals);
        
        % Calculate center position of the current segment
        center_idx = start_idx + half_win;
        
        % Classify current segment directly at its center position
        if (res_variance >= var_threshold) || (abs(slope) > 0.1 * signal_max / window_size)
            if slope > 0
                region_mask(center_idx) = 1;
            else
                region_mask(center_idx) = -1;
            end
        else
            region_mask(center_idx) = 0;
        end
    end
    
    % Generate final logical masks from the region mask
    pos_signal = region_mask == 1;
    neg_signal = region_mask == -1;
end