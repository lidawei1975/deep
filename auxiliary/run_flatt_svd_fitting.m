function [fitted_trend, cleaned_signal] = run_flatt_svd_fitting(signal, region_mask, embedding_dim, sv_keep)
% RUN_FLATT_SVD_FITTING Applies SVD-based filtering to reconstruct the 
% signal trend based on pre-defined FLATT structural vs empty regions.
%
% Inputs:
%   signal        - 1D vector containing the raw input signal.
%   region_mask   - Mask of the same length as signal: 
%                   (0 = Empty/Flat Region, 1 or -1 = Peak Regions).
%   embedding_dim - Window size for the trajectory matrix embedding (Hankel matrix).
%                   Typically chosen based on the expected wavelength/duration of features.
%   sv_keep       - Number of dominant Singular Values to retain for trend 
%                   reconstruction (e.g., 1, 2, or 3 for slow-moving baselines).
%
% Outputs:
%   fitted_trend  - Reconstructed smooth baseline/trend matrix profile.
%   cleaned_signal- Residual signal after extracting/handling the trends.

    signal = signal(:); % Force column vector for matrix transformations
    region_mask = region_mask(:);
    N = length(signal);
    
    % --- Step 1: Trajectory Matrix Embedding (Hankel Matrix) ---
    % Construct a trajectory matrix components mapping. 
    % K is the number of columns in the trajectory matrix.
    K = N - embedding_dim + 1;
    X = zeros(embedding_dim, K);
    
    for i = 1:K
        X(:, i) = signal(i : i + embedding_dim - 1);
    end
    
    % --- Step 2: Weighting/Mask Adjustment (FLATT Concept) ---
    % SVD can be heavily skewed by massive spikes. FLATT uses the mask to 
    % reduce the impact of structural regions during baseline estimation.
    % We create a weight matrix based on whether elements belong to an "Empty Region".
    W_vec = double(region_mask == 0); % 1 for empty regions, 0 for peaks
    
    % Create a corresponding Hankel weight matrix
    W = zeros(embedding_dim, K);
    for i = 1:K
        W(:, i) = W_vec(i : i + embedding_dim - 1);
    end
    
    % Apply weighted element-wise attenuation to structural regions in X
    % This ensures the SVD focuses its principal components primarily on tracking the 
    % background baseline variation found in the empty zones.
    X_weighted = X;
    X_weighted(W == 0) = X_weighted(W == 0) * 0.1; % Dampen peak impact
    
    % --- Step 3: Singular Value Decomposition ---
    [U, S, V] = svd(X_weighted, 'econ');
    
    % --- Step 4: Low-Rank Matrix Approximation ---
    % Reconstruct the trajectory matrix using only the dominant singular values
    S_reduced = zeros(size(S));
    S_reduced(1:sv_keep, 1:sv_keep) = S(1:sv_keep, 1:sv_keep);
    
    X_reconstructed = U * S_reduced * V';
    
    % --- Step 5: Diagonal Averaging (Anti-Diagonalization) ---
    % Reconstruct a 1D signal from the estimated trajectory matrix
    fitted_trend = zeros(N, 1);
    count = zeros(N, 1);
    
    for i = 1:embedding_dim
        for j = 1:K
            idx = i + j - 1;
            fitted_trend(idx) = fitted_trend(idx) + X_reconstructed(i, j);
            count(idx) = count(idx) + 1;
        end
    end
    fitted_trend = fitted_trend ./ count;
    
    % --- Step 6: Output Computations ---
    cleaned_signal = signal - fitted_trend;
    
    % Reshape outputs back to match the original layout orientation
    if size(signal, 2) > size(signal, 1)
        fitted_trend = fitted_trend';
        cleaned_signal = cleaned_signal';
    end
end