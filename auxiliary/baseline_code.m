function calculated_baseline = baseline_code(signal, region_mask, neg_peaks, water_regions)
% BASELINE_CODE Calculates baseline for a given signal.
%   calculated_baseline = baseline_code(signal, region_mask, neg_peaks, water_regions)
%
%   Inputs:
%       signal              - The input signal vector
%       region_mask         - (Optional) Mask from run_flatt
%       neg_peaks           - (Optional) Pre-classified negative peaks mask
%       water_regions       - (Optional) Pre-classified water regions mask
%   Outputs:
%       calculated_baseline - The calculated baseline vector

% Check if input is a row vector to match output orientation
is_row = size(signal, 2) > size(signal, 1);

signal = signal / max(signal(:));
signal = signal';

n = length(signal);

% Determine if pre-classified masks are provided, or classify them internally
if nargin >= 4 && ~isempty(neg_peaks) && ~isempty(water_regions)
    neg_peaks = neg_peaks(:);
    water_regions = water_regions(:);
    pos_peaks = false(n, 1);
elseif nargin >= 2 && ~isempty(region_mask)
    [pos_peaks, neg_peaks, empty_regions, water_regions] = classify_regions(region_mask);
    pos_peaks = pos_peaks(:);
    neg_peaks = neg_peaks(:);
    water_regions = water_regions(:);
else
    pos_peaks = false(n, 1);
    neg_peaks = false(n, 1);
    water_regions = false(n, 1);
end

aa = 1e9;
b = 1e3;

calculated_baseline = zeros(n, 1) + 0.1;

for loop = 1:10
    d = zeros(n, n);
    
    % --- 1. Populate the regularizer (smoothness) terms ---
    % Row 1
    d(1, 1) = 2*aa;
    d(1, 2) = -4*aa;
    d(1, 3) = 2*aa;
    
    % Row 2
    d(2, 1) = -4*aa;
    d(2, 2) = 10*aa;
    d(2, 3) = -8*aa;
    d(2, 4) = 2*aa;
    
    % Rows 3 to n-2
    for i = 3:n-2
        d(i, i-2) = 2*aa;
        d(i, i-1) = -8*aa;
        d(i, i)   = 12*aa;
        d(i, i+1) = -8*aa;
        d(i, i+2) = 2*aa;
    end
    
    % Row n-1
    d(n-1, n-3) = 2*aa;
    d(n-1, n-2) = -8*aa;
    d(n-1, n-1) = 10*aa;
    d(n-1, n)   = -4*aa;
    
    % Row n
    d(n, n-2) = 2*aa;
    d(n, n-1) = -4*aa;
    d(n, n)   = 2*aa;
    
    % --- 2. Apply constraints based on region classification ---
    m = zeros(n, 1);
    for i = 1:n
        if water_regions(i)
            % Water regions: Exclude from constraints (Allow baseline to go above signal)
            m(i) = 1.0;
        elseif neg_peaks(i)
            % Negative peaks: Constraint is active when baseline is below the negative peak
            if calculated_baseline(i) < signal(i)
                d(i, i) = d(i, i) + 2*b;
                m(i) = 2*b*signal(i) + 1.0;
            else
                m(i) = 1.0;
            end
        else
            % Default (empty regions & positive peaks): Constraint is active when baseline goes above the signal
            if calculated_baseline(i) > signal(i)
                d(i, i) = d(i, i) + 2*b;
                m(i) = 2*b*signal(i) + 1.0;
            else
                m(i) = 1.0;
            end
        end
    end
    
    calculated_baseline = inv(d)*m;
end

% Reshape output back to match the original layout orientation
if is_row
    calculated_baseline = calculated_baseline';
end

end
