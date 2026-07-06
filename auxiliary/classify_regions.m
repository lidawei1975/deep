function [pos_peaks, neg_peaks, empty_regions, water_regions] = classify_regions(region_mask)
% CLASSIFY_REGIONS Classifies signal regions into peak types, water, and empty regions.
%
% Inputs:
%   region_mask   - Mask of the same length as signal containing values:
%                    1  = Positive Trend
%                   -1  = Negative Trend
%                    0  = Empty/Baseline Region
%
% Outputs:
%   pos_peaks     - Logical mask of positive peak regions
%   neg_peaks     - Logical mask of negative peak regions
%   empty_regions - Logical mask of empty regions
%   water_regions - Logical mask of water regions

    N = length(region_mask);
    pos_peaks = false(1, N);
    neg_peaks = false(1, N);
    water_regions = false(1, N);
    
    % Identify contiguous non-zero blocks (structural regions)
    non_zero = region_mask ~= 0;
    labeled = zeros(1, N);
    num_blocks = 0;
    in_block = false;
    for i = 1:N
        if non_zero(i)
            if ~in_block
                num_blocks = num_blocks + 1;
                in_block = true;
            end
            labeled(i) = num_blocks;
        else
            in_block = false;
        end
    end
    
    % Process each contiguous block
    for b = 1:num_blocks
        block_indices = find(labeled == b);
        block_values = region_mask(block_indices);
        
        % Extract segments of constant sign within this block
        diffs = diff(block_values);
        transitions = find(diffs ~= 0);
        segment_ends = [transitions, length(block_values)];
        segment_starts = [1, transitions + 1];
        num_segments = length(segment_starts);
        
        segment_signs = zeros(1, num_segments);
        segment_lengths = zeros(1, num_segments);
        for s = 1:num_segments
            idx_range = segment_starts(s):segment_ends(s);
            segment_signs(s) = block_values(idx_range(1));
            segment_lengths(s) = length(idx_range);
        end
        
        is_pos_peak = false;
        is_neg_peak = false;
        is_water = false;
        
        if num_segments >= 1
            s_first = segment_signs(1);
            s_last = segment_signs(end);
            
            if s_first == s_last
                % Case A: Starts and ends with the same sign (odd number of segments)
                if num_segments >= 3
                    % e.g., [pos, neg, pos], [neg, pos, neg]
                    % This is characteristic of water/solvent region distortions.
                    is_water = true;
                else
                    % num_segments == 1 (e.g., only pos or only neg).
                    % This is just a simple slope trend, not a peak or water region.
                    % It defaults to empty.
                end
                
            elseif s_first == 1 && s_last == -1
                % Case B: Starts with pos and ends with neg (even number of segments)
                if num_segments == 2
                    if segment_lengths(1) >= 4 && segment_lengths(2) >= 4
                        is_pos_peak = true;
                    end
                else
                    % pos, neg, pos, neg ... (4 or more segments)
                    is_pos_peak = true;
                end
                
            elseif s_first == -1 && s_last == 1
                % Case C: Starts with neg and ends with pos (even number of segments)
                if num_segments == 2
                    if segment_lengths(1) >= 4 && segment_lengths(2) >= 4
                        is_neg_peak = true;
                    end
                else
                    % neg, pos, neg, pos ... (4 or more segments)
                    is_neg_peak = true;
                end
            end
        end
        
        % Assign block indices to the corresponding mask
        if is_pos_peak
            pos_peaks(block_indices) = true;
        elseif is_neg_peak
            neg_peaks(block_indices) = true;
        elseif is_water
            water_regions(block_indices) = true;
        end
    end
    
    % All remaining points are empty regions by default
    empty_regions = ~(pos_peaks | neg_peaks | water_regions);
end
