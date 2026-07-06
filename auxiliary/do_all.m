[d2, d2ir, spectrumri, spectrumii, dataFreq] = readpipe_2Di('../build/kras.ft2');

% do_level;
% 
% figure(1);
% contour(d2,level(3:end));

ndx=98;
% ndx=136;
% ndx=211;
signal=d2rr(ndx,:);
x = 1:length(signal);

[pos_signal, neg_signal, region_mask]=run_flatt(signal,17,5e4);
[fitted_trend, cleaned_signal] = run_flatt_svd_fitting(signal, region_mask, 1024,3);

figure(2);
clf;
plot(x,signal,'k-',x,pos_signal*1e6,'bs',x,neg_signal*1e6,'rx');


figure(3);
plot(x,signal,'b-',x,fitted_trend,'r-',x,cleaned_signal,'k-');


% Call classify_regions to identify peaks, empty regions, and water regions
[pos_peaks, neg_peaks, empty_regions, water_regions] = classify_regions(region_mask);

% Expand negative peak region by up to 4 data points on both ends,
% unless they touch water regions or positive peaks
for step = 1:16
    neg_indices = find(neg_peaks);
    
    left_indices = neg_indices - 1;
    left_indices = left_indices(left_indices >= 1);
    left_indices = left_indices(~pos_peaks(left_indices) & ~water_regions(left_indices));
    
    right_indices = neg_indices + 1;
    right_indices = right_indices(right_indices <= length(signal));
    right_indices = right_indices(~pos_peaks(right_indices) & ~water_regions(right_indices));
    
    neg_peaks(left_indices) = true;
    neg_peaks(right_indices) = true;
end

figure(4);
clf;
calculated_baseline = baseline_code(signal, [], neg_peaks, water_regions)*max(signal(:));
plot(x, signal, 'b-', x, calculated_baseline, 'r-');
legend('Normalized Signal', 'Calculated Baseline');
title('Baseline Correction (baseline\_code)');


figure(5);
clf;
plot(x,cleaned_signal);
hold on;
plot(x,signal-calculated_baseline);

figure(6);
clf;
plot(x, signal, 'k-', 'LineWidth', 1); hold on;

% Highlight positive peak regions in red
if any(pos_peaks)
    plot(x(pos_peaks), signal(pos_peaks), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 4, 'DisplayName', 'Positive Peak');
end

% Highlight negative peak regions in blue
if any(neg_peaks)
    plot(x(neg_peaks), signal(neg_peaks), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 4, 'DisplayName', 'Negative Peak');
end

% Highlight water regions in green
if any(water_regions)
    plot(x(water_regions), signal(water_regions), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 4, 'DisplayName', 'Water Region');
end

hold off;
legend('Location', 'best');
title('Region Classification (classify\_regions)');
xlabel('Index');
ylabel('Intensity');



