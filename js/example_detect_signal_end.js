/**
 * Detect the end of the FID signal.
 * 
 * Algorithm:
 * 1. Skip first 150 points.
 * 2. Use moving max and moving min (window size = min(256, N/1024)) to extract envelope.
 * 3. Rough signal end: where envelope drops below 1% of max amplitude (A0).
 * 4. Noise estimation:
 *    - Region: from rough signal end to N.
 *    - Method: Split into segments of length 32. Calculate StdDev (RMSD) for each.
 *    - Noise Level = Median of these StdDevs.
 * 5. Final cutoff: where envelope drops below max(2 * Noise, 0.001 * A0).
 * 
 * @param {Float32Array} fid_data - The FID data.
 * @returns {number} The index where the signal ends.
 */
function detect_signal_end(fid_data) {
    const N = fid_data.length;
    if (N < 200) return N;

    // 1. Parameters
    const skip = 150;
    const window_size = Math.max(1, Math.min(4096, Math.floor(N / 16)));

    // 2. Envelope Extraction
    let envelope_amp = []; // value
    let envelope_idx = []; // original index

    // Efficiency: For window_size 256, naive loop is OK.
    for (let i = skip; i <= N - window_size; i++) {
        let local_max = -Infinity;
        let local_min = Infinity;
        for (let j = 0; j < window_size; j++) {
            const val = fid_data[i + j];
            if (val > local_max) local_max = val;
            if (val < local_min) local_min = val;
        }
        let amp = (local_max - local_min) * 0.5;
        envelope_amp.push(amp);
        envelope_idx.push(i);
    }

    if (envelope_amp.length === 0) return N;

    // Estimate A0 (Max of envelope)
    // To be robust, max of first chunk? Or global max of envelope?
    // Envelope should be strictly decaying ideally, so max should be near start.
    let A0 = 0;
    for (let val of envelope_amp) {
        if (val > A0) A0 = val;
    }

    if (A0 === 0) return N;

    // 3. Rough End Definition (1%)
    const thresh_1pct = 0.01 * A0;
    let rough_end_idx_in_envelope = envelope_amp.length - 1;

    // Find first point where it drops and stays low? Or just first point?
    // "use 1% to define end of signal"
    for (let k = 0; k < envelope_amp.length; k++) {
        if (envelope_amp[k] < thresh_1pct) {
            rough_end_idx_in_envelope = k;
            break;
        }
    }

    const noise_start_idx = envelope_idx[rough_end_idx_in_envelope];

    // 4. Noise Estimation (Median of RMSDs of 32-pt segments)
    const segment_len = 32;
    let std_devs = [];

    // Iterate from noise_start_idx to N
    // We iterate on the raw data
    const noise_region_len = N - noise_start_idx;
    const n_segments = Math.floor(noise_region_len / segment_len);

    if (n_segments > 0) {
        for (let s = 0; s < n_segments; s++) {
            const seg_start = noise_start_idx + s * segment_len;

            // Calc Mean
            let sum = 0;
            for (let i = 0; i < segment_len; i++) sum += fid_data[seg_start + i];
            let mean = sum / segment_len;

            // Calc Variance
            let sum_sq_diff = 0;
            for (let i = 0; i < segment_len; i++) {
                const diff = fid_data[seg_start + i] - mean;
                sum_sq_diff += diff * diff;
            }
            // StdDev (Population or Sample? Usually Sample n-1, but for noise n is fine too. Let's use n-1)
            let variance = sum_sq_diff / (segment_len - 1);
            std_devs.push(Math.sqrt(variance));
        }

        // Median
        std_devs.sort((a, b) => a - b);
        var noise_level = std_devs[Math.floor(std_devs.length / 2)];
    } else {
        // Fallback if region is too small: just std of whatever is there
        noise_level = 0; // Or handle gracefully
    }

    // 5. Final Cutoff
    // "reach 2 times noise level or 0.1% of init"
    // "whichever is short" -> The value that is HIGHER (reached earlier).
    const limit = Math.max(6 * noise_level, 0.001 * A0);

    let final_cutoff_idx = N; // Default to end

    for (let k = 0; k < envelope_amp.length; k++) {
        if (envelope_amp[k] < limit) {
            final_cutoff_idx = envelope_idx[k];
            break;
        }
    }

    return final_cutoff_idx;
}