/**
 * TODO / NOTE ON SPECTRAL EXTRACTION HEADER BUGS:
 * 
 * 1. FDREALSIZE Mismatch when Extraction is Enabled:
 *    In create_nmrpipe_dictionary(), FDREALSIZE is calculated as:
 *        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_frq_indirect; (or ndata_frq * ndata_indirect)
 *    However, when extraction is enabled, the actual output dimensions are reduced to n_inner_dim
 *    and n_outer_dim. Since write_nmrpipe_ft2() only writes active extracted elements, this leads to 
 *    a file size smaller than what FDREALSIZE declares, causing downstream tool crashes/validation errors.
 *    Fix: Update FDREALSIZE to use n_inner_dim * n_outer_dim (or n_inner_dim * ndata_indirect).
 * 
 * 2. FDF2APOD Overwritten to 0.0 on Processed Files:
 *    FDF2APOD is computed using:
 *        int extraction_span = extraction_to_pos - extraction_from_pos;
 *        nmrpipe_dict_float["FDF2APOD"] = extraction_span / zf;
 *    When reading an already processed NMRPipe file, extraction_from_pos/extraction_to_pos are both 0.
 *    This causes extraction_span = 0, which overwrites FDF2APOD to 0.0 when writing back.
 *    Fix: Default/preserve FDF2APOD to ndata_original or its original value if no new extraction was run.  
 */

#include <vector>
#include <valarray>
#include <set>
#include <array>
#include <fstream>
#include <iostream>
#include <cstring>

/**
 * Below two lines are required to check whether folder or file exists
 */
#include <sys/types.h>
#include <sys/stat.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

/**
 * For FFT calculation
 */
#include "kiss_fft.h"

#include "json/json.h"

#include "lmminimizer.h"
#include "fid_2d.h"

#ifdef WEBASSEMBLY
/**
* Bind to emscripten with exposed class methods.
*/
#include <emscripten/bind.h>
using namespace emscripten;
#endif

namespace fid_2d_math {
    bool movemean(float *data, float * new_data, int ndata, int nwindow )
    {
        if (nwindow <= 0)
        {
            return false;
        }
        if (nwindow > ndata)
        {
            return false;
        }
        for (int i = 0; i < ndata; i++)
        {
            int n1 = i - nwindow / 2;
            int n2 = i + nwindow / 2;
            if (n1 < 0)
            {
                n1 = 0;
            }
            if (n2 >= ndata)
            {
                n2 = ndata - 1;
            }
            float sum = 0.0;
            for (int j = n1; j <= n2; j++)
            {
                sum += data[j];
            }
            new_data[i] = sum / (n2 - n1 + 1);
        }

        return true;
    };

    bool spectrum_minus(float *data1, float *data2, int  n)
    {
        for (int i = 0; i < n; i++)
        {
            data1[i] -= data2[i];
        }
        return true;
    };

    bool detrend_slice_1d(const float *src, float *dst, int len)
    {
        int seg_len = 32;
        for (int start = 0; start < len; start += seg_len)
        {
            int end = std::min(len, start + seg_len);
            int n = end - start;
            if (n >= 3)
            {
                Eigen::MatrixXd design(n, 3);
                Eigen::VectorXd vals(n);
                for (int i = 0; i < n; ++i)
                {
                    double x_val = (n > 1) ? (2.0 * i / (n - 1) - 1.0) : 0.0;
                    design(i, 0) = 1.0;
                    design(i, 1) = x_val;
                    design(i, 2) = x_val * x_val;
                    vals(i) = static_cast<double>(src[start + i]);
                }
                Eigen::VectorXd coeffs = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vals);
                for (int i = 0; i < n; ++i)
                {
                    double x_val = (n > 1) ? (2.0 * i / (n - 1) - 1.0) : 0.0;
                    double fitted = coeffs(0) + coeffs(1) * x_val + coeffs(2) * x_val * x_val;
                    dst[start + i] = static_cast<float>(src[start + i] - fitted);
                }
            }
            else
            {
                // Fall back to subtracting mean
                double sum = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    sum += src[start + i];
                }
                double mean = sum / n;
                for (int i = 0; i < n; ++i)
                {
                    dst[start + i] = static_cast<float>(src[start + i] - mean);
                }
            }
        }
        return true;
    }

    bool polynomial_baseline(
        const std::vector<float> &x,
        const std::vector<float> &y,
        const std::vector<float> &z,
        int xdim,
        int ydim,
        int order,
        std::vector<float> &baseline_parameters,
        float noise_level = -1.0f,
        std::vector<char> *selected_mask = nullptr
        )
    {
        /**
         * Using SVD to solve the linear equation
         * d * p = b, where
         * p is the baseline parameters p[0] + p[1]*x + p[2]*y + p[3]*x^2 + p[4]*y*y (no cross term in NMR)
         * b is the data
        */
        int n = xdim * ydim;
        int m = order*2 + 1;

        std::vector<int> valid_indices;
        if (noise_level > 0.0f)
        {
            std::vector<char> valid_point(n, 0);
            double threshold = 4.0 * noise_level;

            // Segment size for block processing
            int block_size_row = 16;
            int block_size_col = 16;
            if (xdim < 16) block_size_row = xdim;
            if (ydim < 16) block_size_col = ydim;

            for (int r_start = 0; r_start < xdim; r_start += block_size_row)
            {
                int r_end = std::min(xdim, r_start + block_size_row);
                int r_len = r_end - r_start;
                for (int c_start = 0; c_start < ydim; c_start += block_size_col)
                {
                    int c_end = std::min(ydim, c_start + block_size_col);
                    int c_len = c_end - c_start;

                    int count = r_len * c_len;
                    if (count <= 0) continue;

                    double sum = 0.0;
                    for (int r = r_start; r < r_end; ++r)
                    {
                        for (int c = c_start; c < c_end; ++c)
                        {
                            sum += static_cast<double>(z[r * ydim + c]);
                        }
                    }
                    double mean = sum / count;

                    double sq_sum = 0.0;
                    for (int r = r_start; r < r_end; ++r)
                    {
                        for (int c = c_start; c < c_end; ++c)
                        {
                            double diff = static_cast<double>(z[r * ydim + c]) - mean;
                            sq_sum += diff * diff;
                        }
                    }
                    double rmsd = std::sqrt(sq_sum / count);

                    if (rmsd <= threshold)
                    {
                        for (int r = r_start; r < r_end; ++r)
                        {
                            for (int c = c_start; c < c_end; ++c)
                            {
                                valid_point[r * ydim + c] = 1;
                            }
                        }
                    }
                }
            }

            valid_indices.reserve(n);
            for (int i = 0; i < n; ++i)
            {
                if (valid_point[i])
                {
                    valid_indices.push_back(i);
                }
            }

            // Fallback 1: if too few points, select individual points under threshold
            if (valid_indices.size() < static_cast<size_t>(m))
            {
                valid_indices.clear();
                for (int i = 0; i < n; ++i)
                {
                    if (std::abs(z[i]) <= threshold)
                    {
                        valid_indices.push_back(i);
                    }
                }
            }

            // Fallback 2: if still too few, use all points
            if (valid_indices.size() < static_cast<size_t>(m))
            {
                valid_indices.clear();
                for (int i = 0; i < n; ++i)
                {
                    valid_indices.push_back(i);
                }
            }
        }
        else
        {
            valid_indices.reserve(n);
            for (int i = 0; i < n; ++i)
            {
                valid_indices.push_back(i);
            }
        }

        if (selected_mask != nullptr)
        {
            selected_mask->assign(n, 0);
            for (int i : valid_indices)
            {
                (*selected_mask)[i] = 1;
            }
        }

        int n_valid = static_cast<int>(valid_indices.size());
        Eigen::MatrixXd d(n_valid, m);
        Eigen::VectorXd b(n_valid);
        Eigen::VectorXd solution(m);

        /**
         * Copy selected z to b
         */
        for (int row = 0; row < n_valid; ++row)
        {
            int i = valid_indices[row];
            b(row) = z[i];
        }

        /**
         * Fill d
         */
        for (int row = 0; row < n_valid; ++row)
        {
            int i = valid_indices[row];
            d(row,0) = 1.0;
            if(order>=1)
            {
                d(row,1) = x[i];
                d(row,2) = y[i];
            }
            if(order>=2)
            {
                d(row,3) = x[i]*x[i];
                d(row,4) = y[i]*y[i];
            }
            if(order>=3)
            {
                d(row,5) = x[i]*x[i]*x[i];
                d(row,6) = y[i]*y[i]*y[i];
            }
            if(order>=4)
            {
                d(row,7) = x[i]*x[i]*x[i]*x[i];
                d(row,8) = y[i]*y[i]*y[i]*y[i];
            }
        }

        /**
         * Solve the linear equation
         */
        solution = d.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        /**
         * Copy solution to baseline_parameters
         */
        baseline_parameters.clear();
        for(int i=0;i<m;i++)
        {
            baseline_parameters.push_back(solution(i));
        }

        return true;
    }

    bool polynomial_baseline_1d(
        const float *data,
        int ndata,
        int order,
        std::vector<float> &baseline,
        double threshold,
        int segment_size,
        std::vector<char> *selected_mask = nullptr
        )
    {
        if (order < 0 || ndata <= 0 || segment_size <= 0)
        {
            return false;
        }

        std::vector<char> valid_point(ndata, 0);

        for (int start = 0; start < ndata; start += segment_size)
        {
            const int end = std::min(ndata, start + segment_size);
            const int seg_len = end - start;
            if (seg_len <= 0)
            {
                continue;
            }

            double sum = 0.0;
            for (int r = 0; r < seg_len; ++r)
            {
                sum += static_cast<double>(data[start + r]);
            }
            const double mean = sum / static_cast<double>(seg_len);

            double sq_sum = 0.0;
            for (int r = 0; r < seg_len; ++r)
            {
                const double diff = static_cast<double>(data[start + r]) - mean;
                sq_sum += diff * diff;
            }
            const double rmsd = std::sqrt(sq_sum / static_cast<double>(seg_len));

            if (rmsd <= threshold)
            {
                for (int r = 0; r < seg_len; ++r)
                {
                    valid_point[start + r] = 1;
                }
            }
        }

        std::vector<int> valid_indices;
        valid_indices.reserve(ndata);
        for (int i = 0; i < ndata; ++i)
        {
            if (valid_point[i])
            {
                valid_indices.push_back(i);
            }
        }

        const int n_terms = order + 1;
        if (valid_indices.size() < static_cast<size_t>(n_terms))
        {
            valid_indices.clear();
            for (int i = 0; i < ndata; ++i)
            {
                if (std::abs(data[i]) <= threshold)
                {
                    valid_indices.push_back(i);
                }
            }
        }

        if (valid_indices.size() < static_cast<size_t>(n_terms))
        {
            valid_indices.clear();
            for (int i = 0; i < ndata; ++i)
            {
                valid_indices.push_back(i);
            }
        }

        if (selected_mask != nullptr)
        {
            selected_mask->assign(static_cast<size_t>(ndata), 0);
            for (int i : valid_indices)
            {
                (*selected_mask)[static_cast<size_t>(i)] = 1;
            }
        }

        const int n_valid = static_cast<int>(valid_indices.size());
        Eigen::MatrixXd design(n_valid, n_terms);
        Eigen::VectorXd values(n_valid);

        for (int row = 0; row < n_valid; ++row)
        {
            const int i = valid_indices[row];
            const double x = (ndata > 1) ? (2.0 * static_cast<double>(i) / static_cast<double>(ndata - 1) - 1.0) : 0.0;
            double term = 1.0;
            values(row) = static_cast<double>(data[i]);
            for (int j = 0; j < n_terms; ++j)
            {
                design(row, j) = term;
                term *= x;
            }
        }

        Eigen::VectorXd solution = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);

        baseline.assign(ndata, 0.0f);
        for (int i = 0; i < ndata; ++i)
        {
            const double x = (ndata > 1) ? (2.0 * static_cast<double>(i) / static_cast<double>(ndata - 1) - 1.0) : 0.0;
            double term = 1.0;
            double fitted = 0.0;
            for (int j = 0; j < n_terms; ++j)
            {
                fitted += solution(j) * term;
                term *= x;
            }
            baseline[i] = static_cast<float>(fitted);
        }

        return true;
    }

    bool solve_pentadiagonal(
        int n,
        const std::vector<double>& diag,
        const std::vector<double>& sub1,
        const std::vector<double>& sub2,
        const Eigen::VectorXd& m,
        Eigen::VectorXd& x
    ) {
        if (n <= 0) return false;

        std::vector<double> delta(n, 0.0);
        std::vector<double> c(n - 1, 0.0);
        std::vector<double> g(n - 2, 0.0);

        // i = 0
        delta[0] = diag[0];
        if (std::abs(delta[0]) < 1e-15) return false;
        c[0] = sub1[0] / delta[0];
        g[0] = sub2[0] / delta[0];

        // i = 1
        if (n >= 2) {
            delta[1] = diag[1] - c[0] * c[0] * delta[0];
            if (std::abs(delta[1]) < 1e-15) return false;
            c[1] = (sub1[1] - g[0] * c[0] * delta[0]) / delta[1];
            if (n > 2) {
                g[1] = sub2[1] / delta[1];
            }
        }

        // i = 2 to n - 3
        for (int i = 2; i < n - 2; i++) {
            delta[i] = diag[i] - c[i-1] * c[i-1] * delta[i-1] - g[i-2] * g[i-2] * delta[i-2];
            if (std::abs(delta[i]) < 1e-15) return false;
            c[i] = (sub1[i] - g[i-1] * c[i-1] * delta[i-1]) / delta[i];
            g[i] = sub2[i] / delta[i];
        }

        // i = n - 2
        if (n >= 3) {
            int i = n - 2;
            delta[i] = diag[i] - c[i-1] * c[i-1] * delta[i-1] - g[i-2] * g[i-2] * delta[i-2];
            if (std::abs(delta[i]) < 1e-15) return false;
            c[i] = (sub1[i] - g[i-1] * c[i-1] * delta[i-1]) / delta[i];
        }

        // i = n - 1
        if (n >= 2) {
            int i = n - 1;
            delta[i] = diag[i] - c[i-1] * c[i-1] * delta[i-1] - g[i-2] * g[i-2] * delta[i-2];
            if (std::abs(delta[i]) < 1e-15) return false;
        }

        // Solve L * y = m (forward substitution)
        std::vector<double> y(n, 0.0);
        y[0] = m[0];
        if (n >= 2) {
            y[1] = m[1] - c[0] * y[0];
        }
        for (int i = 2; i < n; i++) {
            y[i] = m[i] - c[i-1] * y[i-1] - g[i-2] * y[i-2];
        }

        // Solve D * z = y
        std::vector<double> z(n, 0.0);
        for (int i = 0; i < n; i++) {
            z[i] = y[i] / delta[i];
        }

        // Solve L^T * x = z (backward substitution)
        x.resize(n);
        x[n-1] = z[n-1];
        if (n >= 2) {
            x[n-2] = z[n-2] - c[n-2] * x[n-1];
        }
        for (int i = n - 3; i >= 0; i--) {
            x[i] = z[i] - c[i] * x[i+1] - g[i] * x[i+2];
        }

        return true;
    }

    bool flatt_identify_regions_1d(
        const float *data,
        int ndata,
        std::vector<bool> &pos_peaks,
        std::vector<bool> &neg_peaks,
        std::vector<bool> &water_regions,
        double noise_level,
        int window_size = 17,
        double var_threshold = -1.0,
        std::vector<int> *out_region_mask = nullptr
    ) {
        if (ndata <= 0) return false;

        // 1. Run FLATT to get region_mask
        double threshold = var_threshold;
        if (threshold < 0) {
            threshold = 4.0 * noise_level;
        }

        int W = window_size;
        if (W % 2 == 0) W += 1;
        double x_bar = (W + 1) / 2.0;
        double sum_x_sq = W * (W * W - 1) / 12.0;
        int half_win = (W - 1) / 2;

        std::vector<int> region_mask(ndata, 0);

        for (int start_idx = 0; start_idx <= ndata - W; start_idx++) {
            double sum_y = 0.0;
            for (int i = 0; i < W; i++) {
                sum_y += data[start_idx + i];
            }
            double y_bar = sum_y / W;

            double num = 0.0;
            for (int i = 0; i < W; i++) {
                num += (i + 1 - x_bar) * data[start_idx + i];
            }
            double slope = num / sum_x_sq;
            double intercept = y_bar - slope * x_bar;

            double rss = 0.0;
            for (int i = 0; i < W; i++) {
                double fitted = slope * (i + 1) + intercept;
                double res = data[start_idx + i] - fitted;
                rss += res * res;
            }
            double res_variance = std::sqrt(rss / (W - 1));

            double slope_threshold = 3.0 * noise_level;
            int center_idx = start_idx + half_win;
            if (res_variance >= threshold || std::abs(slope) > slope_threshold) {
                if (slope > 0) {
                    region_mask[center_idx] = 1;
                } else {
                    region_mask[center_idx] = -1;
                }
            } else {
                region_mask[center_idx] = 0;
            }
        }

        // 2. Classify regions
        pos_peaks.assign(ndata, false);
        neg_peaks.assign(ndata, false);
        water_regions.assign(ndata, false);

        // Identify contiguous non-zero blocks (connected components)
        std::vector<int> labeled(ndata, 0);
        int num_blocks = 0;
        bool in_block = false;
        for (int i = 0; i < ndata; i++) {
            if (region_mask[i] != 0) {
                if (!in_block) {
                    num_blocks++;
                    in_block = true;
                }
                labeled[i] = num_blocks;
            } else {
                in_block = false;
            }
        }

        // Process each contiguous block
        for (int b = 1; b <= num_blocks; b++) {
            std::vector<int> block_indices;
            for (int i = 0; i < ndata; i++) {
                if (labeled[i] == b) {
                    block_indices.push_back(i);
                }
            }
            if (block_indices.empty()) continue;

            // Extract segments of constant sign within this block
            std::vector<int> block_values;
            for (int idx : block_indices) {
                block_values.push_back(region_mask[idx]);
            }

            std::vector<int> transitions;
            for (size_t i = 0; i < block_values.size() - 1; i++) {
                if (block_values[i] != block_values[i+1]) {
                    transitions.push_back(i);
                }
            }

            std::vector<int> segment_starts;
            std::vector<int> segment_ends;
            segment_starts.push_back(0);
            for (int t : transitions) {
                segment_ends.push_back(t);
                segment_starts.push_back(t + 1);
            }
            segment_ends.push_back(block_values.size() - 1);

            int num_segments = segment_starts.size();
            std::vector<int> segment_signs(num_segments);
            std::vector<int> segment_lengths(num_segments);
            for (int s = 0; s < num_segments; s++) {
                segment_signs[s] = block_values[segment_starts[s]];
                segment_lengths[s] = segment_ends[s] - segment_starts[s] + 1;
            }

            bool is_pos_peak = false;
            bool is_neg_peak = false;
            bool is_water = false;

            if (num_segments >= 3) {
                is_water = true;
            } else if (num_segments >= 1) {
                int s_first = segment_signs.front();
                int s_last = segment_signs.back();

                if (s_first == 1 && s_last == -1) {
                    if (num_segments == 2) {
                        if (segment_lengths[0] >= 2 && segment_lengths[1] >= 2) {
                            is_pos_peak = true;
                        }
                    } else {
                        is_pos_peak = true;
                    }
                } else if (s_first == -1 && s_last == 1) {
                    if (num_segments == 2) {
                        if (segment_lengths[0] >= 2 && segment_lengths[1] >= 2) {
                            is_neg_peak = true;
                        }
                    } else {
                        is_neg_peak = true;
                    }
                }
            }

            if (is_pos_peak) {
                for (int idx : block_indices) pos_peaks[idx] = true;
            } else if (is_neg_peak) {
                for (int idx : block_indices) neg_peaks[idx] = true;
            } else if (is_water) {
                for (int idx : block_indices) water_regions[idx] = true;
            }
        }

        if (out_region_mask != nullptr) {
            *out_region_mask = region_mask;
        }

        return true;
    }

    void expand_neg_peaks_1d(
        const std::vector<bool> &pos_peaks,
        std::vector<bool> &neg_peaks,
        const std::vector<bool> &water_regions,
        int ndata,
        int expand_steps = 16
    ) {
        for (int step = 1; step <= expand_steps; step++) {
            std::vector<int> neg_indices;
            for (int i = 0; i < ndata; i++) {
                if (neg_peaks[i]) {
                    neg_indices.push_back(i);
                }
            }

            std::vector<int> to_add_left;
            std::vector<int> to_add_right;

            for (int idx : neg_indices) {
                int left = idx - 1;
                if (left >= 0 && !pos_peaks[left] && !water_regions[left]) {
                    to_add_left.push_back(left);
                }
                int right = idx + 1;
                if (right < ndata && !pos_peaks[right] && !water_regions[right]) {
                    to_add_right.push_back(right);
                }
            }

            for (int idx : to_add_left) neg_peaks[idx] = true;
            for (int idx : to_add_right) neg_peaks[idx] = true;
        }
    }

    bool flatt_solver_1d(
        const float *data,
        int ndata,
        std::vector<float> &baseline,
        const std::vector<bool> &pos_peaks,
        const std::vector<bool> &neg_peaks,
        const std::vector<bool> &water_regions,
        double noise_level,
        double flatt_a = 1e9,
        double flatt_b = 1.5
    ) {
        double max_val = 0.0;
        for (int i = 0; i < ndata; i++) {
            if (std::abs(data[i]) > max_val) {
                max_val = std::abs(data[i]);
            }
        }
        if (max_val <= 0.0) max_val = 1.0;

        std::vector<double> signal(ndata);
        for (int i = 0; i < ndata; i++) {
            signal[i] = data[i] / max_val;
        }

        double noise = noise_level / max_val;
        if (noise <= 0.0) noise = 1e-4;
        double aa = flatt_a / noise;
        double b = flatt_b / noise;

        // Initialize static subdiagonals
        std::vector<double> sub1(ndata - 1);
        sub1[0] = -4.0 * aa;
        if (ndata > 2) {
            sub1[ndata - 2] = -4.0 * aa;
            for (int i = 1; i < ndata - 2; i++) {
                sub1[i] = -8.0 * aa;
            }
        }

        std::vector<double> sub2(ndata - 2, 2.0 * aa);

        Eigen::VectorXd calculated_baseline = Eigen::VectorXd::Constant(ndata, 0.1);

        for (int loop = 0; loop < 10; loop++) {
            // Build diagonal
            std::vector<double> diag(ndata);
            diag[0] = 2.0 * aa;
            if (ndata >= 2) {
                diag[ndata - 1] = 2.0 * aa;
            }
            if (ndata >= 3) {
                diag[1] = 10.0 * aa;
                diag[ndata - 2] = 10.0 * aa;
            }
            for (int i = 2; i < ndata - 2; i++) {
                diag[i] = 12.0 * aa;
            }

            // Apply constraints
            Eigen::VectorXd m = Eigen::VectorXd::Zero(ndata);
            for (int i = 0; i < ndata; i++) {
                if (water_regions[i]) {
                    m(i) = 1.0;
                } else if (neg_peaks[i]) {
                    if (calculated_baseline(i) < signal[i]) {
                        diag[i] += 2.0 * b;
                        m(i) = 2.0 * b * signal[i] + 1.0;
                    } else {
                        m(i) = 1.0;
                    }
                } else {
                    if (calculated_baseline(i) > signal[i]) {
                        diag[i] += 2.0 * b;
                        m(i) = 2.0 * b * signal[i] + 1.0;
                    } else {
                        m(i) = 1.0;
                    }
                }
            }

            // Solve using custom pentadiagonal solver
            Eigen::VectorXd next_baseline;
            if (solve_pentadiagonal(ndata, diag, sub1, sub2, m, next_baseline)) {
                calculated_baseline = next_baseline;
            } else {
                break;
            }
        }

        // Restore scale
        baseline.resize(ndata);
        for (int i = 0; i < ndata; i++) {
            baseline[i] = static_cast<float>(calculated_baseline(i) * max_val);
        }

        return true;
    }

    std::vector<bool> erode_3x3_cross(const std::vector<bool> &mask, int rows, int cols) {
        std::vector<bool> result(rows * cols, false);
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                if (mask[y * cols + x]) {
                    bool keep = true;
                    // top
                    if (y > 0 && !mask[(y - 1) * cols + x]) keep = false;
                    // bottom
                    if (y < rows - 1 && !mask[(y + 1) * cols + x]) keep = false;
                    // left
                    if (x > 0 && !mask[y * cols + (x - 1)]) keep = false;
                    // right
                    if (x < cols - 1 && !mask[y * cols + (x + 1)]) keep = false;
                    
                    if (keep) result[y * cols + x] = true;
                }
            }
        }
        return result;
    }

    std::vector<bool> dilate_3x3_cross(const std::vector<bool> &mask, int rows, int cols) {
        std::vector<bool> result = mask;
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                if (mask[y * cols + x]) {
                    if (y > 0) result[(y - 1) * cols + x] = true;
                    if (y < rows - 1) result[(y + 1) * cols + x] = true;
                    if (x > 0) result[y * cols + (x - 1)] = true;
                    if (x < cols - 1) result[y * cols + (x + 1)] = true;
                }
            }
        }
        return result;
    }

    std::vector<bool> opening_3x3_cross(const std::vector<bool> &mask, int rows, int cols) {
        return dilate_3x3_cross(erode_3x3_cross(mask, rows, cols), rows, cols);
    }

    std::vector<bool> closing_3x3_cross(const std::vector<bool> &mask, int rows, int cols) {
        return erode_3x3_cross(dilate_3x3_cross(mask, rows, cols), rows, cols);
    }

    bool flatt_baseline_1d(
        const float *data,
        int ndata,
        std::vector<float> &baseline,
        double noise_level,
        int window_size = 17,
        double var_threshold = -1.0,
        double flatt_a = 1e9,
        double flatt_b = 1.5
    ) {
        std::vector<bool> pos_peaks(ndata, false);
        std::vector<bool> neg_peaks(ndata, false);
        std::vector<bool> water_regions(ndata, false);
        if (!flatt_identify_regions_1d(data, ndata, pos_peaks, neg_peaks, water_regions, noise_level, window_size, var_threshold)) {
            return false;
        }
        expand_neg_peaks_1d(pos_peaks, neg_peaks, water_regions, ndata, 16);
        return flatt_solver_1d(data, ndata, baseline, pos_peaks, neg_peaks, water_regions, noise_level, flatt_a, flatt_b);
    }
};

namespace ldw_math_spectrum_2d
{

    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext)
    {
        bool b = false;
        std::string file_name_full;
        std::size_t found = str.find_last_of("/\\");
        if (found != std::string::npos)
        {
            path_name = str.substr(0, found);
            file_name_full = str.substr(found + 1);
            b = true;
        }
        else
        {
            path_name = ".";
            file_name_full = str;
        }

        // std::cout<<"file_name_full is "<<file_name_full<<std::endl;

        found = file_name_full.find_last_of(".");
        if (found != std::string::npos)
        {
            file_name = file_name_full.substr(0, found);
            file_name_ext = file_name_full.substr(found + 1);
            b = true;
        }
        else
        {
            file_name_ext = "";
            file_name = file_name_full;
        }

        // std::cout<<"path is "<<path_name<<std::endl;
        // std::cout<<"file_name is "<<file_name<<std::endl;
        // std::cout<<"file_name_ext is "<<file_name_ext<<std::endl;

        return b;
    }

    void get_ppm_from_header(const double ref, const double sw, const double frq, double &stop, double &begin)
    {
        if (fabs(frq) > 1e-10)
        {
            stop = ref / frq;
            begin = stop + sw / frq;
        }
        else
        {
            stop = 2.0;
            begin = 1.0; // arbitary
        }
        return;
    }

    double calcualte_median(std::vector<double> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };

    template <typename T>
    void sortArr(std::vector<T> &arr, std::vector<int> &ndx)
    {
        std::vector<std::pair<T, int>> vp;

        for (int i = 0; i < arr.size(); ++i)
        {
            vp.push_back(std::make_pair(arr[i], i));
        }

        std::sort(vp.begin(), vp.end());

        for (int i = 0; i < vp.size(); i++)
        {
            ndx.push_back(vp[i].second);
        }
    };

    template void sortArr<int>(std::vector<int> &arr, std::vector<int> &ndx);
    template void sortArr<float>(std::vector<float> &arr, std::vector<int> &ndx);
    template void sortArr<double>(std::vector<double> &arr, std::vector<int> &ndx);
    


    int next_power_of_2(int n)
    {
        double logbase2 = log(double(n)) / log(2);

        int count = int(ceil(logbase2));
  
        return int(pow(2, count));
    };

};


/**
 * IMPORTANT:
 * We have udict_acqus_direct, and udict_acqus_direct, which are dictionary to store information from Bruker acqus(2,3,4) files.
 * We also have nmrpipe_dict_string and nmrpipe_dict_float, which are used to store information for nmrPipe header. 
 * They are equivalent to nmrpipe_header_data, but in a more convenient format.
 * 
 * Then we have our own variables, such as spectral_width, observed_frequency, carrier_frequency, which are used for our own processing and will be written to nmrPipe header in the end.
 * They are read from udict_acqus_direct and udict_acqus_indirect, or from nmrPipe header when read in from nmrPipe file. They will be used in our processing and also written to nmrPipe header in the end.
 * 
 * 
 * In nmrPipe header, direct dimension information is always in
 * FDF2SW, FDF2OBS, FDF2CAR, FDF2FTSIZE, FDF2APOD, FDF2ZF, FDF2QUADFLAG
 * and first indirect dimension information is in 
 * FDF1SW, FDF1OBS, FDF1CAR, FDF1FTSIZE, FDF1APOD, FDF1ZF, FDF1QUADFLAG 
 * and second indirect dimension information is in
 * FDF3SW, FDF3OBS, FDF3CAR, FDF3FTSIZE, FDF3APOD, FDF3ZF, FDF3QUADFLAG
 * 
 * no matter whether the data is transposed or not. When transposed, we will swap the value of direct and indirect dimension in our processing,
 * but still write them to the same place in nmrPipe header.
 * 
 * However, FDSIZE is always the inner dimension size, and FDSPECNUM is the outer dimension size, 
 * no matter whether the data is transposed (1,2,3,4) or not (2,1,3,4)
 * or when we run -ext command along direct dimension (in that case, FDSIZE < FDF2FTSIZE for non-transposed data, FDSPECNUM < FDF2FTSIZE for transposed data)
 * Some 2D experiments have the order of (2,3,1,4), in that case, we will use FDF3??? for indirect dimension and FDF2??? for direct dimension. 
 * 
 * FDF?FTSIZE is always 0 for time domain data, 
 * FDF?ZF is negative of total number of points in time domain (after zero filling) for that dimension,
 * FDF?APOD is the number of points in frequency domain for that dimension, which is usually same as FDF?TDSIZE,
 */






/**
 * Constructor and destructor
 */
fid_2d::fid_2d()
{
    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = false;
    grpdly = -1.0; //<0 means not set

    /**
     * These two are for phase correction along indirect dimension, read from pulse program
    */
    indirect_p0 = -400.0; // <-360 means (indirect_p0,indirect_p1) are both N.A
    indirect_p1 = 0.0;

    /**
     * These are user provided phase correction values
    */
    user_p0_direct = 0.0;
    user_p1_direct = 0.0;
    user_p0_indirect = 0.0;
    user_p1_indirect = 0.0;

    aqseq ="321"; //default

    b_negative = false; //default. data is normal

    b_first_only = false; //default. process all spectra

    b_nmrPipe_transposed = false; //default. data is not transposed

    b_frq = false; //default. keep time domain data before FFT

    b_frq_indirect = false; //default. keep time domain data before FFT

    b_imaginary = true; //default. keep imaginary data
    
    b_imaginary_indirect = true; //default. keep imaginary data

    nuslists.clear(); //clear nuslists, not really needed ??
    nusflags.clear(); //clear nusflags, not really needed ??
    nuslists2.clear(); //clear nuslists2, not really needed ??


    apodization_code = 0;
    apodization_code_indirect = 0;

    extraction_use_ppm = false;
    extraction_from = 0.0;
    extraction_to = 1.0;
    extraction_from_ppm = 0.0;
    extraction_to_ppm = 0.0;
    extraction_from_pos = 0;
    extraction_to_pos = 0;

    noise_level = -0.1; // negative nosie means we don't have it yet

    // spectrum range
    begin1 = 100;
    stop1 = -100;
    begin2 = 1000;
    stop2 = -1000; // will be updated once read in spe

    indirect_p0 = -400.0; // <-360 means N.A.
    indirect_p1 = 0.0; //N.A when indirect_p0 is N.A.
}

fid_2d::~fid_2d()
{
    // dtor
}

/**
 * Read nus list from a file. One line per sampling point or one element per sampling point in one line
*/
bool fid_2d::read_nus_list(std::string fname)
{
    std::ifstream infile(fname);
    /**
     * Real all lines from the file
     */
    std::string contents = std::string((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();
    
    if (contents.empty())
    {
        std::cerr << "Error: nus list file is empty or does not exist." << std::endl;
        return false;
    }
    return read_nus_list_from_string(contents);
}

/**
 * Read nus list from a string
 * The string should be a space separated list of integers or new line separated list of integers
 */
bool fid_2d::read_nus_list_from_string(const std::string &nus_string)
{
    std::istringstream iss(nus_string);
    std::string line;
    nuslists.clear();
    while (std::getline(iss, line))
    {
        std::istringstream iss_line(line);
        int temp;
        while (iss_line >> temp)
        {
            nuslists.push_back(temp);
        }
    }

    std::cout<<"Read "<<nuslists.size()<<" nus points from string"<<std::endl;
    for(int i=0;i<nuslists.size();i++)
    {
        std::cout<<nuslists[i]<<" ";
    }
    std::cout<<std::endl;

    return true;
}

/**
 * Read nus simu list from a file. One line per sampling point or one element per sampling point in one line
 */
bool fid_2d::read_nus_simu_list(std::string fname)
{
    std::ifstream infile(fname);
    /**
     * Real all lines from the file
     */
    std::string contents = std::string((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();
    
    if (contents.empty())
    {
        std::cerr << "Error: nus simu list file is empty or does not exist." << std::endl;
        return false;
    }
    return read_nus_simu_list_from_string(contents);
}

/**
 * Read nus simu list from a string
 * The string should be a space separated list of integers or new line separated list of integers
 */
bool fid_2d::read_nus_simu_list_from_string(const std::string &nus_simu_string)
{
    std::istringstream iss(nus_simu_string);
    std::string line;
    nuslists2.clear();
    while (std::getline(iss, line))
    {
        std::istringstream iss_line(line);
        int temp;
        while (iss_line >> temp)
        {
            nuslists2.push_back(temp);
        }
    }

    std::cout<<"Read "<<nuslists2.size()<<" simulated nus points from string"<<std::endl;
    for(int i=0;i<nuslists2.size();i++)
    {
        std::cout<<nuslists2[i]<<" ";
    }
    std::cout<<std::endl;

    return true;
}


/**
 * Load user provided phase correction values from a file
 * 4 values:
 * p0 (direct), p1 (direct), p0 (indirect), p1 (indirect)
*/
bool fid_2d::read_phase_correction(std::string fname)
{
    std::ifstream infile(fname);
    std::string contents = std::string((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();
    return read_phase_correction_from_string(contents);
}

bool fid_2d::read_phase_correction_from_string(const std::string &phase_correction_text)
{
    std::istringstream iss(phase_correction_text);
    if (!(iss >> user_p0_direct >> user_p1_direct >> user_p0_indirect >> user_p1_indirect))
    {
        std::cout << "Error: phase correction string must contain 4 numbers" << std::endl;
        return false;
    }
    std::cout << "User provided phase correction values: " << user_p0_direct << " " << user_p1_direct << " " << user_p0_indirect << " " << user_p1_indirect << std::endl;
    return true;
}

/**
 * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
 * this function will change the value of the following variables:
 * fid_data_float (unused one will have size 0)
 * data_type, data_complexity
 * ndata_bruker, ndata
 * udict_acqus
 * spectral_width, observed_frequency, carrier_frequency
 */
bool fid_2d::read_bruker_folder(std::string folder_name)
{
    /**
     * @brief test if the folder exists, using system call stat
     * sb is a struct of type stat
     */
    struct stat sb;

    /**
     * Calls the function with path as argument. This is a C function.
     * If the file/directory exists at the path returns 0
     */

    int status = stat(folder_name.c_str(), &sb);

    if (status != 0)
    {
        std::cerr << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }
    else if (!(sb.st_mode & S_IFDIR))
    {
        // exist but is not a directory
        std::cerr << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }

    /**
     * now look for folder_name/fid or folder_name/ser
     * If none of them exist, return false
     */

    std::string fid_data_file_name = folder_name + "/fid";
    status = stat(fid_data_file_name.c_str(), &sb);
    if (status == 0)
    {
    }
    else
    {
        fid_data_file_name = folder_name + "/ser";
        status = stat(fid_data_file_name.c_str(), &sb);
        if (status == 0)
        {
        }
        else
        {
            fid_data_file_name = ""; // label it as empty
            std::cerr<< "Error: cannot find fid or ser file in folder " << folder_name << std::endl;
            return false;
        }
    }

    /**
     * Read acqus file and store the content. Required for direct dimension
     */
    std::string acqus_file_name = folder_name + "/acqus";
    status = stat(acqus_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cerr << "Error: cannot find acqus file in folder " << folder_name << std::endl;
        return false;
    }

    /**
     * Read acqus file and store the content. Required for indirect dimension
     */
    std::string acqus_file2_name = folder_name + "/acqu2s";
    status = stat(acqus_file2_name.c_str(), &sb);
    if (status != 0)
    {
        std::cerr << "Error: cannot find acqu2s file in folder " << folder_name << std::endl;
        return false;
    }

    /**
     * Read pulseprogram file and store the content.
     * This file is optional
     */
    std::string pulse_program_file_name = folder_name + "/pulseprogram";
    status = stat(pulse_program_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cout << "Warning: cannot find pulseprogram file in folder " << folder_name << std::endl;
        pulse_program_file_name = "";
    }

    /**
     * if we are here, we have found the fid data file and acqus file
     */
    std::vector<std::string> fid_data_file_names;
    fid_data_file_names.push_back(fid_data_file_name);
    return read_bruker_files(pulse_program_file_name,acqus_file2_name, acqus_file_name, fid_data_file_names);
};


/**
 * @brief read_pulse_program: process pulse program file as a string and extract phase correction values
 * This function will change the value of the following variables:
 * indirect_p0, indirect_p1
 */
bool fid_2d::read_bruker_files_as_strings(const std::string &contents_pulse, const std::string &contents_acqus, const std::string &contents_acqus2)
{
    /**
     * At this time, we only try to read these two lines from pulseprogram file
     * ;PHC0(F1): 90
     * ;PHC1(F1): -180
     * It is possible we can only read p0 but not p1, in that case, p1 = 0.0.
     * If we cannot read both, we will set indirect_p0 = -360.0 (moonlighting as a flag) and indirect_p1 = 0.0
    */
    indirect_p0 = -400.0; 
    indirect_p1 = 0.0;

    std::istringstream infile(contents_pulse);

    std::string line;
    while (std::getline(infile, line))
    {
        if (line.find(";PHC0(F1):") != std::string::npos)
        {
            indirect_p0 = std::stod(line.substr(10));
        }
        else if (line.find(";PHC1(F1):") != std::string::npos)
        {
            indirect_p1 = std::stod(line.substr(10));
        }
    }

    bool b1=process_jcamp_as_string(contents_acqus, udict_acqus_direct);
    bool b2=process_jcamp_as_string(contents_acqus2, udict_acqus_indirect);

    bool b3=process_dictionary();

    return b1 && b2 && b3;
};

/**
 * This function is to receive FID data directly from a vector of float
 * It is for binding to web assemblies or other applications where we want to avoid file I/O.
 */
bool fid_2d::read_bruker_fid_data(const std::vector<float> &fid_data_float_in)
{
    const int points_per_spectrum = ndata_bruker * ndata_bruker_indirect;
    if (points_per_spectrum <= 0)
    {
        std::cout << "Error: invalid Bruker dimensions. Call read_bruker_files_as_strings first." << std::endl;
        return false;
    }
    if (fid_data_float_in.empty() || (fid_data_float_in.size() % points_per_spectrum) != 0)
    {
        std::cout << "Error: Bruker float payload size does not match expected dimensions" << std::endl;
        return false;
    }

    nspectra = static_cast<int>(fid_data_float_in.size() / points_per_spectrum);
    fid_data_float = fid_data_float_in;

    std::cout << "Read " << nspectra << " spectra from memory payload" << std::endl;
    std::cout << " ndata_bruker = " << ndata_bruker << " ndata_bruker_indirect = " << ndata_bruker_indirect << std::endl;

    return process_fid_data();
}

bool fid_2d::read_bruker_fid_data_bytes(const std::vector<unsigned char> &fid_data_bytes)
{
    if (fid_data_bytes.empty())
    {
        std::cout << "Error: empty Bruker FID payload" << std::endl;
        return false;
    }

    const int points_per_spectrum = ndata_bruker * ndata_bruker_indirect;
    if (points_per_spectrum <= 0)
    {
        std::cout << "Error: invalid Bruker dimensions. Call read_bruker_files_as_strings first." << std::endl;
        return false;
    }

    fid_data_float.clear();

    if (data_type == FID_DATA_TYPE_INT32)
    {
        if ((fid_data_bytes.size() % sizeof(int32_t)) != 0)
        {
            std::cout << "Error: Bruker INT32 payload byte size is not aligned" << std::endl;
            return false;
        }
        const size_t n_points = fid_data_bytes.size() / sizeof(int32_t);
        if (n_points == 0 || (n_points % points_per_spectrum) != 0)
        {
            std::cout << "Error: Bruker INT32 payload size does not match expected dimensions" << std::endl;
            return false;
        }

        nspectra = static_cast<int>(n_points / points_per_spectrum);
        fid_data_float.resize(n_points, 0.0f);
        for (size_t i = 0; i < n_points; i++)
        {
            int32_t v = 0;
            std::memcpy(&v, fid_data_bytes.data() + i * sizeof(int32_t), sizeof(int32_t));
            fid_data_float[i] = static_cast<float>(v);
        }
    }
    else if (data_type == FID_DATA_TYPE_FLOAT64)
    {
        if ((fid_data_bytes.size() % sizeof(double)) != 0)
        {
            std::cout << "Error: Bruker FLOAT64 payload byte size is not aligned" << std::endl;
            return false;
        }
        const size_t n_points = fid_data_bytes.size() / sizeof(double);
        if (n_points == 0 || (n_points % points_per_spectrum) != 0)
        {
            std::cout << "Error: Bruker FLOAT64 payload size does not match expected dimensions" << std::endl;
            return false;
        }

        nspectra = static_cast<int>(n_points / points_per_spectrum);
        fid_data_float.resize(n_points, 0.0f);
        for (size_t i = 0; i < n_points; i++)
        {
            double v = 0.0;
            std::memcpy(&v, fid_data_bytes.data() + i * sizeof(double), sizeof(double));
            fid_data_float[i] = static_cast<float>(v);
        }
    }
    else
    {
        std::cout << "Error: unsupported Bruker data type" << std::endl;
        return false;
    }

    std::cout << "Read " << nspectra << " spectra from byte payload" << std::endl;
    std::cout << " ndata_bruker = " << ndata_bruker << " ndata_bruker_indirect = " << ndata_bruker_indirect << std::endl;

    return process_fid_data();
}

/**
 * This is the main function to read Bruker data
 * It requires 1 pulse_program_name, 2 acqus files and a vector of fid data file names (1 is also ok)
 */
bool fid_2d::read_bruker_files(const std::string &pulse_program_name,const std::string &acqus_file2_name, const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names)
{
    /**
     * At this time, we only try to read these two lines from pulseprogram file
     * ;PHC0(F1): 90
     * ;PHC1(F1): -180
     * It is possible we can only read p0 but not p1, in that case, p1 = 0.0.
     * If we cannot read both, we will set indirect_p0 = -360.0 (moonlighting as a flag) and indirect_p1 = 0.0
    */
    indirect_p0 = -400.0; 
    indirect_p1 = 0.0; 

    if(pulse_program_name != "")
    {
        std::ifstream infile(pulse_program_name);
        std::string line;
        while (std::getline(infile, line))
        {
            if (line.find(";PHC0(F1):") != std::string::npos)
            {
                indirect_p0 = std::stod(line.substr(10));
            }
            else if (line.find(";PHC1(F1):") != std::string::npos)
            {
                indirect_p1 = std::stod(line.substr(10));
            }
        }
    }
    
    if(!read_jcamp(acqus_file_name, udict_acqus_direct) || !read_jcamp(acqus_file2_name, udict_acqus_indirect))
    {
        std::cerr << "Error: cannot read acqus or acqu2s file" << std::endl;
        return false;
    }

    if(!process_dictionary())
    {
        std::cerr << "Error: cannot process dictionary" << std::endl;
        return false;
    }

    /**
     * now we can actually read the fid data
     * For complex data, real and imaginary parts are stored interleaved by Bruker.
     * Here we leave them interleaved in fid_data_float or fid_data_int
     *
     * Major order of fid_data_float:
     * indrect_dim, direct_dim, real/imaginary interleaved
     */
    for (int i = 0; i < fid_data_file_names.size(); i++)
    {
        std::vector<int32_t> fid_data_int;       // 32 bit
        std::vector<double> temp_fid_data_float; // 64 bit
        FILE *fp_fid_data = fopen(fid_data_file_names[i].c_str(), "rb");

        if (data_type == FID_DATA_TYPE_INT32)
        {
            nspectra = 0;
            fid_data_int.clear();
            fid_data_int.resize(ndata_bruker * ndata_bruker_indirect);
            while(fread(fid_data_int.data()+nspectra*ndata_bruker * ndata_bruker_indirect, sizeof(int32_t), ndata_bruker * ndata_bruker_indirect, fp_fid_data)==ndata_bruker * ndata_bruker_indirect)
            {
                nspectra++;
                /**
                 * Get space for the next spectrum (in case of pseudo 3D NMR)
                 */
                fid_data_int.resize(ndata_bruker * ndata_bruker_indirect * (nspectra + 1));
            }
            /**
             * Set correct size for fid_data_int
             */
            fid_data_int.resize(ndata_bruker * ndata_bruker_indirect * nspectra);
            if(nspectra==0)
            {
                std::cout << "Error: cannot read int32 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT64)
        {
            nspectra = 0;
            temp_fid_data_float.clear();
            temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect);
            nspectra = 0;
            while(fread(temp_fid_data_float.data()+nspectra*ndata_bruker * ndata_bruker_indirect, sizeof(double), ndata_bruker * ndata_bruker_indirect, fp_fid_data) == ndata_bruker * ndata_bruker_indirect)
            {
                nspectra++;
                /**
                 * Get space for the next spectrum (in case of pseudo 3D NMR)
                 */
                temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect * (nspectra + 1));
            }
            /**
             * Set correct size for temp_fid_data_float
             */
            temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect * nspectra);
            if(nspectra==0)
            {
                std::cout << "Error: cannot read float64 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }

        /**
         * Add data to fid_data_float
         */
        fid_data_float.resize(ndata_bruker * ndata_bruker_indirect * nspectra, 0.0);
        if (data_type == FID_DATA_TYPE_INT32)
        {
            for (int i = 0; i < ndata_bruker * ndata_bruker_indirect * nspectra; i++)
            {
                fid_data_float[i] += (float)fid_data_int[i];
            }
        }
        else
        {
            for (int i = 0; i < ndata_bruker * ndata_bruker_indirect * nspectra; i++)
            {
                fid_data_float[i] += (float)temp_fid_data_float[i];
            }
        }
        fclose(fp_fid_data);
    }

    std::cout << "Read " << nspectra << " spectra from " << fid_data_file_names.size() << " files" << std::endl;
    std::cout <<" ndata_bruker = "<<ndata_bruker<<" ndata_bruker_indirect = "<<ndata_bruker_indirect<<std::endl;

    return process_fid_data();
}

bool fid_2d::process_fid_data()
{
    /**
     * Now we need to reorganize fid_data_float when order is 321 (from inner to outer), which means order is
     * - indirect dimension (acqu3s ==> 1)
     * - - pseudo dimension (acqu2s ==> 2)
     * - - - direct dimension (acqus ==> 3)
     * We need to reorganize it to 312, which means order is: pseudo dimension, indirect dimension, direct dimension
     * then we can process all 2D spectra at one pseudo dimension data point one by one
    */
    if(nspectra>1)
    {
        if(aqseq=="321")
        {
            /**
             * Rroganize fid_data_float, in case 2nd dimension is pseudo 3D dimension while 3rd dimension is the indirect dimension
            */
            std::vector<float> fid_data_float_temp=fid_data_float;
            fid_data_float.clear();
            fid_data_float.resize(nspectra * ndata_bruker * ndata_bruker_indirect, 0.0f);
            for(int m=0;m<nspectra;m++)
            {
                for(int i=0;i<ndata_bruker_indirect;i++)
                {
                    int k_from = m + nspectra * i;
                    int k_to = m * ndata_bruker_indirect + i;
                    /**
                     * copy from fid_data_float_temp[k_from*ndata_bruker:(k_from+1)*ndata_bruker] to 
                     * fid_data_float[k_to*ndata_bruker:(k_to+1)*ndata_bruker]
                     */
                    std::memcpy(fid_data_float.data()+k_to*ndata_bruker,fid_data_float_temp.data()+k_from*ndata_bruker,sizeof(float)*ndata_bruker);
                }
            }
            /**
             * Release memory
             */
            fid_data_float_temp.clear();
        }
        else if(aqseq=="312")
        {
            /**
             * Order is already 312 (from iner to outer), do nothing
             * - pseudo dimension (acqu2s ==> 2)
             * - - indirect dimension (acqu3s ==> 1)
             * - - - direct dimension (acqus ==> 3)
            */
        }
        else
        {
            /**
             * Won't happen because we have checked aqseq in set_aqseq
            */
        }
    }

    /**
     * If b_first_only is true, we only process the first spectrum
     */
    if(b_first_only)
    {
        nspectra=1;
        /**
         * Resize fid_data_float to contain only the first spectrum, discard the rest
         */
        fid_data_float.resize(ndata_bruker * ndata_bruker_indirect);
    }

    /**
     * Expand fid_data_float to its original size (with zeros in non-sampled traces) if nuslists is not empty
     */
    if(ndata_bruker_indirect_original > ndata_bruker_indirect)
    {
        /**
         * Use a temporary buffer to safely expand data regardless of nuslists order.
         * This avoids the assumption that nuslists is sorted and prevents data corruption
         * from in-place moves.
         */
        std::vector<float> temp_compressed = fid_data_float;
        fid_data_float.clear();
        fid_data_float.resize(nspectra * ndata_bruker * ndata_bruker_indirect_original, 0.0f);
        
        /**
         * Copy each sampled trace from compressed buffer to its correct location in expanded buffer
         * We iterate over nuslists in forward order, which works regardless of whether
         * it is sorted or not
         */
        for(int m=0; m<nspectra; m++)
        {
            int spectrum_start_target = m * ndata_bruker * ndata_bruker_indirect_original;
            int spectrum_start_source = m * ndata_bruker * ndata_bruker_indirect;
            
            for(int n=0; n<nuslists.size(); n++)
            {
                /**
                 * Source location in compressed buffer: index n*2 (real) and n*2+1 (imaginary)
                 * Target location in expanded buffer: index nuslists[n]*2 (real) and nuslists[n]*2+1 (imaginary)
                 */
                
                /**
                 * Copy real part (at index n*2 in compressed)
                 */
                int compressed_trace_real = n * 2;
                int expanded_trace_real = nuslists[n] * 2;
                int source_start_real = spectrum_start_source + compressed_trace_real * ndata_bruker;
                int target_start_real = spectrum_start_target + expanded_trace_real * ndata_bruker;
                std::memcpy(fid_data_float.data() + target_start_real,
                           temp_compressed.data() + source_start_real,
                           sizeof(float) * ndata_bruker);
                
                /**
                 * Copy imaginary part (at index n*2+1 in compressed)
                 */
                int compressed_trace_imag = n * 2 + 1;
                int expanded_trace_imag = nuslists[n] * 2 + 1;
                int source_start_imag = spectrum_start_source + compressed_trace_imag * ndata_bruker;
                int target_start_imag = spectrum_start_target + expanded_trace_imag * ndata_bruker;
                std::memcpy(fid_data_float.data() + target_start_imag,
                           temp_compressed.data() + source_start_imag,
                           sizeof(float) * ndata_bruker);
            }
        }
        
        /**
         * Restore ndata_bruker_indirect to its original size
         */
        ndata_bruker_indirect = ndata_bruker_indirect_original;
        ndata_indirect = ndata_bruker_indirect / 2;
    }

    /**
     * Keep this cached value in sync with the possibly expanded indirect size.
     * It is used by downstream loops (FnMODE conversion and rr/ri/ir/ii split).
     */
    ndata_bruker_indirect_by2 = ndata_bruker_indirect;
    if (ndata_bruker_indirect % 2 == 1)
    {
        ndata_bruker_indirect_by2 = ndata_bruker_indirect - 1;
    }
       
    /**
     * Adjust indirect dimension data format according to FnMODE
     * fid_data_float[ndata_bruker_indirect][ndata_bruker]. Data is interleaved along both dimensions (real,imaginary,real,imaginary,...)
     */
    if (fnmode == 6)
    {
        for(int index_spectrum=0;index_spectrum<nspectra;index_spectrum++)
        {
            int data_start = index_spectrum * ndata_bruker * ndata_bruker_indirect;
            /**
             * Echo anti-echo pulse sequence.
             *
             * Step 1: along indirect dimension, (increment 0 - increment 1)/2.0 ==> increment 0
             * Step 2: along indirect dimension, (increment 0 + increment 1)/2.0 ==> increment 1, and so on
             * Step 3, for increment 1,3,5,..., apply 90 degree correction along direct dimension (i.e., multiply by i, real -> imaginary, imaginary -> -real)
             */


            for (int i = 0; i < ndata_bruker_indirect_by2; i += 2)
            {
                std::vector<float> temp1(ndata_bruker, 0.0f), temp2(ndata_bruker, 0.0f), temp3(ndata_bruker, 0.0f);
                for (int j = 0; j < ndata_bruker; j++)
                {
                    /**
                     * Step 1 here
                    */
                    temp1[j] = (fid_data_float[data_start + i * ndata_bruker + j] - fid_data_float[data_start + (i + 1) * ndata_bruker + j]) / 2.0f;
                    /**
                     * Step 2 here
                    */
                    temp2[j] = (fid_data_float[data_start + i * ndata_bruker + j] + fid_data_float[data_start + (i + 1) * ndata_bruker + j]) / 2.0f;
                }
                /**
                 * Step 3 here
                 */
                for (int k = 0; k < ndata_bruker; k += 2)
                {
                    temp3[k] = -temp2[k + 1];
                    temp3[k + 1] = temp2[k];
                }
                /**
                 * Copy temp1 and temp3 back to fid_data_float
                 */
                for (int j = 0; j < ndata_bruker; j++)
                {
                    fid_data_float[data_start + i * ndata_bruker + j] = temp1[j];
                    fid_data_float[data_start + (i + 1) * ndata_bruker + j] = temp3[j];
                }
            } // for (int i = 0; i < ndata_bruker_indirect; i += 2)
        } // for(int index_spectrum=0;index_spectrum<nspectra;index_spectrum++)
    }

    /**
     * Convert fid_data_float to active stage container (spectrum_* in FID stage).
     * fid_data_float is interleaved (rr, ri of 1st trace, ir, ii of 1st trace, rr, ri of 2nd trace, ...)
     */
    spectrum_real_real.resize(nspectra * ndata * ndata_indirect, 0.0f);
    spectrum_real_imag.resize(nspectra * ndata * ndata_indirect, 0.0f);
    spectrum_imag_real.resize(nspectra * ndata * ndata_indirect, 0.0f);
    spectrum_imag_imag.resize(nspectra * ndata * ndata_indirect, 0.0f);
    for(int k=0;k<nspectra;k++)
    {
        int data_start = k * ndata_bruker * ndata_bruker_indirect;
        int data_start2 = k * ndata * ndata_indirect; //same for all 4 vector. 
        for (int i = 0; i < ndata_bruker_indirect_by2; i+=2)
        {
            int trace_start = i * ndata_bruker;
            int trace_start2 = i/2 * ndata;
            for (int j = 0; j < ndata_bruker; j += 2)
            {
                spectrum_real_real[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + i * ndata_bruker + j];
                spectrum_real_imag[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + i * ndata_bruker + j + 1];
                spectrum_imag_real[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + (i + 1) * ndata_bruker + j];
                spectrum_imag_imag[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + (i + 1) * ndata_bruker + j + 1];
            }
        }
    }
    
    /**
     * To save memory, we can release fid_data_float
     */
    fid_data_float.clear();

    b_frq = false;
    b_frq_indirect = false;
    ndata_frq = 0;
    ndata_frq_indirect = 0;
    b_nmrPipe_transposed = false;
    n_outer_dim = ndata_indirect;
    n_inner_dim = ndata;
    spect = spectrum_real_real.data();

    /**
     * Define nusflags, for convenience. nusflags.size() == ndata_indirect
     * and nusflags[i] == 1 means the i-th trace is sampled, 0 means the i-th trace is not sampled
     */
    if(nuslists.size()>0)
    {
        nusflags.clear();
        nusflags.resize(ndata_indirect, 0);
        for(int i=0;i<nuslists.size();i++)
        {
            nusflags[nuslists[i]]=1;
        }
    }
    /**
     * For simulate nus, we have full data, but we want to mark some traces as non-sampled for testing. So we use nuslists2 to mark non-sampled traces, and set nusflags to 0 for those traces.
     * Note that nuslists2 is for simulated nus, so it marks non-sampled traces, while nuslists is for real nus, so it marks sampled traces. This is a bit confusing
     */
    else if(nuslists2.size()>0)
    {
        nusflags.clear();
        nusflags.resize(ndata_indirect, 0);  // mark all as non-sampled first
        for(int i=0;i<nuslists2.size();i++)
        {
            if(nuslists2[i] >= 0 && nuslists2[i] < ndata_indirect)  // bounds check
            {
                nusflags[nuslists2[i]]=1;  // mark as sampled if it is in nuslists2
            }
            else
            {
                std::cerr << "Warning: nuslists2[" << i << "]=" << nuslists2[i] << " is out of range [0," << ndata_indirect << "), skipping." << std::endl;
            }
        }
    }
    else
    { 
        /**
         * nuslists is empty. This is a fully sampled experiments, so all traces are sampled.
        */
        nusflags.clear();
        nusflags.resize(ndata_indirect, 1);
    }

    return true;
}



bool fid_2d::process_dictionary()
{
    /**
     * Check udic_acqus["AQ_mod"] to determine the data type: real or complex
     */
    if (udict_acqus_direct["AQ_mod"] == "3" || udict_acqus_direct["AQ_mod"] == "1")
    {
        data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    }
    else // 0 or 2
    {
        data_complexity = FID_DATA_COMPLEXITY_REAL;
        std::cerr << "Error: FID data is real, not complex" << std::endl;
        return false;
    }

    /**
     * Check udic_acqus["BYTORDA"] to determine the data type: int32 or float32
     */
    if (udict_acqus_direct["DTYPA"] == "2")
    {
        data_type = FID_DATA_TYPE_FLOAT64;
    }
    else
    {
        data_type = FID_DATA_TYPE_INT32;
    }

    /**
     * check udic_acqus["TD"] to determine size of fid data
     */
    if (udict_acqus_direct.find("TD") == udict_acqus_direct.end())
    {
        std::cout << "Error: cannot find TD in acqus file" << std::endl;
        return false;
    }

    int td0 = std::stoi(udict_acqus_direct["TD"]);

    /**
     * Now we need to check the indirect dimension
     * check udic_acqus_indirect["TD"] to determine size of fid data
     */
    if (udict_acqus_indirect.find("TD") == udict_acqus_indirect.end())
    {
        std::cout << "Error: cannot find TD in acqu2s file" << std::endl;
        return false;
    }
    int td2 = std::stoi(udict_acqus_indirect["TD"]);


    /**
     * Check for indirect dimension encoding
     */
    if (udict_acqus_indirect.find("FnMODE") == udict_acqus_indirect.end())
    {
        std::cout << "Error: cannot find FnMODE in acqu2s file" << std::endl;
        return false;
    }
    fnmode = std::stoi(udict_acqus_indirect["FnMODE"]);

    /**
     * According to Bruker manu,
     * when data_type = FID_DATA_TYPE_FLOAT64, fid is padded to 128 bytes
     * when data_type = FID_DATA_TYPE_INT32, fid is padded to 256 bytes
     * ndata_bruker is the direct dimension size
     */

    if (data_type == FID_DATA_TYPE_INT32)
    {
        int ntemp = int(std::ceil((double)td0 / 256.0));
        ndata_bruker = ntemp * 256;
    }
    else
    {
        int ntemp = int(std::ceil((double)td0 / 128.0));
        ndata_bruker = ntemp * 128;
    }

    ndata_bruker_indirect = td2;

      /**
     * Bruker TD is the number of data points, not the number of complex points for complex data
     * But we define ndata as the number of complex points
     */
    if (data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        ndata = ndata_bruker / 2;
        ndata_indirect = ndata_bruker_indirect / 2;
        ndata_original = td0 / 2;
    }
    else
    {
        ndata = ndata_bruker;
        ndata_indirect = ndata_bruker_indirect;
        ndata_original = td0;
    }

    nucleus = udict_acqus_direct["NUC1"];
    nucleus_indirect = udict_acqus_indirect["NUC1"];

        /**
     * get parameters "GRPDLY" from acqus file
     */

    if (udict_acqus_direct.find("GRPDLY") != udict_acqus_direct.end())
    {
        grpdly = std::stod(udict_acqus_direct["GRPDLY"]);
        /**
         * grpdly must > 0. Otherwiae it is from an early days Bruker spectrometer
         * We don't support this case at this time
         */
        if (grpdly <= 0.0)
        {
            std::cout << "Error: GRPDLY = " << grpdly << " is not supported" << std::endl;
        }
    }
    else
    {
        std::cout << "Error: cannot find GRPDLY in acqus file" << std::endl;
    }

    /**
     * If nuslists is not empty, we need to adjustt ndata_bruker_indirect
     * We save the original value in ndata_bruker_indirect_original, so that we can expand the data to its original size (with zeros in non-sampled traces)
     */
    ndata_bruker_indirect_original = ndata_bruker_indirect;
    if(nuslists.size()>0)
    {
        ndata_bruker_indirect = nuslists.size()*2; //complex data
        ndata_bruker_indirect_original = (nuslists[nuslists.size()-1]+1)*2;
    }

     /**
     * get receiver_gain from acqus if it exists
     */
    if (udict_acqus_direct.find("RG") != udict_acqus_direct.end())
    {
        receiver_gain = std::stod(udict_acqus_direct["RG"]);
    }
    else
    {
        std::cout << "Warning: cannot find RG in acqus file" << std::endl;
        receiver_gain = 1.0; // default
    }

    /**
     * Now setup the following variables:
     * spectral_width, observed_frequency, carrier_frequency
     */
    if (udict_acqus_direct.find("SW_h") != udict_acqus_direct.end())
    {
        spectral_width = std::stod(udict_acqus_direct["SW_h"]);
    }
    else
    {
        std::cout << "Error: cannot find SW_h in acqus file" << std::endl;
        return false;
    }

    if (udict_acqus_direct.find("SFO1") != udict_acqus_direct.end())
    {
        observed_frequency = std::stod(udict_acqus_direct["SFO1"]);
    }
    else
    {
        std::cout << "Error: cannot find SFO1 in acqus file" << std::endl;
        return false;
    }

    if (udict_acqus_direct.find("O1") != udict_acqus_direct.end())
    {
        carrier_frequency = std::stod(udict_acqus_direct["O1"]);
    }
    else
    {
        std::cout << "Error: cannot find O1 in acqus file" << std::endl;
        return false;
    }

    /**
     * Do the same for indirect dimension
     */
    if (udict_acqus_indirect.find("SW_h") != udict_acqus_indirect.end())
    {
        spectral_width_indirect = std::stod(udict_acqus_indirect["SW_h"]);
    }
    else
    {
        std::cout << "Error: cannot find SW_h in acqu2s file" << std::endl;
        return false;
    }

    if (udict_acqus_indirect.find("SFO1") != udict_acqus_indirect.end())
    {
        observed_frequency_indirect = std::stod(udict_acqus_indirect["SFO1"]);
    }
    else
    {
        std::cout << "Error: cannot find SFO1 in acqu2s file" << std::endl;
        return false;
    }

    if (udict_acqus_indirect.find("O1") != udict_acqus_indirect.end())
    {
        carrier_frequency_indirect = std::stod(udict_acqus_indirect["O1"]);
    }
    else
    {
        std::cout << "Error: cannot find O1 in acqu2s file" << std::endl;
        return false;
    }

    n_center = int(ndata / 2.0f)+1;
    n_center_indirect = int(ndata_indirect / 2.0f)+1;
    origin_frequency = (carrier_frequency - spectral_width * (ndata - n_center) / ndata);
    origin_frequency_indirect = (carrier_frequency_indirect - spectral_width_indirect * (ndata_indirect - n_center_indirect) / ndata_indirect);

    b_read_bruker_acqus_and_fid = true;
    b_read_nmrpipe_fid = false;



    ndata_bruker_indirect_by2 = ndata_bruker_indirect;
    /**
     * In case ndata_bruker_indirect is odd, the last trace is not used. We need to adjust ndata_bruker_indirect_by2
     */
    if (ndata_bruker_indirect % 2 == 1)
    {
        ndata_bruker_indirect_by2 = ndata_bruker_indirect - 1;
        /**
         * ndata_indirect is always ndata_bruker_indirect_by2/2, because we write
         * ndata_indirect = ndata_bruker_indirect/2. When ndata_bruker_indirect is odd, ndata_indirect (type is int) will discard the 0.5
         */
    }

    n_outer_dim = ndata_indirect;
    n_inner_dim = ndata;

    return true;
}



/**
 * Read nmrPipe .fid file
 */
bool fid_2d::read_nmrpipe_file(const std::string &fname)
{
    /**
     * Read 512 float32 from file as nmrPipe header
     */

    nmrpipe_header_data.clear();
    nmrpipe_header_data.resize(512, 0.0f);

    FILE *fp = fopen(fname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }
    int n_read = fread(nmrpipe_header_data.data(), sizeof(float), 512, fp);
    if (n_read != 512)
    {
        std::cout << "Error: cannot read 512 float from file " << fname << std::endl;
        fclose(fp);
        return false;
    }

    nmrPipe::nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    if (nmrpipe_dict_float["FDDIMCOUNT"] != 2.0)
    {
        std::cout << "FDDIMCOUNT = " << nmrpipe_dict_float["FDDIMCOUNT"] << ", this is not true 2D data" << std::endl;
    }

    set_varibles_from_nmrpipe_dictionary();


    /**
     * Read time domain data if FDF2FTFLAG = 0 and FDF1FTFLAG = 0
     * We suppose all data is complex at this time
     */
    if (b_frq == false && b_frq_indirect == false)
    {
        spectrum_real_real.resize(ndata * ndata_indirect, 0.0f);
        spectrum_real_imag.resize(ndata * ndata_indirect, 0.0f);
        spectrum_imag_real.resize(ndata * ndata_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata * ndata_indirect, 0.0f);


        for (int i = 0; i < ndata_indirect; i++)
        {
            /**
             * At this time, we suppose the data is complex for time domain data
            */
            if (fread(spectrum_real_real.data() + i * ndata, sizeof(float), ndata, fp) != static_cast<size_t>(ndata) ||
                fread(spectrum_real_imag.data() + i * ndata, sizeof(float), ndata, fp) != static_cast<size_t>(ndata) ||
                fread(spectrum_imag_real.data() + i * ndata, sizeof(float), ndata, fp) != static_cast<size_t>(ndata) ||
                fread(spectrum_imag_imag.data() + i * ndata, sizeof(float), ndata, fp) != static_cast<size_t>(ndata))
            {
                std::cout << "Error: cannot read time domain data from file " << fname << std::endl;
                fclose(fp);
                return false;
            }
        }
        b_frq = false;
        b_frq_indirect = false;
    }
    /**
     * Read frq data if FDF2FTFLAG = 1 and FDF1FTFLAG = 1
    */
    else if(b_frq == true && b_frq_indirect == true)
    {
        ndata_frq = n_inner_dim;
        ndata_frq_indirect = n_outer_dim;

        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_real_imag.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        

        for (int i = 0; i < n_outer_dim; i++)
        {
            if (fread(spectrum_real_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
            {
                std::cout << "Error: cannot read spectrum_real_real data from file " << fname << std::endl;
                fclose(fp);
                return false;
            }
            if(b_imaginary==true)
            {
                if (fread(spectrum_real_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
                {
                    std::cout << "Error: cannot read spectrum_real_imag data from file " << fname << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if(b_imaginary_indirect==true)
            {
                if (fread(spectrum_imag_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
                {
                    std::cout << "Error: cannot read spectrum_imag_real data from file " << fname << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            if(b_imaginary==true && b_imaginary_indirect==true)
            {
                if (fread(spectrum_imag_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
                {
                    std::cout << "Error: cannot read spectrum_imag_imag data from file " << fname << std::endl;
                    fclose(fp);
                    return false;
                }
            }
        }
        b_frq = true;
        b_frq_indirect = true;
    }
        /**
     * Read intermediate data if FDF2FTFLAG = 1 and FDF1FTFLAG = 0
     * We supose b_imaginary_indirect = true
     */
    else if (b_frq == true && b_frq_indirect == false)
    {
        spectrum_real_real.resize(ndata_frq * ndata_indirect, 0.0f);
        spectrum_real_imag.resize(ndata_frq * ndata_indirect, 0.0f);
        spectrum_imag_real.resize(ndata_frq * ndata_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata_frq * ndata_indirect, 0.0f);

        std::cout<<"n_out_dim = "<<n_outer_dim<<" n_in_dim = "<<n_inner_dim<<std::endl;

        for (int i = 0; i < n_outer_dim; i++)
        {
            if (fread(spectrum_real_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
            {
                std::cout << "Error: cannot read spectrum_real_real data from file " << fname << std::endl;
                fclose(fp);
                return false;
            }
            if(b_imaginary==true)
            {
                if (fread(spectrum_real_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim) ||
                    fread(spectrum_imag_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim) ||
                    fread(spectrum_imag_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
                {
                    std::cout << "Error: cannot read complex intermediate data from file " << fname << std::endl;
                    fclose(fp);
                    return false;
                }
            }
            else
            {
                if (fread(spectrum_imag_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp) != static_cast<size_t>(n_inner_dim))
                {
                    std::cout << "Error: cannot read spectrum_imag_real data from file " << fname << std::endl;
                    fclose(fp);
                    return false;
                }
            }
        }
        b_frq = true;
        b_frq_indirect = false;
    }
    else
    {
        std::cout << "Error: FDF2FTFLAG = " << nmrpipe_dict_float["FDF2FTFLAG"] << ", FDF1FTFLAG = " << nmrpipe_dict_float["FDF1FTFLAG"] << " is not supported" << std::endl;
        fclose(fp);
        return false;
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;

    nspectra = 1; // only one spectrum, not a pseudo 3D NMR

    spect = spectrum_real_real.data(); //alias to active stage container

    nusflags.clear();
    nusflags.resize(ndata_indirect, 1);

    estimate_noise_level();

    fclose(fp);

    return true;
}

bool fid_2d::read_nmrpipe_file_from_buffer(const std::vector<unsigned char> &nmrpipe_bytes)
{
    const size_t header_bytes = 512 * sizeof(float);
    if (nmrpipe_bytes.size() < header_bytes)
    {
        std::cout << "Error: nmrPipe buffer is too small to contain header" << std::endl;
        return false;
    }

    nmrpipe_header_data.clear();
    nmrpipe_header_data.resize(512, 0.0f);
    std::memcpy(nmrpipe_header_data.data(), nmrpipe_bytes.data(), header_bytes);

    nmrPipe::nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    if (nmrpipe_dict_float["FDDIMCOUNT"] != 2.0)
    {
        std::cout << "FDDIMCOUNT = " << nmrpipe_dict_float["FDDIMCOUNT"] << ", this is not true 2D data" << std::endl;
    }

    set_varibles_from_nmrpipe_dictionary();


    size_t offset = header_bytes;
    auto read_trace = [&](float *dst, int n_elem) -> bool {
        const size_t need = static_cast<size_t>(n_elem) * sizeof(float);
        if (offset + need > nmrpipe_bytes.size())
        {
            return false;
        }
        std::memcpy(dst, nmrpipe_bytes.data() + offset, need);
        offset += need;
        return true;
    };

    if (b_frq == false && b_frq_indirect == false)
    {
        spectrum_real_real.resize(ndata * ndata_indirect, 0.0f);
        spectrum_real_imag.resize(ndata * ndata_indirect, 0.0f);
        spectrum_imag_real.resize(ndata * ndata_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata * ndata_indirect, 0.0f);

        for (int i = 0; i < ndata_indirect; i++)
        {
            if (!read_trace(spectrum_real_real.data() + i * ndata, ndata) ||
                !read_trace(spectrum_real_imag.data() + i * ndata, ndata) ||
                !read_trace(spectrum_imag_real.data() + i * ndata, ndata) ||
                !read_trace(spectrum_imag_imag.data() + i * ndata, ndata))
            {
                std::cout << "Error: insufficient nmrPipe payload for time-domain data" << std::endl;
                return false;
            }
        }
    }
    else if (b_frq == true && b_frq_indirect == true)
    {
        ndata_frq = n_inner_dim;
        ndata_frq_indirect = n_outer_dim;

        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_real_imag.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata_frq * ndata_frq_indirect, 0.0f);

        for (int i = 0; i < n_outer_dim; i++)
        {
            if (!read_trace(spectrum_real_real.data() + i * n_inner_dim, n_inner_dim))
            {
                std::cout << "Error: insufficient nmrPipe payload for rr data" << std::endl;
                return false;
            }
            if (b_imaginary == true && !read_trace(spectrum_real_imag.data() + i * n_inner_dim, n_inner_dim))
            {
                std::cout << "Error: insufficient nmrPipe payload for ri data" << std::endl;
                return false;
            }
            if (b_imaginary_indirect == true && !read_trace(spectrum_imag_real.data() + i * n_inner_dim, n_inner_dim))
            {
                std::cout << "Error: insufficient nmrPipe payload for ir data" << std::endl;
                return false;
            }
            if (b_imaginary == true && b_imaginary_indirect == true && !read_trace(spectrum_imag_imag.data() + i * n_inner_dim, n_inner_dim))
            {
                std::cout << "Error: insufficient nmrPipe payload for ii data" << std::endl;
                return false;
            }
        }

    }
    else if (b_frq == true && b_frq_indirect == false)
    {
        spectrum_real_real.resize(ndata_frq * ndata_indirect, 0.0f);
        spectrum_real_imag.resize(ndata_frq * ndata_indirect, 0.0f);
        spectrum_imag_real.resize(ndata_frq * ndata_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata_frq * ndata_indirect, 0.0f);

        for (int i = 0; i < n_outer_dim; i++)
        {
            if (!read_trace(spectrum_real_real.data() + i * n_inner_dim, n_inner_dim))
            {
                std::cout << "Error: insufficient nmrPipe payload for intermediate rr data" << std::endl;
                return false;
            }
            if (b_imaginary == true)
            {
                if (!read_trace(spectrum_real_imag.data() + i * n_inner_dim, n_inner_dim) ||
                    !read_trace(spectrum_imag_real.data() + i * n_inner_dim, n_inner_dim) ||
                    !read_trace(spectrum_imag_imag.data() + i * n_inner_dim, n_inner_dim))
                {
                    std::cout << "Error: insufficient nmrPipe payload for intermediate complex data" << std::endl;
                    return false;
                }
            }
            else if (!read_trace(spectrum_imag_real.data() + i * n_inner_dim, n_inner_dim))
            {
                std::cout << "Error: insufficient nmrPipe payload for intermediate ir data" << std::endl;
                return false;
            }
        }
    }
    else
    {
        std::cout << "Error: FDF2FTFLAG = " << nmrpipe_dict_float["FDF2FTFLAG"] << ", FDF1FTFLAG = " << nmrpipe_dict_float["FDF1FTFLAG"] << " is not supported" << std::endl;
        return false;
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;
    nspectra = 1;
    spect = spectrum_real_real.data();

    nusflags.clear();
    nusflags.resize(ndata_indirect, 1);

    estimate_noise_level();

    std::cout<<"Successfully read nmrPipe data from buffer, n_outer_dim = "<<n_outer_dim<<" n_inner_dim = "<<n_inner_dim<<std::endl;
    std::cout<<"ndata_frq_indirect = "<<ndata_frq_indirect<<" ndata_frq = "<<ndata_frq<<" b_imaginary = "<<b_imaginary<<" b_imaginary_indirect = "<<b_imaginary_indirect<<std::endl;
    return true;
}

/**
 * This function will fillin our own variables based on nmrPipe dictionary. 
 * We need to call this function after reading nmrPipe header and before reading nmrPipe data, because we need to know the size of data to read.
 */
bool fid_2d::set_varibles_from_nmrpipe_dictionary()
{
    /**
     * These are not used in nmrPipe dictionary ?
    */
    if(nmrpipe_dict_float.find("GRPDLY") != nmrpipe_dict_float.end())
    {
        grpdly = nmrpipe_dict_float["GRPDLY"];
        if(grpdly <= 0.0)
        {
            grpdly = -1.0;  // 0.0f means not set in nmrPipe header. We use < 0 as flag.
        }
    }
    
    data_type = FID_DATA_TYPE::FID_DATA_TYPE_NOT_DEFINED; // not defined
    receiver_gain = 1.0;

    /**
     * We read nmrPipe dictionary, we suppose it is complex data (originally)
     * and use flag b_imaginary and b_imaginary_indirect to determine whether we have 
     * Remember in nmrPipe, 0.0 means complex, 1.0 means real
    */
    data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    b_imaginary = nmrpipe_dict_float["FDF2QUADFLAG"] == 0.0;
    

    /**
     * Set spectral_width, observed_frequency, carrier_frequency
     */
    spectral_width = nmrpipe_dict_float["FDF2SW"];
    observed_frequency = nmrpipe_dict_float["FDF2OBS"];
    carrier_frequency = nmrpipe_dict_float["FDF2CAR"]*observed_frequency;
    origin_frequency = nmrpipe_dict_float["FDF2ORIG"];

    if(nmrpipe_dict_float["FDF2FTFLAG"] == 1.0)
    {
        b_frq = true;
    }
    else
    {
        b_frq = false;
    }
   

    /**
     * Set begin1, step1, stop1, begin2, step2, stop2
    */
    int direct_ndx = nmrpipe_dict_float["FDDIMORDER1"]; //must be 2 for no transposed data
    int indirect_ndx = nmrpipe_dict_float["FDDIMORDER2"]; //can be 1 or 3 for no transposed data

    if (indirect_ndx == 3)
    {
        spectral_width_indirect = nmrpipe_dict_float["FDF3SW"];
        observed_frequency_indirect = nmrpipe_dict_float["FDF3OBS"];
        carrier_frequency_indirect = nmrpipe_dict_float["FDF3CAR"] * observed_frequency_indirect;
        origin_frequency_indirect = nmrpipe_dict_float["FDF3ORIG"];
        b_imaginary_indirect = nmrpipe_dict_float["FDF3QUADFLAG"] == 0.0;
        b_frq_indirect = nmrpipe_dict_float["FDF3FTFLAG"] == 1.0;
    }
    else //must be 1
    {
        spectral_width_indirect = nmrpipe_dict_float["FDF1SW"];
        observed_frequency_indirect = nmrpipe_dict_float["FDF1OBS"];
        carrier_frequency_indirect = nmrpipe_dict_float["FDF1CAR"]*observed_frequency_indirect;
        origin_frequency_indirect = nmrpipe_dict_float["FDF1ORIG"];
        b_imaginary_indirect = nmrpipe_dict_float["FDF1QUADFLAG"] == 0.0;
        b_frq_indirect = nmrpipe_dict_float["FDF1FTFLAG"] == 1.0;
    }

    

    /**
     * This is true for nmrPipe data, both frq or time domain data, transposed or not, 
     * when both dimensions are complex, FDSPECNUM is 2 * n_outer_dim; 
     * when one dimension is real, FDSPECNUM is n_outer_dim; when both dimensions are real, FDSPECNUM is n_outer_dim
    */
    n_outer_dim = int(nmrpipe_dict_float["FDSPECNUM"]);
    n_inner_dim = int(nmrpipe_dict_float["FDSIZE"]);

         
    if(b_imaginary && b_imaginary_indirect)
    {
        n_outer_dim /= 2; //nmrPipe convention, when both complex, FDSPECNUM = 2 * n_outer_dim; 
    }

    int n_direct_dimension = n_inner_dim;
    int n_indirect_dimension = n_outer_dim;

    if(nmrpipe_dict_float["FDTRANSPOSED"] == 1.0f) 
    {
        b_nmrPipe_transposed = true;
        std::swap(n_direct_dimension,n_indirect_dimension);
    }
    else
    {
        b_nmrPipe_transposed = false;
    }

    /**
     * Read remaining data from the file
     */
    if(b_frq==true){
        ndata_frq = n_direct_dimension;
        ndata = int(nmrpipe_dict_float["FDF2TDSIZE"]);
    }
    else{
        ndata = n_direct_dimension;
        ndata_frq = 0; //not used
    }
    n_center = int(nmrpipe_dict_float["FDF2CENTER"]);


    if(b_frq_indirect==true){
        ndata_frq_indirect = n_indirect_dimension;
        ndata_indirect = indirect_ndx == 1 ? int(nmrpipe_dict_float["FDF1TDSIZE"]) : int(nmrpipe_dict_float["FDF3TDSIZE"]);
    }
    else{
        ndata_indirect = n_indirect_dimension;
        ndata_frq_indirect = 0; //not used
    }
    n_center_indirect = indirect_ndx == 1 ? int(nmrpipe_dict_float["FDF1CENTER"]) : int(nmrpipe_dict_float["FDF3CENTER"]);


    
    if(b_imaginary){
        ndata_bruker = ndata * 2;
    }

    ndata_original = nmrpipe_dict_float["FDF2TDSIZE"];

    if(b_imaginary_indirect){
        ndata_bruker_indirect = ndata_indirect * 2;
    }

   
    /**
     * Need to update begin1,step1,stop1,begin2,step2,stop2 from nmrPipe dictionary
     * if not FTed data, we use ndata to replace ndata_frq.
     */
    stop1 = origin_frequency / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency;
    if(b_frq == false) step1 = (stop1 - begin1) / ndata; // direct dimension
    else step1 = (stop1 - begin1) / (ndata_frq); // direct dimension
    begin1 += step1;                        // here we have to + step because we want to be consistent with nmrpipe program,I have no idea why nmrpipe is different than topspin

    stop2 = origin_frequency_indirect / observed_frequency_indirect;
    begin2 = stop2 + spectral_width_indirect / observed_frequency_indirect;
    if (b_frq_indirect == false) step2 = (stop2 - begin2) / ndata_indirect; // indirect dimension
    else step2 = (stop2 - begin2) / (ndata_frq_indirect); // indirect dimension
    begin2 += step2;                                 // here we have to + step because we want to be consistent with nmrpipe program,I have no idea why nmrpipe is different than topspin

    return true;
}



/**
 * A low level function to run fft
 * @param n_dim1: size of first dimension.
 * @param dim1_flag: whether to run FFT for this trace. Size is n_dim1
 * @param n_dim2: size of second dimension. FFT is run along this dimension
 * @param n_dim2_frq: size of second dimension after FFT (because of zero filling)
 * @param in1: input data, real part [n_dim1][n_dim2] 
 * @param in2: input data, imaginary part [n_dim1][n_dim2]
 * @param out1: output data [n_dim1][n_dim2_frq] FFT result, real part
 * @param out2: output data [n_dim1][n_dim2_frq] FFT result, imaginary part
 * @param b_remove_filter: whether to remove Bruker digitizer filter
 * @param b_swap: whether to swap left and right halves of spectrum
 * 
 * Addtional information on b_swap
 * kiss_fft has different behavior than Bruker or nmrPipe FFT in a way that result needs to be swapped and flipped
 * So when b_swap == false, we swap left and right halves of spectrum.
 * When b_swap == true, we still do the swap, but we also apply alternate sign of input data (effectively a swap)
 */
bool fid_2d::fft_worker(int n_dim1, std::vector<int> dim1_flag, int n_dim2, int n_dim2_frq,
                        const float  *in1, const float  *in2,
                        float * out1, float * out2,
                        bool b_remove_filter,
                        bool b_swap,
                        double grpdly_) const
{
    /**
     * Loop over first dimension
     */
    for (int j = 0; j < n_dim1; j++)
    {
        if(dim1_flag[j]==0)
        {
            /**
             * Set output to 0.0
            */
           for (int i = 0; i < n_dim2_frq; i++)
            {
                out1[i + j * n_dim2_frq] = 0.0f;
                out2[i + j * n_dim2_frq] = 0.0f;
            }
            continue;
        }
        /**
         * Initialize cx_in and cx_out
         */
        kiss_fft_cpx *cx_in = new kiss_fft_cpx[n_dim2_frq];
        kiss_fft_cpx *cx_out = new kiss_fft_cpx[n_dim2_frq];

        /**
         * Copy data from in to cx_in
         */
        for (int i = 0; i < n_dim2; i++)
        {
            /**
             * If b_swap is true, we apply alternate sign to input data
            */
            if(b_swap == true && i%2==1) 
            {
                cx_in[i].r = -in1[i  + j * n_dim2 ];
                cx_in[i].i = -in2[i  + j * n_dim2];
            }
            else
            {
                cx_in[i].r = in1[i + j * n_dim2];
                cx_in[i].i = in2[i + j * n_dim2 ];
            }
        }

        /**
         * Zero filling. Input data has already been apodized
         */
        for (int i = n_dim2; i < n_dim2_frq; i++)
        {
            cx_in[i].r = 0.0;
            cx_in[i].i = 0.0;
        }

        /**
         * Run fft
         */
        if (kiss_fft_cfg cfg = kiss_fft_alloc(n_dim2_frq, 0, NULL, NULL))
        {
            kiss_fft(cfg, cx_in, cx_out);
            free(cfg);
        }
        else
        {
            std::cout << "Error: cannot allocate memory for kiss_fft_cfg" << std::endl;
            return false;
        }

        /**
         * Copy data from cx_out to spectrum_real and spectrum_imag
         */
        std::vector<float> spectrum_real(n_dim2_frq), spectrum_imag(n_dim2_frq);
        for (int i = 0; i < n_dim2_frq; i++)
        {
            spectrum_real[i] = cx_out[i].r / sqrt(float(n_dim2_frq));
            spectrum_imag[i] = cx_out[i].i / sqrt(float(n_dim2_frq)); // scale by sqrt(ndata_frq) to be consistent with standard fft
        }

        /**
         * Important: this step need to be done before fft result swap and flip
         */
        if (b_remove_filter)
        {
            remove_bruker_digitizer_filter(grpdly_, spectrum_real, spectrum_imag);
        }

        std::vector<float> spectrum_real_reoganized(n_dim2_frq, 0.0f);
        std::vector<float> spectrum_imag_reoganized(n_dim2_frq, 0.0f);

        /**
         * Swap left and right halves of spectrum. Aways do this because kiss_fft has different behavior than Bruker or nmrPipe FFT
         */
        
        for (int i = 1; i <= n_dim2_frq; i++)
        {
            spectrum_real_reoganized[i - 1] = (float)spectrum_real[(i + n_dim2_frq / 2) % n_dim2_frq];
            spectrum_imag_reoganized[i - 1] = (float)spectrum_imag[(i + n_dim2_frq / 2) % n_dim2_frq];
        }
       

        /**
         * Flip the spectrum then copy data to out1 and out2, because kiss_fft has different behavior than Bruker or nmrPipe FFT
         */
       
        for (int i = 0; i < n_dim2_frq; i++)
        {
            out1[i + j * n_dim2_frq] = spectrum_real_reoganized[n_dim2_frq - i - 1];
            out2[i + j * n_dim2_frq] = spectrum_imag_reoganized[n_dim2_frq - i - 1];
        }
        
    }
    return true;
}

/**
 * @brief fid_2d::phase_correction_worker: apply phase correction along dim2 for all rows in spectrum
 * @param n_dim1: size of first dimension.
 * @param n_dim2: size of second dimension. FFT is run along this dimension
 * @param spectrum_real: input/output data [n_dim1][n_dim2] real part
 * @param spectrum_imag: input/output data [n_dim1][n_dim2] imaginary part
 * @param p0: phase correction degree 0-th order
 * @param p1: phase correction degree 1-st order
*/
bool fid_2d::phase_correction_worker(int n_dim1,int n_dim2,float *spectrum_real, float *spectrum_imag, double p0, double p1) const
{
    /**
     * Apply phase correction along dim2.
     * Loop over dim1
    */
    for (int i = 0; i < n_dim1; i++)
    {
        for (int j = 0; j < n_dim2; j++)
        {
            float phase = p0 + p1 * j / n_dim2;
            float cos_phase = cos(phase*M_PI/180.0);
            float sin_phase = sin(phase*M_PI/180.0);

            float real = spectrum_real[j + i * n_dim2];
            float imag = spectrum_imag[j + i * n_dim2];

            spectrum_real[j + i * n_dim2] = real * cos_phase - imag * sin_phase;
            spectrum_imag[j + i * n_dim2] = real * sin_phase + imag * cos_phase;
        }
    }
    return true;
}

bool fid_2d::set_up_apodization(apodization *a1, apodization *a2)
{
    apodization_direct = a1;
    apodization_indirect = a2;
    return true;
};

bool fid_2d::set_up_apodization_from_string(const std::string &apodization_string_direct, const std::string &apodization_string_indirect)
{
    apodization_direct = new apodization(apodization_string_direct);
    apodization_indirect = new apodization(apodization_string_indirect);
    return true;
}

/**
 * @brief fid_2d::write_json: write some information
 * This is mainly for web-server
 */
bool fid_2d::write_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"] = ndata_power_of_2;
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"] = observed_frequency;
    root["spectral_width"] = spectral_width;

    /**
     * ft2 part. 0 if not exist
     */
    root["n_direct"] = ndata_frq;
    root["n_indirect"] = ndata_frq_indirect;
    root["noise_level"] = noise_level;
    root["begin_direct"] = begin1; 
    root["begin_indirect"]=begin2;
    root["step_direct"]=step1;
    root["step_indirect"]=step2;
    

    outfile << root << std::endl;
    outfile.close();

    return true;
}

/**
 * @brief fid_2d::write_json_as_string: write some information as a string
 * This is mainly for web-server and web assembly
 */
std::string fid_2d::write_json_as_string()
{
    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"] = ndata_power_of_2;
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"] = observed_frequency;
    root["spectral_width"] = spectral_width;

    /**
     * ft2 part. 0 if not exist
     */
    root["n_direct"] = ndata_frq;
    root["n_indirect"] = ndata_frq_indirect;
    root["noise_level"] = noise_level;
    root["begin_direct"] = begin1; 
    root["begin_indirect"]=begin2;
    root["step_direct"]=step1;
    root["step_indirect"]=step2;

    Json::StreamWriterBuilder writer;
    std::string output = Json::writeString(writer, root);
    
    return output;
}


bool fid_2d::write_pseudo3d_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["spectra"]=nspectra;

    outfile << root << std::endl;
    outfile.close();

    return true;
}

std::string fid_2d::write_pseudo3d_json_as_string()
{
    Json::Value root;
    root["spectra"] = nspectra;

    Json::StreamWriterBuilder writer;
    return Json::writeString(writer, root);
}

/**
 * @brief process 2D FID data to get 2D frequency domain spectra
 * @return true on success, false on failure
 */
bool fid_2d::full_process(bool b_di_direct,bool b_indirect_direct)
{
    run_direct_dimension(b_di_direct);
    transpose_intermediate_data();
    run_indirect_dimension(b_indirect_direct);
    untranspose_frquency_domain_data();
    return true;
}

/**
 * @brief direct dimension process only
*/
bool fid_2d::direct_only_process(bool b_di_direct)
{
    run_direct_dimension(b_di_direct);
    transpose_intermediate_data();
    return true;
}

/**
 * @brief indirect dimension process only, need to read in a half processed data
*/
bool fid_2d::indirect_only_process(bool b_indirect_direct)
{
    if (!(b_frq == true && b_frq_indirect == false))
    {
        std::cout << "Error: indirect_only_process requires INTERMEDIATE stage" << std::endl;
        return false;
    }

    /**
     * Assess that b_frq is true and b_nmrPipe_transposed is true
    */
    if(b_frq == false || b_nmrPipe_transposed == false)
    {
        std::cout<<" Require b_frq = true and b_nmrPipe_transposed = true"<<std::endl;
        std::cout << "Error: b_frq = " << b_frq << " and/or b_nmrPipe_transposed = " << b_nmrPipe_transposed << std::endl;
        return false;
    }
    run_indirect_dimension(b_indirect_direct);
    untranspose_frquency_domain_data();
    return true;
}

/**
 * Other process functions
 * Can run phase correction and remove imaginary part
*/
bool fid_2d::other_process(bool b_di_direct,  bool b_di_indirect)
{
    if (!(b_frq == true && b_frq_indirect == true))
    {
        std::cout << "Error: other_process requires FREQUENCY stage" << std::endl;
        return false;
    }

    /**
     * Apply phase correction along direct dimension only if b_imaginary is true
    */
    if(b_imaginary==true)
    {
        phase_correction_worker(ndata_frq_indirect, ndata_frq, spectrum_real_real.data(), spectrum_real_imag.data(), user_p0_direct /** P0 */,user_p1_direct /** P1*/);
        /**
         * If b_imaginary_indirct is also true, we need to apply phase correction for the imag part of indirect dimension too
         */
        if(b_imaginary_indirect==true)
        {
            phase_correction_worker(ndata_frq_indirect, ndata_frq, spectrum_imag_real.data(), spectrum_imag_imag.data(), user_p0_direct /** P0 */,user_p1_direct /** P1*/);
        }
    }

    
    /**
     * Apply phase correction along indirect dimension when b_imaginary_indirect is true
    */
    if(b_imaginary_indirect==true)
    {
        transpose_2d(spectrum_real_real.data(),ndata_frq,ndata_frq_indirect);
        transpose_2d(spectrum_imag_real.data(),ndata_frq,ndata_frq_indirect);
        phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_real.data(), spectrum_imag_real.data(), user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);
        transpose_2d(spectrum_real_real.data(),ndata_frq_indirect,ndata_frq);
        transpose_2d(spectrum_imag_real.data(),ndata_frq_indirect,ndata_frq);
    }

    if(b_di_direct == true)
    {
        /**
         * If b_di_direct is true, we set b_imaginary to false, and we do not need real_imag and imag_imag
        */
        b_imaginary = false;
        spectrum_real_imag.clear();
        spectrum_imag_imag.clear();
    }
    else
    {
        b_imaginary = true;
        if(b_imaginary_indirect==true)
        {  
            transpose_2d(spectrum_real_imag.data(),ndata_frq,ndata_frq_indirect);
            transpose_2d(spectrum_imag_imag.data(),ndata_frq,ndata_frq_indirect);
            phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_imag.data(), spectrum_imag_imag.data(), user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);
            transpose_2d(spectrum_real_imag.data(),ndata_frq_indirect,ndata_frq);
            transpose_2d(spectrum_imag_imag.data(),ndata_frq_indirect,ndata_frq);
        }
    }

    if(b_di_indirect == true)
    {
        /**
         * If b_imaginary_indirect is false, we do not need real_imag and imag_imag
        */
        b_imaginary_indirect = false;
        spectrum_real_imag.clear(); //might be already cleared, but no harm to clear again
        spectrum_imag_imag.clear();
    }
    
    
    return true;
}

/**
 * Remove huge water signal from the first trace (of the indirect dimension)
 * This is done after FFT along direct dimension
 * because I don't want to deal with Bruker digitazer filter 
 * (first 68 or 76 points are artifical and the following several points are crushed by the filter)
*/
bool fid_2d::water_suppression()
{
    if (!(b_frq == false && b_frq_indirect == false))
    {
        std::cout << "Error: water_suppression requires FID stage" << std::endl;
        return false;
    }

    if(spectrum_real_real.size()==0 || spectrum_real_imag.size()==0 || spectrum_imag_real.size()==0 || spectrum_imag_imag.size()==0)
    {
        std::cout<<"Error: fid_data is empty"<<std::endl;
        return false;
    }

    int grpdly_int = int(grpdly+0.5);
    float * ave_signal = new float[ndata-grpdly_int];

    for(int i=0;i<ndata_indirect;i++)
    {
        fid_2d_math::movemean(spectrum_real_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(spectrum_real_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);

        fid_2d_math::movemean(spectrum_real_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(spectrum_real_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);

        fid_2d_math::movemean(spectrum_imag_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(spectrum_imag_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);

        fid_2d_math::movemean(spectrum_imag_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(spectrum_imag_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);
    }


    return true;
};

/**
 * Run a polynomial baseline correction on the final spectrum (real real part only)
 * @param order: order of the polynomial (0,1 or 2)
*/
bool fid_2d::polynorminal_baseline(int order, bool b_trace_by_trace, double flatt_a, double flatt_b, int region_identification_flag)
{
    if (!(b_frq == true && b_frq_indirect == true))
    {
        std::cout << "Error: polynorminal_baseline requires FREQUENCY stage" << std::endl;
        return false;
    }

    if(spectrum_real_real.empty())
    {
        std::cout<<"Error: spectrum_real_real is empty"<<std::endl;
        return false;
    }

    // Estimate noise level if not already estimated
    if (noise_level <= 0.0)
    {
        spect = spectrum_real_real.data();
        estimate_noise_level();
    }
    if (noise_level <= 0.0)
    {
        noise_level = 0.1;
    }
    std::cout << "Using noise level: " << noise_level << " for peak exclusion." << std::endl;

    if (b_trace_by_trace)
    {
        double threshold = 4.0 * noise_level;
        int trace_len = n_inner_dim;

        for (int k = 0; k < nspectra; k++)
        {
            size_t spectrum_offset = static_cast<size_t>(k) * n_outer_dim * n_inner_dim;
            std::vector<char> full_selected_mask(n_outer_dim * n_inner_dim, 0);
            std::vector<float> full_baseline(n_outer_dim * n_inner_dim, 0.0f);
            std::vector<char> full_regions_map(n_outer_dim * n_inner_dim, 0);

            if (order == -1 && (region_identification_flag == 1 || region_identification_flag == 5))
            {
                // Method 1 or 5: 2D region identification
                std::vector<bool> full_pos_peaks(n_outer_dim * n_inner_dim, false);
                std::vector<bool> full_neg_peaks(n_outer_dim * n_inner_dim, false);
                std::vector<bool> full_water_regions(n_outer_dim * n_inner_dim, false);

                if (region_identification_flag == 1)
                {
                    // Run row-wise (F2) region identification
                    std::vector<bool> row_pos(n_inner_dim, false);
                    std::vector<bool> row_neg(n_inner_dim, false);
                    std::vector<bool> row_water(n_inner_dim, false);

                    std::vector<bool> f2_pos(n_outer_dim * n_inner_dim, false);
                    std::vector<bool> f2_neg(n_outer_dim * n_inner_dim, false);
                    std::vector<bool> f2_water(n_outer_dim * n_inner_dim, false);

                    for (int i = 0; i < n_outer_dim; i++)
                    {
                        const float *trace_data = spectrum_real_real.data() + spectrum_offset + i * n_inner_dim;
                        std::vector<int> row_region_mask;
                        fid_2d_math::flatt_identify_regions_1d(trace_data, n_inner_dim, row_pos, row_neg, row_water, noise_level, 17, -1.0, &row_region_mask);
                        for (int j = 0; j < n_inner_dim; j++)
                        {
                            f2_pos[i * n_inner_dim + j] = row_pos[j];
                            f2_neg[i * n_inner_dim + j] = row_neg[j];
                            f2_water[i * n_inner_dim + j] = row_water[j];
                            full_selected_mask[i * n_inner_dim + j] = row_region_mask[j];
                        }
                    }

                    // Run column-wise (F1) region identification
                    std::vector<bool> col_pos(n_outer_dim, false);
                    std::vector<bool> col_neg(n_outer_dim, false);
                    std::vector<bool> col_water(n_outer_dim, false);

                    std::vector<bool> f1_pos(n_outer_dim * n_inner_dim, false);
                    std::vector<bool> f1_neg(n_outer_dim * n_inner_dim, false);

                    std::vector<float> col_data(n_outer_dim);
                    for (int j = 0; j < n_inner_dim; j++)
                    {
                        for (int i = 0; i < n_outer_dim; i++)
                        {
                            col_data[i] = spectrum_real_real[spectrum_offset + i * n_inner_dim + j];
                        }
                        fid_2d_math::flatt_identify_regions_1d(col_data.data(), n_outer_dim, col_pos, col_neg, col_water, noise_level, 17, -1.0);
                        for (int i = 0; i < n_outer_dim; i++)
                        {
                            f1_pos[i * n_inner_dim + j] = col_pos[i];
                            f1_neg[i * n_inner_dim + j] = col_neg[i];
                        }
                    }

                    // Consensus
                    for (int idx = 0; idx < n_outer_dim * n_inner_dim; idx++)
                    {
                        full_pos_peaks[idx] = f2_pos[idx] && f1_pos[idx];
                        full_neg_peaks[idx] = f2_neg[idx] && f1_neg[idx];
                        full_water_regions[idx] = f2_water[idx]; // water region has no requirement along indirect dimension
                    }

                    // 2D Dilation on full_neg_peaks (structuring element: ellipsoid Rx=16, Ry=4)
                    std::vector<bool> dilated_neg_peaks = full_neg_peaks;
                    for (int y = 0; y < n_outer_dim; y++)
                    {
                        for (int x = 0; x < n_inner_dim; x++)
                        {
                            if (full_neg_peaks[y * n_inner_dim + x])
                            {
                                for (int dy = -4; dy <= 4; dy++)
                                {
                                    for (int dx = -16; dx <= 16; dx++)
                                    {
                                        if ((dx * dx) / 256.0 + (dy * dy) / 16.0 <= 1.0)
                                        {
                                            int ny = y + dy;
                                            int nx = x + dx;
                                            if (ny >= 0 && ny < n_outer_dim && nx >= 0 && nx < n_inner_dim)
                                            {
                                                if (!full_pos_peaks[ny * n_inner_dim + nx] && !full_water_regions[ny * n_inner_dim + nx])
                                                {
                                                    dilated_neg_peaks[ny * n_inner_dim + nx] = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    full_neg_peaks = dilated_neg_peaks;
                }
                else if (region_identification_flag == 5)
                {
                    // Run row-wise (F2) region identification (F2 only)
                    std::vector<bool> row_pos(n_inner_dim, false);
                    std::vector<bool> row_neg(n_inner_dim, false);
                    std::vector<bool> row_water(n_inner_dim, false);

                    for (int i = 0; i < n_outer_dim; i++)
                    {
                        const float *trace_data = spectrum_real_real.data() + spectrum_offset + i * n_inner_dim;
                        std::vector<int> row_region_mask;
                        fid_2d_math::flatt_identify_regions_1d(trace_data, n_inner_dim, row_pos, row_neg, row_water, noise_level, 17, -1.0, &row_region_mask);
                        for (int j = 0; j < n_inner_dim; j++)
                        {
                            full_pos_peaks[i * n_inner_dim + j] = row_pos[j];
                            full_neg_peaks[i * n_inner_dim + j] = row_neg[j];
                            full_water_regions[i * n_inner_dim + j] = row_water[j];
                            full_selected_mask[i * n_inner_dim + j] = row_region_mask[j];
                        }
                    }

                    // 2D Dilation on full_neg_peaks (structuring element: ellipsoid Rx=16, Ry=4)
                    std::vector<bool> dilated_neg_peaks = full_neg_peaks;
                    for (int y = 0; y < n_outer_dim; y++)
                    {
                        for (int x = 0; x < n_inner_dim; x++)
                        {
                            if (full_neg_peaks[y * n_inner_dim + x])
                            {
                                for (int dy = -4; dy <= 4; dy++)
                                {
                                    for (int dx = -16; dx <= 16; dx++)
                                    {
                                        if ((dx * dx) / 256.0 + (dy * dy) / 16.0 <= 1.0)
                                        {
                                            int ny = y + dy;
                                            int nx = x + dx;
                                            if (ny >= 0 && ny < n_outer_dim && nx >= 0 && nx < n_inner_dim)
                                            {
                                                if (!full_pos_peaks[ny * n_inner_dim + nx] && !full_water_regions[ny * n_inner_dim + nx])
                                                {
                                                    dilated_neg_peaks[ny * n_inner_dim + nx] = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    full_neg_peaks = dilated_neg_peaks;

                    // 2D Opening on full_pos_peaks and full_neg_peaks
                    full_pos_peaks = fid_2d_math::opening_3x3_cross(full_pos_peaks, n_outer_dim, n_inner_dim);
                    full_neg_peaks = fid_2d_math::opening_3x3_cross(full_neg_peaks, n_outer_dim, n_inner_dim);

                    // 2D Closing on full_water_regions
                    full_water_regions = fid_2d_math::closing_3x3_cross(full_water_regions, n_outer_dim, n_inner_dim);
                }

                // Populate full_regions_map (full_selected_mask contains raw slopes)
                for (int idx = 0; idx < n_outer_dim * n_inner_dim; idx++)
                {
                    if (full_pos_peaks[idx]) full_regions_map[idx] = 1;
                    else if (full_neg_peaks[idx]) full_regions_map[idx] = 2;
                    else if (full_water_regions[idx]) full_regions_map[idx] = 3;
                    else full_regions_map[idx] = 0;
                }

                // Solve baseline trace-by-trace and subtract
                for (int i = 0; i < n_outer_dim; i++)
                {
                    float *trace_data = spectrum_real_real.data() + spectrum_offset + i * n_inner_dim;
                    std::vector<bool> trace_pos(n_inner_dim);
                    std::vector<bool> trace_neg(n_inner_dim);
                    std::vector<bool> trace_water(n_inner_dim);
                    for (int j = 0; j < n_inner_dim; j++)
                    {
                        trace_pos[j] = full_pos_peaks[i * n_inner_dim + j];
                        trace_neg[j] = full_neg_peaks[i * n_inner_dim + j];
                        trace_water[j] = full_water_regions[i * n_inner_dim + j];
                    }

                    std::vector<float> trace_baseline;
                    fid_2d_math::flatt_solver_1d(trace_data, n_inner_dim, trace_baseline, trace_pos, trace_neg, trace_water, noise_level, flatt_a, flatt_b);

                    for (int j = 0; j < n_inner_dim; j++)
                    {
                        full_baseline[i * n_inner_dim + j] = trace_baseline[j];
                        trace_data[j] -= trace_baseline[j];
                    }
                }
            }
            else
            {
                // Original trace-by-trace polynomial or Method 0 FLATT baseline logic
                for (int i = 0; i < n_outer_dim; i++)
                {
                    float *trace_data = spectrum_real_real.data() + spectrum_offset + i * n_inner_dim;
                    std::vector<float> trace_baseline;

                    if (order == -1)
                    {
                        std::vector<bool> trace_pos(trace_len, false);
                        std::vector<bool> trace_neg(trace_len, false);
                        std::vector<bool> trace_water(trace_len, false);
                        std::vector<int> row_region_mask;
                        fid_2d_math::flatt_identify_regions_1d(trace_data, trace_len, trace_pos, trace_neg, trace_water, noise_level, 17, -1.0, &row_region_mask);
                        fid_2d_math::expand_neg_peaks_1d(trace_pos, trace_neg, trace_water, trace_len, 16);
                        
                        for (int j = 0; j < trace_len; j++)
                        {
                            if (trace_pos[j]) full_regions_map[i * trace_len + j] = 1;
                            else if (trace_neg[j]) full_regions_map[i * trace_len + j] = 2;
                            else if (trace_water[j]) full_regions_map[i * trace_len + j] = 3;
                            else full_regions_map[i * trace_len + j] = 0;

                            full_selected_mask[i * trace_len + j] = row_region_mask[j];
                        }

                        fid_2d_math::flatt_solver_1d(trace_data, trace_len, trace_baseline, trace_pos, trace_neg, trace_water, noise_level, flatt_a, flatt_b);
                    }
                    else
                    {
                        std::vector<char> trace_mask;
                        fid_2d_math::polynomial_baseline_1d(trace_data, trace_len, order, trace_baseline, threshold, 32, &trace_mask);

                        // Copy to full mask and regions map
                        for (int j = 0; j < trace_len; j++)
                        {
                            full_selected_mask[i * trace_len + j] = trace_mask[j];
                            full_regions_map[i * trace_len + j] = (trace_mask[j] == 0) ? 1 : 0;
                        }
                    }

                    // Copy to full baseline
                    for (int j = 0; j < trace_len; j++)
                    {
                        full_baseline[i * trace_len + j] = trace_baseline[j];
                    }

                    // Subtract baseline
                    for (int j = 0; j < trace_len; j++)
                    {
                        trace_data[j] -= trace_baseline[j];
                    }
                }
            }

#ifdef DEBUG
            {
                std::string regions_filename = "debug_regions_map.txt";
                std::string map_filename = "debug_selected_map.txt";
                std::string base_filename = "debug_baseline.txt";
                if (nspectra > 1)
                {
                    regions_filename = "debug_regions_map_" + std::to_string(k) + ".txt";
                    map_filename = "debug_selected_map_" + std::to_string(k) + ".txt";
                    base_filename = "debug_baseline_" + std::to_string(k) + ".txt";
                }
                std::ofstream out_regions(regions_filename);
                std::ofstream out_map(map_filename);
                std::ofstream out_base(base_filename);
                if (out_regions.good() && out_map.good() && out_base.good())
                {
                    for (int i = 0; i < n_outer_dim; i++)
                    {
                        for (int j = 0; j < n_inner_dim; j++)
                        {
                            int region_val = full_regions_map[i * n_inner_dim + j];
                            out_regions << region_val << (j == n_inner_dim - 1 ? "" : " ");

                            int is_selected = full_selected_mask[i * n_inner_dim + j];
                            out_map << is_selected << (j == n_inner_dim - 1 ? "" : " ");

                            float baseline_val = full_baseline[i * n_inner_dim + j];
                            out_base << baseline_val << (j == n_inner_dim - 1 ? "" : " ");
                        }
                        out_regions << "\n";
                        out_map << "\n";
                        out_base << "\n";
                    }
                }
            }
#endif
        }
        if (order == -1)
        {
            std::cout << "Baseline correction done using FLATT workflow with region flag " << region_identification_flag << std::endl;
        }
        else
        {
            std::cout << "Baseline correction done using 1D trace-by-trace polynorminal at order " << order << std::endl;
        }
        return true;
    }
    else
    {
        int n_dim1 = 4;
        int n_dim2 = 64;

        if (n_dim1 > n_outer_dim) n_dim1 = n_outer_dim;
        if (n_dim2 > n_inner_dim) n_dim2 = n_inner_dim;

        int step1 = n_outer_dim / n_dim1;
        if (step1 < 1) step1 = 1;
        int step2 = n_inner_dim / n_dim2;
        if (step2 < 1) step2 = 1;

        n_dim1 = n_outer_dim / step1;
        n_dim2 = n_inner_dim / step2;

        std::vector<float> x(n_dim1*n_dim2,0.0f);
        std::vector<float> y(n_dim1*n_dim2,0.0f);
        std::vector<float> z(n_dim1*n_dim2,0.0f);

        for(int i=0;i<n_dim1;i++)
        {
            for(int j=0;j<n_dim2;j++)
            {
                x[i*n_dim2+j] = j * step2;
                y[i*n_dim2+j] = i * step1;
            }
        }

        for (int k = 0; k < nspectra; k++)
        {
            size_t spectrum_offset = static_cast<size_t>(k) * n_outer_dim * n_inner_dim;

            for(int i=0;i<n_dim1;i++)
            {
                for(int j=0;j<n_dim2;j++)
                {
                    z[i*n_dim2+j] = spectrum_real_real[spectrum_offset + i*step1*n_inner_dim + j*step2];
                }
            }

            std::vector<float> baseline_params;
            std::vector<char> selected_mask;
            fid_2d_math::polynomial_baseline(x,y,z,n_dim1,n_dim2,order,baseline_params, noise_level, &selected_mask);

#ifdef DEBUG
            {
                std::string regions_filename = "debug_regions_map.txt";
                std::string map_filename = "debug_selected_map.txt";
                std::string base_filename = "debug_baseline.txt";
                if (nspectra > 1)
                {
                    regions_filename = "debug_regions_map_" + std::to_string(k) + ".txt";
                    map_filename = "debug_selected_map_" + std::to_string(k) + ".txt";
                    base_filename = "debug_baseline_" + std::to_string(k) + ".txt";
                }
                std::ofstream out_regions(regions_filename);
                std::ofstream out_map(map_filename);
                std::ofstream out_base(base_filename);
                if (out_regions.good() && out_map.good() && out_base.good())
                {
                    for (int i = 0; i < n_outer_dim; i++)
                    {
                        int di = i / step1;
                        if (di >= n_dim1) di = n_dim1 - 1;
                        for (int j = 0; j < n_inner_dim; j++)
                        {
                            int dj = j / step2;
                            if (dj >= n_dim2) dj = n_dim2 - 1;

                            int val_idx = di * n_dim2 + dj;
                            int is_selected = (val_idx >= 0 && val_idx < static_cast<int>(selected_mask.size())) ? selected_mask[val_idx] : 0;
                            int region_val = (is_selected == 0) ? 1 : 0;
                            out_regions << region_val << (j == n_inner_dim - 1 ? "" : " ");
                            out_map << is_selected << (j == n_inner_dim - 1 ? "" : " ");

                            float baseline_val = baseline_params[0];
                            if (order >= 1)
                            {
                                baseline_val += baseline_params[1] * j;
                                baseline_val += baseline_params[2] * i;
                            }
                            if (order >= 2)
                            {
                                baseline_val += baseline_params[3] * j * j;
                                baseline_val += baseline_params[4] * i * i;
                            }
                            if (order >= 3)
                            {
                                baseline_val += baseline_params[5] * j * j * j;
                                baseline_val += baseline_params[6] * i * i * i;
                            }
                            if (order >= 4)
                            {
                                baseline_val += baseline_params[7] * j * j * j * j;
                                baseline_val += baseline_params[8] * i * i * i * i;
                            }
                            out_base << baseline_val << (j == n_inner_dim - 1 ? "" : " ");
                        }
                        out_regions << "\n";
                        out_map << "\n";
                        out_base << "\n";
                    }
                }
            }
#endif

            /**
             * Substract baseline from the spectrum
            */
            for(int i=0;i<n_outer_dim;i++)
            {
                for(int j=0;j<n_inner_dim;j++)
                {
                    float baseline = baseline_params[0]; //constant term
                    if(order>=1) 
                    {
                        baseline += baseline_params[1]*j; //linear term along dim2(inner dimension)
                        baseline += baseline_params[2]*i; //linear term along dim1(outer dimension)
                    }
                    if(order>=2)
                    {
                        baseline += baseline_params[3]*j*j; //quadratic term along dim2(inner dimension)
                        baseline += baseline_params[4]*i*i; //quadratic term along dim1(outer dimension)
                        /** No cross term in NMR */
                    }
                    if(order>=3)
                    {
                        baseline += baseline_params[5]*j*j*j; //cubic term along dim2(inner dimension)
                        baseline += baseline_params[6]*i*i*i; //cubic term along dim1(outer dimension)
                        /** No cross term in NMR */
                    }
                    if(order>=4)
                    {
                        baseline += baseline_params[7]*j*j*j*j; //quartic term along dim2(inner dimension)
                        baseline += baseline_params[8]*i*i*i*i; //quartic term along dim1(outer dimension)
                        /** No cross term in NMR */
                    }
                    spectrum_real_real[spectrum_offset + i*n_inner_dim + j] -= baseline;
                }
            }
        }
        std::cout<<"Baseline correction done using polynorminal at order "<<order<<std::endl;

        return true;
    }
}


bool fid_2d::run_direct_dimension(bool b_di_direct)
{
    if (!(b_frq == false && b_frq_indirect == false))
    {
        std::cerr << "Error: run_direct_dimension requires FID stage." << std::endl;
        return false;
    }

    /**
     * get ndata_power_of_2, which is the smallest power of 2
     * that is larger than or equal ndata
     */
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }
    ndata_frq = ndata_power_of_2 * zf;

    n_center = int(ndata_frq / 2.0f) + 1;
    origin_frequency = (carrier_frequency - spectral_width * (ndata_frq - n_center) / ndata_frq);


    std::vector<float> out_real_real(nspectra * ndata_frq * ndata_indirect, 0.0f);
    std::vector<float> out_real_imag(nspectra * ndata_frq * ndata_indirect, 0.0f);
    std::vector<float> out_imag_real(nspectra * ndata_frq * ndata_indirect, 0.0f);
    std::vector<float> out_imag_imag(nspectra * ndata_frq * ndata_indirect, 0.0f);


   
    for(int k=0;k<nspectra;k++)
    {
        int fid_start = k * ndata * ndata_indirect;
        int intermediate_start = k * ndata_frq * ndata_indirect;

        /**
         * Apodization along direct dimension
         */
        apodization_direct->set_sw(spectral_width);
        apodization_direct->set_n(ndata_original);
        apodization_direct->set_first_point(0.5);
        for(int j=0;j<ndata_indirect;j++)
        {
            /**
             * Skip non-sampled data
            */
            if(nusflags[j]==0) continue;

            apodization_direct->run_apodization(spectrum_real_real.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
            apodization_direct->run_apodization(spectrum_real_imag.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
            apodization_direct->run_apodization(spectrum_imag_real.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
            apodization_direct->run_apodization(spectrum_imag_imag.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
        }
        apodization_code = 1;

        std::vector<int> single_trace_flag(1, 1);
        for (int j = 0; j < ndata_indirect; j++)
        {
            if (nusflags[j] == 0)
            {
                continue;
            }

            const int fid_trace_start = fid_start + j * ndata;
            const int intermediate_trace_start = intermediate_start + j * ndata_frq;

            /**
             * FFT and phase correction along direct dimension, trace-by-trace.
             */
            fft_worker(
                1, single_trace_flag, ndata, ndata_frq,
                spectrum_real_real.data() + fid_trace_start, spectrum_real_imag.data() + fid_trace_start,
                out_real_real.data() + intermediate_trace_start, out_real_imag.data() + intermediate_trace_start,
                true /** digital fileter*/, false /** swap*/, grpdly
            );
            fft_worker(
                1, single_trace_flag, ndata, ndata_frq,
                spectrum_imag_real.data() + fid_trace_start, spectrum_imag_imag.data() + fid_trace_start,
                out_imag_real.data() + intermediate_trace_start, out_imag_imag.data() + intermediate_trace_start,
                true /** digital fileter*/, false /** swap*/, grpdly
            );
            phase_correction_worker(
                1, ndata_frq,
                out_real_real.data() + intermediate_trace_start,
                out_real_imag.data() + intermediate_trace_start,
                user_p0_direct /** P0 */, user_p1_direct /** P1 */
            );
            phase_correction_worker(
                1, ndata_frq,
                out_imag_real.data() + intermediate_trace_start,
                out_imag_imag.data() + intermediate_trace_start,
                user_p0_direct /** P0 */, user_p1_direct /** P1 */
            );
        }
    }

    spectrum_real_real.swap(out_real_real);
    spectrum_real_imag.swap(out_real_imag);
    spectrum_imag_real.swap(out_imag_real);
    spectrum_imag_imag.swap(out_imag_imag);

    b_nmrPipe_transposed = false;
    b_frq = true;
    b_frq_indirect = false;
    spect = spectrum_real_real.data();

    if(b_di_direct == true)
    {
        /**
         * If b_imaginary is false, we do not need real_imag and imag_imag
        */
        b_imaginary = false;
        spectrum_real_imag.clear();
        spectrum_imag_imag.clear();
    }

    n_outer_dim = ndata_indirect;
    n_inner_dim = ndata_frq;

    /**
     * Run direct dimension extraction here
     * [extraction_from, extraction_to] is the range of data to be extracted.
     *  [0,1] is the full range
    */
    if (extraction_use_ppm)
    {
        const double stop_ppm = origin_frequency / observed_frequency;
        double begin_ppm = stop_ppm + spectral_width / observed_frequency;
        const double step_ppm = (stop_ppm - begin_ppm) / ndata_frq;
        begin_ppm += step_ppm;

        const int pos1 = int((extraction_from_ppm - begin_ppm) / step_ppm + 0.5);
        const int pos2 = int((extraction_to_ppm - begin_ppm) / step_ppm + 0.5);
        extraction_from_pos = std::min(pos1, pos2);
        extraction_to_pos = std::max(pos1, pos2);
    }
    else
    {
        extraction_from_pos = int(ndata_frq * extraction_from + 0.5);
        extraction_to_pos = int(ndata_frq * extraction_to + 0.5);
    }

    if(extraction_from_pos<0) extraction_from_pos = 0;
    if(extraction_to_pos>ndata_frq) extraction_to_pos = ndata_frq;

    if (((extraction_to_pos - extraction_from_pos) % 2) != 0)
    {
        if (extraction_to_pos < ndata_frq)
        {
            extraction_to_pos += 1;
        }
        else if (extraction_from_pos > 0)
        {
            // Keep extraction size even when upper boundary cannot move further.
            extraction_from_pos -= 1;
        }
    }

    const double stop_ppm = origin_frequency / observed_frequency;
    double begin_ppm = stop_ppm + spectral_width / observed_frequency;
    const double step_ppm = (stop_ppm - begin_ppm) / ndata_frq;
    begin_ppm += step_ppm;

    const double extracted_from_ppm = begin_ppm + step_ppm * extraction_from_pos;
    const double extracted_to_ppm = begin_ppm + step_ppm * extraction_to_pos;

    if(extraction_from_pos>=extraction_to_pos)
    {
        std::cout << "Error: invalid extraction range. index [" << extraction_from_pos << ", " << extraction_to_pos
                  << "), ppm [" << extracted_from_ppm << ", " << extracted_to_ppm << "). Skip extraction." << std::endl;
    }
    else
    {
        std::cout << "Extraction direct dimension: index [" << extraction_from_pos << ", " << extraction_to_pos
                  << "), ppm [" << extracted_from_ppm << ", " << extracted_to_ppm << ")" << std::endl;
        int n_inner_dim_new = extraction_to_pos - extraction_from_pos;
        
        /**
         * Note: we can copy in place, because new data is always smaller than original data
        */
        for(int k=0;k<nspectra;k++)
        {
            int spectrum_start = k * n_inner_dim * n_outer_dim; 
            int spectrum_start_new = k * n_inner_dim_new * n_outer_dim;
            for(int i=0;i<n_outer_dim;i++)
            {
                int line_start = i * n_inner_dim + spectrum_start;
                int line_start_new = i * n_inner_dim_new + spectrum_start_new;
                /**
                 * Copy data from line_start to line_start_new, for a total of n_inner_dim_new
                */
                for(int j=0;j<n_inner_dim_new;j++)
                {
                    spectrum_real_real[line_start_new+j] = spectrum_real_real[line_start+extraction_from_pos+j];
                    spectrum_imag_real[line_start_new+j] = spectrum_imag_real[line_start+extraction_from_pos+j];
                    if(b_imaginary == true)
                    {
                        spectrum_real_imag[line_start_new+j] = spectrum_real_imag[line_start+extraction_from_pos+j];
                        spectrum_imag_imag[line_start_new+j] = spectrum_imag_imag[line_start+extraction_from_pos+j];
                    }
                }
            }
        }
        /**
         * Resize spectrum_real_real, spectrum_imag_real, spectrum_real_imag, spectrum_imag_imag
        */
        spectrum_real_real.resize(nspectra * n_inner_dim_new * n_outer_dim);
        spectrum_imag_real.resize(nspectra * n_inner_dim_new * n_outer_dim);
        if(b_imaginary == true)
        {
            spectrum_real_imag.resize(nspectra * n_inner_dim_new * n_outer_dim);
            spectrum_imag_imag.resize(nspectra * n_inner_dim_new * n_outer_dim);
        }

        origin_frequency = carrier_frequency - spectral_width  / n_inner_dim * (extraction_to_pos - n_center);
        n_center -= extraction_from_pos;
        spectral_width = spectral_width * ((double)n_inner_dim_new / n_inner_dim);
        n_inner_dim = n_inner_dim_new;
    }

    return true;
}

bool fid_2d::transpose_intermediate_data()
{
    if (!(b_frq == true && b_frq_indirect == false))
    {
        std::cerr << "Error: transpose_intermediate_data requires INTERMEDIATE stage." << std::endl;
        return false;
    }

    std::vector<float> &active_real_real = spectrum_real_real;
    std::vector<float> &active_imag_real = spectrum_imag_real;
    std::vector<float> &active_real_imag = spectrum_real_imag;
    std::vector<float> &active_imag_imag = spectrum_imag_imag;

    for(int k=0;k<nspectra;k++)
    {
        int intermediate_start = k * n_inner_dim * n_outer_dim;
        transpose_2d(active_real_real.data()+intermediate_start, n_outer_dim, n_inner_dim);
        transpose_2d(active_imag_real.data()+intermediate_start, n_outer_dim, n_inner_dim);
        
        if(b_imaginary ==true)
        {
            transpose_2d(active_real_imag.data()+intermediate_start, n_outer_dim, n_inner_dim);
            transpose_2d(active_imag_imag.data()+intermediate_start, n_outer_dim, n_inner_dim);
        }
    }

    b_nmrPipe_transposed = true;

    std::swap(n_outer_dim,n_inner_dim);
    return true;
}

bool fid_2d::run_indirect_dimension(bool b_di_indirect)
{
    if (!(b_frq == true && b_frq_indirect == false))
    {
        std::cerr << "Error: run_indirect_dimension requires INTERMEDIATE stage." << std::endl;
        return false;
    }

    std::vector<float> &active_real_real = spectrum_real_real;
    std::vector<float> &active_real_imag = spectrum_real_imag;
    std::vector<float> &active_imag_real = spectrum_imag_real;
    std::vector<float> &active_imag_imag = spectrum_imag_imag;

     /**
     * Do the same for indirect dimension
     */
    ndata_power_of_2_indirect = 1;
    while (ndata_power_of_2_indirect < ndata_indirect)
    {
        ndata_power_of_2_indirect *= 2;
    }
    ndata_frq_indirect = ndata_power_of_2_indirect * zf_indirect;
    n_center_indirect = int(ndata_frq_indirect / 2.0f) + 1;
    origin_frequency_indirect = (carrier_frequency_indirect - spectral_width_indirect * (ndata_frq_indirect - n_center_indirect) / ndata_frq_indirect);


     /**
     * clear frq domain spectrum then allocate memory
     */
    std::vector<float> out_real_real(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
    std::vector<float> out_imag_real(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
    std::vector<float> out_real_imag;
    std::vector<float> out_imag_imag;
    
    if(b_imaginary ==true)
    {
        out_real_imag.resize(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
        out_imag_imag.resize(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
    }



    for(int k=0;k<nspectra;k++)
    {
        int intermediate_start = k * ndata_frq * ndata_indirect;
        int spectrum_start = k * ndata_frq * ndata_frq_indirect; 
   
        /**
         * size of intermediate_spectrum is [ndata_frq][ndata_bruker_indirect]
        */
        apodization_indirect->set_sw(spectral_width_indirect);
        apodization_indirect->set_n(ndata_indirect);
        /**
         * For indirect dimension, if user_p1_indirect is around 0 (unit is degree), we set first point to 0.5
         * otherwise, we set first point to 1.0
         * IMPORTANT: This means we can finalize phased spectrum using phased spectrum from program phase_2d, if user_p1_indirect is around +-180 degree
         * we have to rerun this program (fid_2d) with phase information.
        */
        if(fabs(user_p1_indirect)<2.0)
        {
            apodization_indirect->set_first_point(0.5);
        }
        else
        {
            apodization_indirect->set_first_point(1.0);
        }
        for(int i=0;i<ndata_frq;i++)
        {
            apodization_indirect->run_apodization(active_real_real.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
            apodization_indirect->run_apodization(active_imag_real.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
            if(b_imaginary ==true)
            {
                apodization_indirect->run_apodization(active_real_imag.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
                apodization_indirect->run_apodization(active_imag_imag.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
            }
        }


        /**
         * FFT along indirect dimension.
         * fnmode == 6 means echo anti-echo pulse sequence
        */
        bool b_swap = false;
        if(fnmode==6)
        {
            b_swap = false;
        }
        else if(fnmode==5) //state-tppi. Need to swap
        {
            b_swap = true;
        }
        /**
         * TODO: other cases
        */

        /**
         * If b_negative == true. Change sign of imaginary part of intermediate_spectrum and intermediate_spectrum_imag
         * For both intermediate_spectrum and intermediate_spectrum_imag, real part is at even index (0,2,4), imaginary part is at odd index (1,3, 5)
         */
        if (b_negative)
        {

            for (int i = 0; i < ndata_frq * ndata_indirect; i++)
            {
                active_imag_real[intermediate_start + i] = -active_imag_real[intermediate_start + i];
                if(b_imaginary ==true)
                {
                    active_imag_imag[intermediate_start + i] = -active_imag_imag[intermediate_start + i];
                }
            }
        }
        /**
         * FFT along indirect dimension. there is no NUS flag for indirect dimension
        */
        std::vector<int> single_trace_flag(1, 1);
        for (int i = 0; i < ndata_frq; i++)
        {
            const int intermediate_trace_start = intermediate_start + i * ndata_indirect;
            const int spectrum_trace_start = spectrum_start + i * ndata_frq_indirect;

            fft_worker(
                1, single_trace_flag, ndata_indirect, ndata_frq_indirect,
                active_real_real.data() + intermediate_trace_start,
                active_imag_real.data() + intermediate_trace_start,
                out_real_real.data() + spectrum_trace_start, out_imag_real.data() + spectrum_trace_start,
                false /** remove filter*/, b_swap /** swap*/, -1 /**grpdly is not applicable*/
            );
            phase_correction_worker(
                1, ndata_frq_indirect,
                out_real_real.data() + spectrum_trace_start,
                out_imag_real.data() + spectrum_trace_start,
                user_p0_indirect /** P0 */, user_p1_indirect /** P1 */
            );

            if (b_imaginary == true)
            {
                fft_worker(
                    1, single_trace_flag, ndata_indirect, ndata_frq_indirect,
                    active_real_imag.data() + intermediate_trace_start,
                    active_imag_imag.data() + intermediate_trace_start,
                    out_real_imag.data() + spectrum_trace_start, out_imag_imag.data() + spectrum_trace_start,
                    false /** remove filter*/, b_swap /** swap*/, -1 /**grpdly is not applicable*/
                );
                phase_correction_worker(
                    1, ndata_frq_indirect,
                    out_real_imag.data() + spectrum_trace_start,
                    out_imag_imag.data() + spectrum_trace_start,
                    user_p0_indirect /** P0 */, user_p1_indirect /** P1 */
                );
            }
        }

    }

    spectrum_real_real.swap(out_real_real);
    spectrum_imag_real.swap(out_imag_real);
    if (b_imaginary == true)
    {
        spectrum_real_imag.swap(out_real_imag);
        spectrum_imag_imag.swap(out_imag_imag);
    }

    b_frq_indirect = true;
    spect = spectrum_real_real.data();

    apodization_code_indirect = 1;

    if(b_di_indirect == true)
    {
        /**
         * If b_imaginary_indirect is false, we do not need imag_real and imag_imag
        */
        b_imaginary_indirect = false;
        spectrum_imag_real.clear();
        spectrum_imag_imag.clear();
    }
    
    
    /**
     * Apply receiver_gain if > 1.0
    */
    if (receiver_gain > 1.0)
    {
        for (int i = 0; i < spectrum_real_real.size(); i++)
        {
            spectrum_real_real[i] *= receiver_gain;
            if(b_imaginary_indirect ==true)
            {
                spectrum_imag_real[i] *= receiver_gain;
            }
            if(b_imaginary ==true)
            {
                spectrum_real_imag[i] *= receiver_gain;
                if(b_imaginary_indirect ==true)
                {
                    spectrum_imag_imag[i] *= receiver_gain;
                }
            }
        }
    }
    b_nmrPipe_transposed = true;
    n_inner_dim = ndata_frq_indirect;

    return true;
}

bool fid_2d::untranspose_frquency_domain_data()
{
    if (!(b_frq == true && b_frq_indirect == true))
    {
        std::cerr << "Error: untranspose_frquency_domain_data requires FREQUENCY stage." << std::endl;
        return false;
    }

    for(int k=0;k<nspectra;k++)
    {
        int spectrum_start = k * n_outer_dim * n_inner_dim; 
        transpose_2d(spectrum_real_real.data()+spectrum_start, n_outer_dim, n_inner_dim);
        if(b_imaginary_indirect ==true)
        {
            transpose_2d(spectrum_imag_real.data()+spectrum_start, n_outer_dim, n_inner_dim);
        }

        if(b_imaginary==true)
        {
            transpose_2d(spectrum_real_imag.data()+spectrum_start, n_outer_dim, n_inner_dim);
            if(b_imaginary_indirect ==true)
            {
                transpose_2d(spectrum_imag_imag.data()+spectrum_start, n_outer_dim, n_inner_dim);
            }
        }
    }
    b_nmrPipe_transposed = false;

    std::swap(n_outer_dim,n_inner_dim);
    spect = spectrum_real_real.data();

    return true;
}

/**
 * Transpose a 2D array 
 * @param in: input array. size is n_dim1*n_dim2
 * @param n_dim1: size of first dimension (before transpose)
 * @param n_dim2: size of second dimension (before transpose)
*/
bool fid_2d::transpose_2d(float *in, int n_dim1, int n_dim2)
{
    std::vector<float> temp(n_dim1 * n_dim2, 0.0f);
    for (int i = 0; i < n_dim1; i++)
    {
        for (int j = 0; j < n_dim2; j++)
        {
            temp[j * n_dim1 + i] = in[i*n_dim2+j];
        }
    }
    /**
     * Copy data back to in
    */
    for (int i = 0; i < n_dim1 * n_dim2; i++)
    {
        in[i] = temp[i];
    }
    return true;
}

/**
 * @brief create_nmrpipe_dictionary: create nmrpipe dictionary from udict_acqus and derived values
 * @param b_init: whether to initialize nmrpipe_dict_string and nmrpipe_dict_float
 * @param nmrpipe_dict_string: output dictionary for string values
 * @param nmrpipe_dict_float: output dictionary for float values
 */
bool fid_2d::create_nmrpipe_dictionary(std::map<std::string, std::string> &nmrpipe_dict_string, std::map<std::string, float> &nmrpipe_dict_float, bool b_init) const
{
    /**
     * @brief create_default_nmrpipe_dictionary. 
     * This is useful we are NOT reading nmrPipe file.
     */
    if(b_init)
    {
        nmrPipe::create_default_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float);
    }

    /**
     * Fill in some parameters from what we have
     */
    nmrpipe_dict_float["FDDIMCOUNT"] = 2.0f; // 2D data

    if(b_nmrPipe_transposed)
    {
        nmrpipe_dict_float["FDTRANSPOSED"] = 1.0f; // transposed
        nmrpipe_dict_float["FDDIMORDER1"] = 1.0f; // inner loop is FDF1
        nmrpipe_dict_float["FDDIMORDER2"] = 2.0f; // outer loop is FDF2
    }
    else
    {
        nmrpipe_dict_float["FDTRANSPOSED"] = 0.0f; // not transposed
        nmrpipe_dict_float["FDDIMORDER1"] = 2.0f; // inner loop is FDF2
        nmrpipe_dict_float["FDDIMORDER2"] = 1.0f; // outer loop is FDF1
    }

    nmrpipe_dict_string["FDF2LABEL"] = nucleus;
    nmrpipe_dict_string["FDF1LABEL"] = nucleus_indirect;
    

    /**
     * Some examples of parameters that we need to put in nmrpipe_dict_float
     * 'sw': 12019.2307692308
     *  'complex':True
     *   'obs': 600.06
     *  'car': 2820.28200001605
     *   'size': 32768
     *  'label': '1H'
     */

    nmrpipe_dict_float["FDF2SW"] = spectral_width;
    nmrpipe_dict_float["FDF2OBS"] = observed_frequency;
    nmrpipe_dict_float["FDF2CAR"] = carrier_frequency / observed_frequency; // normalized to observed frequency, per nmrPipe convention
    nmrpipe_dict_string["FDF2LABEL"] = "1H";                                // hardcoded for now

    /**
     * indirect dimension
     */
    nmrpipe_dict_float["FDF1SW"] = spectral_width_indirect;
    nmrpipe_dict_float["FDF1OBS"] = observed_frequency_indirect;
    nmrpipe_dict_float["FDF1CAR"] = carrier_frequency_indirect / observed_frequency_indirect; // normalized to observed frequency, per nmrPipe convention
    nmrpipe_dict_string["FDF1LABEL"] = "13C";                                                 // hardcoded for now!!

    

    if(b_imaginary==true)
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 0.0f;
    }
    else
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 1.0f;
    }

    if(b_imaginary_indirect==true)
    {
        nmrpipe_dict_float["FDF1QUADFLAG"] = 0.0f;
    }
    else
    {
        nmrpipe_dict_float["FDF1QUADFLAG"] = 1.0f;
    }

   
    /**
     * Copied from nmrglue. no idea what they mean
     */
    if (nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF2QUADFLAG"] && nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF3QUADFLAG"] && nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF4QUADFLAG"] && nmrpipe_dict_float["FDF1QUADFLAG"] == 1.0f)
    {
        nmrpipe_dict_float["FDQUADFLAG"] = 1.0f;
    }

   
    /**
     * Outer loop size need to be adjusted when both are complex, per nmrPipe convention
    */
    int n_scale_factor = 1;
    if(b_imaginary==true && b_imaginary_indirect==true)
    {
        n_scale_factor = 2; 
    }

    /**
     * For both frq and time data, need save these
     * ndata_original is the original size before 0-pad, apoization, ZF and FT
     * ndata is the original size after 0-pad (by Bruker in its fid file), but before apoization, ZF and FT
     * Note: there is no 0-padding along indirect dimension(s)
     */
    nmrpipe_dict_float["FDF2TDSIZE"] = ndata_original;
    nmrpipe_dict_float["FDF1TDSIZE"] = ndata_indirect;

    /**
     * These are the size before apod.
     * nmrPipe ext will change these values, but I am not 100% sure how.
     * It seems FDF2APOD = FDF2TDSIZE * percentage of ext operation
     */
    int extraction_span = extraction_to_pos - extraction_from_pos;
    nmrpipe_dict_float["FDF2APOD"] = extraction_span / zf; // this is the size after extraction, but before apodization. We will use this to do apodization in nmrPipe
    nmrpipe_dict_float["FDF1APOD"] = ndata_indirect;

    /**
     * Basically we need to adjust CENTER and ORIG according to zero filling and extract (extract has not been implemented yet)
     * In nmrPipe, "-fn EXT -x1 10.8ppm -xn 5.8ppm -sw" will only change ORIG and CENTER, but not SW, OBS, CAR
     */
    nmrpipe_dict_float["FDF2CENTER"] = n_center;
    nmrpipe_dict_float["FDF2ORIG"] = origin_frequency;

    nmrpipe_dict_float["FDF1CENTER"] = n_center_indirect;
    nmrpipe_dict_float["FDF1ORIG"] = origin_frequency_indirect;

    nmrpipe_dict_float["FDF2APODCODE"] = apodization_code;
    nmrpipe_dict_float["FDF1APODCODE"] = apodization_code_indirect;

    /**
     * Both dimension are frq data
    */
    if (b_frq && b_frq_indirect)
    {
        /**
         * we are saving frq data, so set FDF2FTFLAG to 1 and set value for FDF2FTSIZE
         * FDSIZE is the size of inner loop
         * FDSPECNUM is the size of outer loop (when both are complex, we need to double FDSPECNUM)
         * When spectrum is not transposed, FDF2FTSIZE == FDSIZE and FDF1FTSIZE == FDSPECNUM
         * When spectrum is transposed, FDF2FTSIZE == FDSPECNUM and FDF1FTSIZE == FDSIZE
         * In nmrPipe, ext operation will change FDSIZE and/or FDSPECNUM, but not FDF2FTSIZE and FDF1FTSIZE
         */
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF1FTSIZE"] = ndata_frq_indirect;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_frq_indirect;

        nmrpipe_dict_float["FDF2ZF"] = -ndata_frq; //negative of size after zf
        nmrpipe_dict_float["FDF1ZF"] = -ndata_frq_indirect; //negative of size after zf
       
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f;
        nmrpipe_dict_float["FDF1FTFLAG"] = 1.0f;

        nmrpipe_dict_float["FDSIZE"] = n_inner_dim;
        nmrpipe_dict_float["FDSPECNUM"] = n_outer_dim * n_scale_factor;

        nmrpipe_dict_float["FDF2P0"] = user_p0_direct;
        nmrpipe_dict_float["FDF2P1"] = user_p1_direct;
    }
    /**
     * Case 2, only direct dimension is frq data. This is required to run Smile NUS reconstruction
    */
    else if(b_frq == true && b_frq_indirect == false)
    {
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF1FTSIZE"] = 0;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_indirect;

        nmrpipe_dict_float["FDF2ZF"] = -ndata_frq; //negative of size after zf
        nmrpipe_dict_float["FDF1ZF"] = 0; //negative of size after zf
       
        
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f;
        nmrpipe_dict_float["FDF1FTFLAG"] = 0.0f;


        nmrpipe_dict_float["FDSIZE"] = n_inner_dim;
        nmrpipe_dict_float["FDSPECNUM"] = n_outer_dim * n_scale_factor;

        nmrpipe_dict_float["FDF2P0"] = user_p0_direct;
        nmrpipe_dict_float["FDF2P1"] = user_p1_direct;
    }

    /**
     * Todo: other cases  only indirect dimension is frq data.
     * Not sure if this is needed
    */
    else if(b_frq == false && b_frq_indirect == false)
    {
        nmrpipe_dict_float["FDF2FTSIZE"] = 0;
        nmrpipe_dict_float["FDF1FTSIZE"] = 0;
        nmrpipe_dict_float["FDREALSIZE"] = 0;

        nmrpipe_dict_float["FDF2ZF"] = 0; //negative of size after zf
        nmrpipe_dict_float["FDF1ZF"] = 0; //negative of size after zf
       
        
        nmrpipe_dict_float["FDF2FTFLAG"] = 0.0f;
        nmrpipe_dict_float["FDF1FTFLAG"] = 0.0f;


        nmrpipe_dict_float["FDSIZE"] = n_inner_dim;
        nmrpipe_dict_float["FDSPECNUM"] = n_outer_dim * n_scale_factor;     
    }

   
    /**
     * Save indirect dimension phase correction information, which are read from pulse program
     * indirect_p0 < -360 means not set. 
     */
    nmrpipe_dict_float["FDF1P0"] = indirect_p0;
    nmrpipe_dict_float["FDF1P1"] = indirect_p1;
    

    return true;
}


bool fid_2d::prepare_header_for_nmrpipe()
{
    if (b_read_bruker_acqus_and_fid == true)
    {
        /**
         * create nmrpipe header.
         * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
         * from udict_acqus and derived values
         */
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, true);

        /**
         * define header vector, because the function call nmrpipe_dictionary_to_header won't apply for memory
         */
        nmrpipe_header_data.clear();
        nmrpipe_header_data.resize(512, 0.0f);

        nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
    }
    else
    {
        /**
         * We read nmrpipe file, so we already have nmrpipe_header_data
         * But we updated some values in nmrpipe_dict_float, so we need to update nmrpipe_header_data
         */
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, false);
        nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
    }
    return true;
}

/**
 * @brief fid_2d::write_nmrpipe_ft2: write the current frequency-domain spectrum to an nmrPipe file
 * Before writing, define the nmrPipe header from the current object state.
 * @param outfname: output file name
 * @return true on success, false on failure
 */
bool fid_2d::write_nmrpipe_ft2(std::string outfname)
{
    if (!b_frq)
    {
        std::cerr << "Error: write_nmrpipe_ft2 requires frequency-domain direct data." << std::endl;
        return false;
    }
    if (spectrum_real_real.empty())
    {
        std::cerr << "Error: no spectrum data available for write_nmrpipe_ft2." << std::endl;
        return false;
    }

    prepare_header_for_nmrpipe();

    /**
     * Write nmrPipe header and active spectrum data to the file.
     */
    FILE *fp = fopen(outfname.c_str(), "wb");
    if (fp == NULL)
    {
        std::cerr << "Error: cannot open file " << outfname << " for writing" << std::endl;
        return false;
    }
    /**
     * Write header first
     */
    fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);

    /**
     * Data are written trace by trace in nmrPipe layout.
     */
    for (int j = 0; j < n_outer_dim; j++)
    {
        fwrite(spectrum_real_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        /**
         * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
         */
        if (b_imaginary == true && b_imaginary_indirect == true)
        {
            fwrite(spectrum_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(spectrum_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(spectrum_imag_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary_indirect == true)
        {
            fwrite(spectrum_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary == true)
        {
            fwrite(spectrum_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
    }
    fclose(fp);

    /**
     * If nspcetra>1, we need to save the rest of the spectra to separate files
     */
    if (nspectra > 1)
    {
        /**
         * Find filename extension: after last . in outfname, such as .ft2.
         * In case there is no . in outfname, ext will be empty and basename will be outfname
         */
        std::string ext = outfname.substr(outfname.find_last_of(".") + 1);
        std::string basename = outfname.substr(0, outfname.find_last_of("."));

        for (int i = 1; i < nspectra; i++) // i starts from 1 because we already saved the first spectrum
        {
            std::string outfname2 = basename + "_" + std::to_string(i) + "." + ext;
            FILE *fp2 = fopen(outfname2.c_str(), "wb");
            if (fp2 == NULL)
            {
                std::cerr << "Error: cannot open file " << outfname2 << " for writing" << std::endl;
                return false;
            }
            /**
             * Write header first
             */
            fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp2);

            /**
             * Write the rest of the spectra
             */
            for (int j = 0; j < n_outer_dim; j++)
            {
                fwrite(spectrum_real_real.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                /**
                 * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
                 */
                if (b_imaginary == true && b_imaginary_indirect == true)
                {
                    fwrite(spectrum_real_imag.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                    fwrite(spectrum_imag_real.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                    fwrite(spectrum_imag_imag.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                }
                else if (b_imaginary_indirect == true)
                {
                    fwrite(spectrum_imag_real.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                }
                else if (b_imaginary == true)
                {
                    fwrite(spectrum_real_imag.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                }
            }
            fclose(fp2);
        }
    }

    return true;
}

uintptr_t fid_2d::write_nmrpipe_ft2_to_buffer()
{
    return write_nmrpipe_ft2_to_buffer_index(0);
}

uintptr_t fid_2d::write_nmrpipe_ft2_to_buffer_index(int spec_index)
{
    if (!b_frq)
    {
        std::cerr << "Error: write_nmrpipe_ft2_to_buffer requires frequency-domain direct data." << std::endl;
        return 0;
    }
    if (spectrum_real_real.empty())
    {
        std::cerr << "Error: no spectrum data available for write_nmrpipe_ft2_to_buffer." << std::endl;
        return 0;
    }
    if (n_outer_dim <= 0 || n_inner_dim <= 0)
    {
        std::cerr << "Error: invalid dimensions for write_nmrpipe_ft2_to_buffer." << std::endl;
        return 0;
    }
    if (spec_index < 0 || spec_index >= nspectra)
    {
        std::cerr << "Error: spectrum index " << spec_index << " out of range (0 to " << nspectra - 1 << ")." << std::endl;
        return 0;
    }

    size_t expected_size = static_cast<size_t>(n_outer_dim) * n_inner_dim;
    size_t required_elements = static_cast<size_t>(nspectra) * expected_size;

    if (spectrum_real_real.size() < required_elements)
    {
        std::cerr << "Error: spectrum_real_real data size mismatch." << std::endl;
        return 0;
    }

    if (b_imaginary == true && b_imaginary_indirect == true)
    {
        if (spectrum_real_imag.size() < required_elements ||
            spectrum_imag_real.size() < required_elements ||
            spectrum_imag_imag.size() < required_elements)
        {
            std::cerr << "Error: spectrum components size mismatch." << std::endl;
            return 0;
        }
    }
    else if (b_imaginary_indirect == true)
    {
        if (spectrum_imag_real.size() < required_elements)
        {
            std::cerr << "Error: spectrum components size mismatch." << std::endl;
            return 0;
        }
    }
    else if (b_imaginary == true)
    {
        if (spectrum_real_imag.size() < required_elements)
        {
            std::cerr << "Error: spectrum components size mismatch." << std::endl;
            return 0;
        }
    }

    prepare_header_for_nmrpipe();

    int num_components = 1;
    if (b_imaginary == true && b_imaginary_indirect == true)
    {
        num_components = 4;
    }
    else if (b_imaginary_indirect == true || b_imaginary == true)
    {
        num_components = 2;
    }

    size_t total_floats = 512 + expected_size * num_components;
    ft2_buffer.resize(total_floats);

    // Copy header (512 floats)
    std::memcpy(ft2_buffer.data(), nmrpipe_header_data.data(), 512 * sizeof(float));

    size_t offset = 512;
    size_t plane_offset = static_cast<size_t>(spec_index) * n_outer_dim * n_inner_dim;

    for (int j = 0; j < n_outer_dim; j++)
    {
        std::memcpy(ft2_buffer.data() + offset, spectrum_real_real.data() + plane_offset + j * n_inner_dim, n_inner_dim * sizeof(float));
        offset += n_inner_dim;
        if (b_imaginary == true && b_imaginary_indirect == true)
        {
            std::memcpy(ft2_buffer.data() + offset, spectrum_real_imag.data() + plane_offset + j * n_inner_dim, n_inner_dim * sizeof(float));
            offset += n_inner_dim;
            std::memcpy(ft2_buffer.data() + offset, spectrum_imag_real.data() + plane_offset + j * n_inner_dim, n_inner_dim * sizeof(float));
            offset += n_inner_dim;
            std::memcpy(ft2_buffer.data() + offset, spectrum_imag_imag.data() + plane_offset + j * n_inner_dim, n_inner_dim * sizeof(float));
            offset += n_inner_dim;
        }
        else if (b_imaginary_indirect == true)
        {
            std::memcpy(ft2_buffer.data() + offset, spectrum_imag_real.data() + plane_offset + j * n_inner_dim, n_inner_dim * sizeof(float));
            offset += n_inner_dim;
        }
        else if (b_imaginary == true)
        {
            std::memcpy(ft2_buffer.data() + offset, spectrum_real_imag.data() + plane_offset + j * n_inner_dim, n_inner_dim * sizeof(float));
            offset += n_inner_dim;
        }
    }

    return reinterpret_cast<uintptr_t>(ft2_buffer.data());
}

int fid_2d::get_ft2_size_in_float32()
{
    return static_cast<int>(ft2_buffer.size());
}

bool fid_2d::write_nmrpipe_fid(std::string outfname)
{
    if (!(b_frq == false && b_frq_indirect == false))
    {
        std::cerr << "Error: write_nmrpipe_fid requires FID stage." << std::endl;
        return false;
    }
    if (spectrum_real_real.empty())
    {
        std::cerr << "Error: no FID data available for write_nmrpipe_fid." << std::endl;
        return false;
    }

    prepare_header_for_nmrpipe();

    /**
     * now write nmrpipe_header_data and spectral to the file
     */
    FILE *fp = fopen(outfname.c_str(), "wb");
    if (fp == NULL)
    {
        std::cerr << "Error: cannot open file " << outfname << " for writing" << std::endl;
        return false;
    }
    /**
     * Write header first
     */
    fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);

    /**
     * This is how data are organized in nmrpipe file
     */
    for (int j = 0; j < n_outer_dim; j++)
    {
        fwrite(spectrum_real_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        /**
         * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
         */
        if (b_imaginary == true && b_imaginary_indirect == true)
        {
            fwrite(spectrum_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(spectrum_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(spectrum_imag_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary_indirect == true)
        {
            fwrite(spectrum_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary == true)
        {
            fwrite(spectrum_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
    }
    fclose(fp);

    return true;
}

#ifdef WEBASSEMBLY
/**
 * Function calls for web assembly to read header and data directly from variables without addtional copy
 */
uintptr_t fid_2d::get_nmrpipe_header_data()
{
    return reinterpret_cast<uintptr_t>(nmrpipe_header_data.data());
}
uintptr_t fid_2d::get_data_of_rr()
{
    return reinterpret_cast<uintptr_t>(spectrum_real_real.data());
}
uintptr_t fid_2d::get_data_of_ri()
{
    return reinterpret_cast<uintptr_t>(spectrum_real_imag.data());
}
uintptr_t fid_2d::get_data_of_ir()
{
    return reinterpret_cast<uintptr_t>(spectrum_imag_real.data());
}

uintptr_t fid_2d::get_data_of_ii()
{
    return reinterpret_cast<uintptr_t>(spectrum_imag_imag.data());
}


#endif
/**
 * Functions copied from fid_2d.cpp when combine them
*/

bool fid_2d::read_spectrum(std::string infname)
{
    bool b_read = 0;

    std::string stxt(".txt");
    std::string sft2(".ft2");
    std::string sft3(".ft3");
    std::string sucsf(".ucsf");
    std::string scsv(".csv");

    if (std::equal(stxt.rbegin(), stxt.rend(), infname.rbegin()))
    {
        b_read = read_topspin_txt(infname);
    }
    else if (std::equal(sft2.rbegin(), sft2.rend(), infname.rbegin()))
    {
        b_read = read_nmrpipe_file(infname);
    }
    else if (std::equal(sft3.rbegin(), sft3.rend(), infname.rbegin()))
    {
        b_read = read_nmrpipe_file(infname);
    }
    else if (std::equal(sucsf.rbegin(), sucsf.rend(), infname.rbegin()))
    {
        b_read = read_sparky(infname);
    }
    else if (std::equal(scsv.rbegin(), scsv.rend(), infname.rbegin()))
    {
        b_read = read_mnova(infname);
    }
    else
    {
        b_read = false;
    }

    return b_read;
};

bool fid_2d::init(std::string infname, int noise_flag)
{

    // std::cout<<"peak_diag is "<<peak_diag<<std::endl;
    // std::cout<<"flag_shoulder is "<<flag_shoulder<<std::endl;
    bool b_read;

    b_read = read_spectrum(infname);

    if (b_read)
    {
        std::cout << "Done reading" << std::endl;

        if (noise_flag == 1)
        {
            // estimate_noise_level_mad();    //estimate noise level using MAD
            estimate_noise_level(); // estimate noise level using region by region standard deviation
        }
    }
    return b_read;
}

bool fid_2d::save_mnova(std::string outfname)
{
    std::ofstream fout;

    std::vector<double> ppm1, ppm2;

    for (int i = 0; i < ndata_frq; i++)
    {
        ppm1.push_back(begin1 + i * step1);
    }

    for (int i = 0; i < ndata_frq_indirect; i++)
    {
        ppm2.push_back(begin2 + i * step2);
    }

    // first line, ppm along direct dimension
    for (int i = 0; i < ndata_frq; i++)
    {
        fout << ppm1[i] << " ";
    }
    fout << std::endl;

    for (int i = 0; i < ndata_frq_indirect; i++)
    {
        fout << ppm2[i]; // first element is ppm along indirect dimension
        for (int j = 0; j < ndata_frq; j++)
        {
            fout << " " << spect[i * ndata_frq + j];
        }
        fout << std::endl;
    }
    fout.close();

    return true;
}

bool fid_2d::read_mnova(std::string infname)
{
    int tp;
    std::string line, p;
    std::vector<std::string> ps, lines;
    std::stringstream iss;
    std::ifstream fin(infname.c_str());

    getline(fin, line); // first line
    iss.clear();
    iss.str(line);
    ps.clear();
    while (iss >> p)
    {
        ps.push_back(p);
    }

    if (ps.size() == 2) // new format in mnova 14.0
    {
        std::vector<double> x, y, z;

        lines.push_back(line);
        while (getline(fin, line))
        {
            lines.push_back(line);
        }

        std::string p1, p2;
        for (int i = 0; i < lines.size(); i++)
        {
            iss.clear();
            iss.str(lines[i]);
            iss >> p1;
            iss >> p2;
            z.push_back(atof(p2.c_str()));

            p1.erase(0, 1);
            p1.pop_back();
            iss.clear();
            iss.str(p1);
            ps.clear();
            while (std::getline(iss, p, ':'))
            {
                ps.push_back(p);
            }
            x.push_back(atof(ps[0].c_str()));
            y.push_back(atof(ps[1].c_str()));
        }

        std::set<double> xx(x.begin(), x.end());
        std::set<double> yy(y.begin(), y.end());

        ndata_frq = xx.size();
        ndata_frq_indirect = yy.size();

        begin1 = *xx.rbegin();
        stop1 = *xx.begin();
        step1 = -(begin1 - stop1) / (ndata_frq - 1);
        begin2 = *yy.rbegin();
        stop2 = *yy.begin();
        step2 = -(begin2 - stop2) / (ndata_frq_indirect - 1);

        std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
        std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm, stop at " << stop1 << std::endl;
        std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm, stop at " << stop2 << std::endl;

        spectrum_real_real.clear();
        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spect = spectrum_real_real.data();

        for (int i = 0; i < z.size(); i++)
        {
            int k1 = round(-(begin1 - x[i]) / step1);
            int k2 = round(-(begin2 - y[i]) / step2);
            if (k1 < 0 || k1 >= ndata_frq || k2 < 0 || k2 >= ndata_frq_indirect)
            {
                std::cout << "ERROR in read mnova format, k1=" << k1 << " k2=" << k2 << std::endl;
            }
            spect[k2 * ndata_frq + k1] = float(z[i]);
            // std::cout<<x[i]<<" "<<k1<<" "<<y[i]<<" "<<k2<<" "<<z[i]<<std::endl;
        }
    }

    else // format in mnova 10.0
    {

        if (fabs(atof(ps[ps.size() - 1].c_str())) < 0.0000001)
        {
            ps.erase(ps.end() - 1); // last element is meaningless for first line if it is zero!!
        }

        tp = ps.size();
        ndata_frq = tp;

        float nouse;
        double tswap;
        bool bswap1 = 0, bswap2 = 0;

        begin1 = atof(ps[0].c_str());
        stop1 = atof(ps[tp - 1].c_str());

        if (begin1 < stop1)
        {
            tswap = begin1;
            begin1 = stop1;
            stop1 = tswap;
            bswap1 = 1;
        }

        step1 = (stop1 - begin1) / (ndata_frq - 1);

        while (getline(fin, line))
        {
            if (line.length() > ndata_frq * 2 - 1)
            {
                lines.push_back(line);
            }
        }

        ndata_frq_indirect = lines.size();

        float *spect0;
        spect0 = new float[ndata_frq * ndata_frq_indirect];
        spectrum_real_real.clear();
        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spect = spectrum_real_real.data();
        for (int i = 0; i < lines.size(); i++)
        {
            iss.clear();
            iss.str(lines[i]);
            if (i == 0)
                iss >> begin2;
            else if (i == ndata_frq_indirect - 1)
                iss >> stop2;
            else
                iss >> nouse;

            for (int j = 0; j < ndata_frq; j++)
            {
                iss >> nouse;
                spect0[i * ndata_frq + j] = nouse;
            }
        }
        if (begin2 < stop2)
        {
            tswap = begin2;
            begin2 = stop2;
            stop2 = tswap;
            bswap2 = 1;
        }
        step2 = (stop2 - begin2) / (ndata_frq_indirect - 1);

        if (bswap2 == 0)
        {
            for (int i = 0; i < ndata_frq_indirect; i++)
            {
                for (int j = 0; j < ndata_frq; j++)
                {
                    if (bswap1 == 1)
                        spect[i * ndata_frq + ndata_frq - 1 - j] = spect0[i * ndata_frq + j];
                    else
                        spect[i * ndata_frq + j] = spect0[i * ndata_frq + j];
                }
            }
        }
        else
        {
            for (int i = 0; i < ndata_frq_indirect; i++)
            {
                for (int j = 0; j < ndata_frq; j++)
                {
                    if (bswap1 == 1)
                        spect[(ndata_frq_indirect - 1 - i) * ndata_frq + ndata_frq - 1 - j] = spect0[i * ndata_frq + j];
                    else
                        spect[(ndata_frq_indirect - 1 - i) * ndata_frq + j] = spect0[i * ndata_frq + j];
                }
            }
        }
        delete[] spect0;
    }

    // aribitary frq becuase we don't have that infor
    observed_frequency = observed_frequency_indirect = 850.0;
    spectral_width = observed_frequency * (begin1 - stop1);
    spectral_width_indirect = observed_frequency_indirect * (begin2 - stop2);
    origin_frequency = stop1 * observed_frequency;
    origin_frequency_indirect = stop2 * observed_frequency_indirect;

    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    b_frq = true;
    b_frq_indirect = true;
    b_imaginary = false;
    b_imaginary_indirect = false;
    b_nmrPipe_transposed = false;
    n_outer_dim = ndata_frq_indirect;
    n_inner_dim = ndata_frq;

    return true;
}

bool fid_2d::get_ppm_from_point()
{
    // get ppm
    p1_ppm.clear();
    p2_ppm.clear();

    for (unsigned int i = 0; i < p1.size(); i++)
    {
        double f1 = begin1 + step1 * (p1[i]); // direct dimension
        double f2 = begin2 + step2 * (p2[i]); // indirect dimension
        p1_ppm.push_back(f1);
        p2_ppm.push_back(f2);
    }

    return true;
};

// read topspin file in ASCIC format, genearted using command totxt
bool fid_2d::read_topspin_txt(std::string infname)
{

    int tp;
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;
    std::ifstream fin(infname.c_str());

    if (!fin)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    double f1left, f1right, f2left, f2right;

    f1left = f1right = f2left = f2right = 0.0;
    ndata_frq = ndata_frq_indirect = 0;

    // read in head information
    while (getline(fin, line) && (ndata_frq == 0 || ndata_frq_indirect == 0))
    {
        if (line.find("F1LEFT") != std::string::npos && line.find("F1RIGHT") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 9)
            {
                f1left = atof(ps[3].c_str());
                f1right = atof(ps[7].c_str());
            }
        }

        if (line.find("F2LEFT") != std::string::npos && line.find("F2RIGHT") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 9)
            {
                f2left = atof(ps[3].c_str());
                f2right = atof(ps[7].c_str());
            }
        }

        if (line.find("NROWS") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 4)
            {
                ndata_frq_indirect = atoi(ps[3].c_str());
            }
        }

        if (line.find("NCOLS") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 4)
            {
                ndata_frq = atoi(ps[3].c_str());
            }
        }
    }

    spectrum_real_real.clear();
    spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
    spect = spectrum_real_real.data();

    int row_index = -1;
    int flag = 0;
    int col_index;

    while (getline(fin, line))
    {
        if (line.find("# row = ") != std::string::npos)
        {
            row_index = atoi(line.substr(7).c_str());
            col_index = 0;
            continue;
        }
        else if (line.find("#") != std::string::npos)
        {
            continue;
        }

        spect[col_index + row_index * ndata_frq] = atof(line.c_str());
        col_index++;
    }

    begin1 = f2left;
    stop1 = f2right;
    step1 = (stop1 - begin1) / (ndata_frq);
    stop1 = f2right + step1; // stop is the ppm of the last col.

    begin2 = f1left;
    stop2 = f1right;
    step2 = (stop2 - begin2) / (ndata_frq_indirect);
    stop2 = f1right + step2; // stop is the ppm of the last col.

    // fill in required variable to save pipe format
    //  set frq= 850. This is arbitary
    observed_frequency = observed_frequency_indirect = 850.0;
    spectral_width = observed_frequency * (begin1 - stop1);
    spectral_width_indirect = observed_frequency_indirect * (begin2 - stop2);
    origin_frequency = stop1 * observed_frequency;
    origin_frequency_indirect = stop2 * observed_frequency_indirect;

    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    b_frq = true;
    b_frq_indirect = true;
    b_imaginary = false;
    b_imaginary_indirect = false;
    b_nmrPipe_transposed = false;
    n_outer_dim = ndata_frq_indirect;
    n_inner_dim = ndata_frq;

    return true;
}

bool fid_2d::read_txt(std::string infname)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;
    std::vector<float> temp;

    std::ifstream fin(infname.c_str());

    if (!fin)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    ndata_frq = ndata_frq_indirect = 0;
    while (getline(fin, line))
    {
        ndata_frq_indirect++;
        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            temp.push_back(atof(p.c_str()));
        }
        if (ndata_frq == 0)
        {
            ndata_frq = temp.size();
        }
    }

    spectrum_real_real.clear();
    spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
    spectrum_real_imag.clear();
    spectrum_imag_real.clear();
    spectrum_imag_imag.clear();
    for (unsigned int i = 0; i < ndata_frq * ndata_frq_indirect; i++)
    {
        spectrum_real_real[i] = temp[i];
    }
    spect = spectrum_real_real.data();

    begin1 = 0.6;
    stop1 = 0;
    step1 = (stop1 - begin1) / (ndata_frq);
    stop1 = stop1 - step1; // stop is the ppm of the last col.

    begin2 = 0.6;
    stop2 = 0;
    step2 = (stop2 - begin2) / (ndata_frq_indirect);
    stop2 = stop2 - step2; // stop is the ppm of the last col.

    b_frq = true;
    b_frq_indirect = true;
    b_imaginary = false;
    b_imaginary_indirect = false;
    b_nmrPipe_transposed = false;
    n_outer_dim = ndata_frq_indirect;
    n_inner_dim = ndata_frq;

    return true;
}


/**
 * This is a protected function, not to be called by user
 * This fucntion only write a ft2 file, for which deep pickerfitter and nmrdraw can read.
 * it is not a 100% compatible ft2 file.
 */

bool fid_2d::write_pipe_from_spect(std::vector<std::vector<float>> spect, std::string fname)
{

    if(b_read_nmrpipe_fid == false)
    {
        /**
         * Initialize the header with all zeros
        */
        nmrpipe_header_data.clear();
        nmrpipe_header_data.resize(512, 0.0f);
    }

    if (1) // otherwise we just use the input header
    {
        nmrpipe_header_data[0] = 0.0f;
        nmrpipe_header_data[219] = float(ndata_frq_indirect);
        nmrpipe_header_data[99] = float(ndata_frq);
        nmrpipe_header_data[106] = float(1);
        nmrpipe_header_data[256] = float(2);

        nmrpipe_header_data[220] = 1.0; //frequency domain data along direct dimension
        nmrpipe_header_data[222] = 1.0; //frequency domain data along indirect dimension

        nmrpipe_header_data[9] = 2;      // dimension is 2
        nmrpipe_header_data[57] = 0.0f;  // 2d spectrum
        nmrpipe_header_data[221] = 0.0f; // not transposed
        nmrpipe_header_data[24] = 2.0f;  // first dimension is 2
        nmrpipe_header_data[25] = 1.0f;  // second dimension is 1
        nmrpipe_header_data[26] = 3.0f;  // z dimension is 3
        nmrpipe_header_data[27] = 4.0f;  // A dimension is 4

        nmrpipe_header_data[101 - 1] = (float)spectral_width; // second dimension
        nmrpipe_header_data[120 - 1] = (float)observed_frequency;
        nmrpipe_header_data[102 - 1] = (float)origin_frequency;

        nmrpipe_header_data[230 - 1] = (float)spectral_width_indirect; // first dimension
        nmrpipe_header_data[219 - 1] = (float)observed_frequency_indirect;
        nmrpipe_header_data[250 - 1] = (float)origin_frequency_indirect;
    }

    nmrpipe_header_data[56 - 1] = 1.0f; // real data only along indirect dimension
    nmrpipe_header_data[57 - 1] = 1.0f; // real data only along direct dimension

    FILE *fp = fopen(fname.c_str(), "w");
    if (fp == NULL)
    {
        std::cout << "cannot open file " << fname.c_str() << "to write" << std::endl;
    }
    else
    {
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
        for (unsigned int i = 0; i < ndata_frq_indirect; i++)
        {
            fwrite(spect[i].data(), sizeof(float), ndata_frq, fp);
        }
        fclose(fp);
    }

    return true;
};

/**
 * Write a ft2 file, for which deep pickerfitter and nmrdraw can read.
 */
bool fid_2d::write_pipe(std::string fname,bool b_real_only)
{
    FILE *fp = fopen(fname.c_str(), "w");
    if (fp == NULL)
    {
        std::cout << "cannot open file " << fname.c_str() << "to write" << std::endl;
    }
    else
    {
        if(b_real_only==true)
        {
            /**
             * We read in complex data, but only write real data
            */
            if(b_imaginary==true && b_imaginary_indirect==true)
            {
                nmrpipe_header_data[56 - 1] = 1.0f; // real data only along indirect dimension
                nmrpipe_header_data[57 - 1] = 1.0f; // real data only along direct dimension
                nmrpipe_header_data[220 - 1] = ndata_frq_indirect; // ydim is half of the original ydim
            }
        }
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
        if (b_imaginary_indirect== false && b_imaginary == false)
        {
            fwrite(spect, sizeof(float), ndata_frq*ndata_frq_indirect, fp);
        }
        else if (b_imaginary == true && b_imaginary_indirect == true)
        {
            for (unsigned int i = 0; i < ndata_frq_indirect; i++)
            {
                fwrite(spectrum_real_real.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                /**
                 * We have complex data available, and we want to write complex data (not real only)
                */
                if(b_real_only == false && b_imaginary == true && b_imaginary_indirect == true)
                {
                    fwrite(spectrum_real_imag.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                    fwrite(spectrum_imag_real.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                    fwrite(spectrum_imag_imag.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                }
            }
        }
        fclose(fp);
    }

    return true;
};


bool fid_2d::read_sparky(std::string infname)
{
    FILE *fp;

    char buffer[10];
    int temp;

    fp = fopen(infname.c_str(), "rb");

    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    if (fread(buffer, 1, 10, fp) != 10)
    {
        std::cout << "Error reading sparky file header" << std::endl;
        fclose(fp);
        return false;
    }
    // std::cout<<buffer<<std::endl;

    if (fread(buffer, 1, 1, fp) != 1)
    {
        std::cout << "Error reading sparky file dimension" << std::endl;
        fclose(fp);
        return false;
    }
    temp = int(buffer[0]);
    if (temp != 2)
    {
        std::cout << "Error in sparky format file, dimension is not 2" << std::endl;
        fclose(fp);
        return false;
    }

    if (fread(buffer, 1, 1, fp) != 1)
    {
        std::cout << "Error reading sparky file quad type" << std::endl;
        fclose(fp);
        return false;
    }
    fseek(fp, 1, SEEK_CUR);
    temp = int(buffer[0]);
    if (temp != 1)
    {
        std::cout << "Error in sparky format file, it is not in real data" << std::endl;
        fclose(fp);
        return false;
    }

    if (fread(buffer, 1, 1, fp) != 1)
    {
        std::cout << "Error reading sparky file version" << std::endl;
        fclose(fp);
        return false;
    }
    // std::cout<<"Version is "<< int(buffer[0])<<std::endl;
    fseek(fp, 166, SEEK_CUR);

    // read 2d header

    float center1, center2;
    int tile1, tile2;

    if (fread(buffer, 1, 6, fp) != 6)
    {
        std::cout << "Error reading indirect dimension nucleus" << std::endl;
        fclose(fp);
        return false;
    }
    buffer[6] = '\0';
    std::cout << "Indirect dimension nuleus " << buffer << std::endl;
    fseek(fp, 2, SEEK_CUR);
    ndata_frq_indirect = read_int(fp);
    fseek(fp, 4, SEEK_CUR);
    tile2 = read_int(fp);
    observed_frequency_indirect = read_float(fp);
    spectral_width_indirect = read_float(fp);
    center2 = read_float(fp);
    fseek(fp, 96, SEEK_CUR);

    if (fread(buffer, 1, 6, fp) != 6)
    {
        std::cout << "Error reading direct dimension nucleus" << std::endl;
        fclose(fp);
        return false;
    }
    buffer[6] = '\0';
    std::cout << "Direct dimension nuleus " << buffer << std::endl;
    fseek(fp, 2, SEEK_CUR);
    ndata_frq = read_int(fp);
    fseek(fp, 4, SEEK_CUR);
    tile1 = read_int(fp);
    observed_frequency = read_float(fp);
    spectral_width = read_float(fp);
    center1 = read_float(fp);
    fseek(fp, 96, SEEK_CUR);

    // read in data here
    spectrum_real_real.clear();
    spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
    spect = spectrum_real_real.data();

    int ntile1 = int(ceil((double(ndata_frq)) / tile1));
    int ntile2 = int(ceil((double(ndata_frq_indirect)) / tile2));

    int last_tile1 = ndata_frq % tile1;
    if (last_tile1 == 0)
        last_tile1 = tile1;

    float *float_buff;
    float_buff = new float[tile1];

    for (int i = 0; i < ntile2; i++)
    {
        for (int j = 0; j < ntile1; j++)
        {
            for (int m = 0; m < tile2; m++)
            {
                read_float(fp, tile1, float_buff);
                if (i * tile2 + m < ndata_frq_indirect)
                {
                    if (j == ntile1 - 1)
                        memcpy(spect + (i * tile2 + m) * ndata_frq + (j * tile1), float_buff, last_tile1 * 4);
                    else
                        memcpy(spect + (i * tile2 + m) * ndata_frq + (j * tile1), float_buff, tile1 * 4);
                }
            }
        }
    }

    delete[] float_buff;

    float range1 = spectral_width / observed_frequency;
    step1 = -range1 / ndata_frq;
    begin1 = center1 + range1 / 2;
    stop1 = center1 - range1 / 2;

    float range2 = spectral_width_indirect / observed_frequency_indirect;
    step2 = -range2 / ndata_frq_indirect;
    begin2 = center2 + range2 / 2;
    stop2 = center2 - range2 / 2;

    // file in ref from center, so that we can save in pipe format if required.
    origin_frequency = center1 * observed_frequency - spectral_width / 2;
    origin_frequency_indirect = center2 * observed_frequency_indirect - spectral_width_indirect / 2;

    std::cout << "Spectrum width are " << spectral_width << " Hz and " << spectral_width_indirect << " Hz" << std::endl;
    std::cout << "Fields are " << observed_frequency << " mHz and " << observed_frequency_indirect << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    b_frq = true;
    b_frq_indirect = true;
    b_imaginary = false;
    b_imaginary_indirect = false;
    b_nmrPipe_transposed = false;
    n_outer_dim = ndata_frq_indirect;
    n_inner_dim = ndata_frq;

    fclose(fp);

    return true;
};

void fid_2d::estimate_noise_level_mad()
{
    std::cout << "In noise estimation, ndata_frq*ydim is " << ndata_frq * ndata_frq_indirect << std::endl;

    std::vector<float> t(spect, spect + ndata_frq * ndata_frq_indirect);

    for (unsigned int i = 0; i < t.size(); i++)
    {
        if (t[i] < 0)
            t[i] *= -1;
    }

    std::vector<float> scores = t;

    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2] * 1.4826;
    if (noise_level <= 0.0)
        noise_level = 0.1; // artificail spectrum w/o noise

    // noise_level=87353.0;
    // std::cout<<"ERROR: set noise level to "<<noise_level<<std::endl;

    std::cout << "First round, noise level is " << noise_level << std::endl;

    std::vector<int> flag(ndata_frq * ndata_frq_indirect, 0); // flag

    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        for (int i = 0; i < ndata_frq; i++)
        {
            if (t[j * ndata_frq + i] > 5.5 * noise_level)
            {
                int ystart = std::max(j - 5, 0);
                int yend = std::min(j + 6, ndata_frq_indirect);
                int xstart = std::max(i - 5, 0);
                int xend = std::min(i + 6, ndata_frq);
                for (int m = ystart; m < yend; m++)
                {
                    for (int n = xstart; n < xend; n++)
                    {
                        flag[m * ndata_frq + n] = 1;
                    }
                }
            }
        }
    }
    // after this, all datapoint > 5.5*noise and their surrounding datapoint are labeled as 1.
    scores.clear();
    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        for (int i = 0; i < ndata_frq; i++)
        {
            if (flag[j * ndata_frq + i] == 0)
            {
                scores.push_back(t[j * ndata_frq + i]);
            }
        }
    }
    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2] * 1.4826;
    if (noise_level <= 0.0)
        noise_level = 0.1; // artificail spectrum w/o noise
    std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;

    // estimate noise level column by column for TOCSY t1 noise belt identification!!
    for (int i = 0; i < ndata_frq; i++)
    {
        std::vector<float> col(ndata_frq_indirect);
        for (int j = 0; j < ndata_frq_indirect; j++)
        {
            col[j] = spect[j * ndata_frq + i];
        }
        std::vector<float> detrended_col(ndata_frq_indirect);
        fid_2d_math::detrend_slice_1d(col.data(), detrended_col.data(), ndata_frq_indirect);

        std::vector<float> scores;
        scores.reserve(ndata_frq_indirect);
        for (int j = 0; j < ndata_frq_indirect; j++)
        {
            scores.push_back(std::abs(detrended_col[j]));
        }
        sort(scores.begin(), scores.end());
        noise_level_columns.push_back(scores[ndata_frq_indirect / 3] * 1.4826);
    }

    // estimate noise level row by row
    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        std::vector<float> row(ndata_frq);
        for (int i = 0; i < ndata_frq; i++)
        {
            row[i] = spect[j * ndata_frq + i];
        }
        std::vector<float> detrended_row(ndata_frq);
        fid_2d_math::detrend_slice_1d(row.data(), detrended_row.data(), ndata_frq);

        std::vector<float> scores;
        scores.reserve(ndata_frq);
        for (int i = 0; i < ndata_frq; i++)
        {
            scores.push_back(std::abs(detrended_row[i]));
        }
        sort(scores.begin(), scores.end());
        noise_level_rows.push_back(scores[ndata_frq / 3] * 1.4826);
    }
};

void fid_2d::estimate_noise_level()
{
    std::cout << "In noise estimation, ndata_frq*ydim is " << ndata_frq * ndata_frq_indirect << std::endl;

    int n_segment_x = ndata_frq / 32;
    int n_segment_y = ndata_frq_indirect / 32;

    Eigen::MatrixXd D(1024, 5);
    for (int r = 0; r < 32; ++r)
    {
        double y_r = 2.0 * r / 31.0 - 1.0;
        for (int c = 0; c < 32; ++c)
        {
            double x_c = 2.0 * c / 31.0 - 1.0;
            int idx = r * 32 + c;
            D(idx, 0) = 1.0;
            D(idx, 1) = x_c;
            D(idx, 2) = y_r;
            D(idx, 3) = x_c * x_c;
            D(idx, 4) = y_r * y_r;
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeThinU | Eigen::ComputeThinV);

    std::vector<float> variances;      // variance of each segment
    std::vector<float> maximal_values; // maximal value of each segment

    /**
     * loop through each segment, and calculate variance
     */
    for (int i = 0; i < n_segment_x; i++)
    {
        for (int j = 0; j < n_segment_y; j++)
        {
            std::vector<float> t;
            t.reserve(1024);
            float max_of_t = 0.0f;
            Eigen::VectorXd b(1024);

            for (int m = 0; m < 32; m++)
            {
                for (int n = 0; n < 32; n++)
                {
                    float val = spect[(j * 32 + m) * ndata_frq + i * 32 + n];
                    t.push_back(val);
                    b(m * 32 + n) = static_cast<double>(val);
                    if (std::abs(val) > max_of_t)
                    {
                        max_of_t = std::abs(val);
                    }
                }
            }

            Eigen::VectorXd coeffs = svd.solve(b);
            Eigen::VectorXd resid = b - D * coeffs;
            float variance_of_t = static_cast<float>(resid.squaredNorm() / 1024.0);

            variances.push_back(variance_of_t);
            maximal_values.push_back(max_of_t);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    std::vector<float> variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);
    std::cout << "Noise level is " << noise_level << " using variance estimation." << std::endl;

    /**
     * loop through maximal_values, remove the ones that are larger than 10.0*noise_level
     * remove the corresponding variance as well
     */
    for (int i = maximal_values.size() - 1; i >= 0; i--)
    {
        if (maximal_values[i] > 10.0 * noise_level)
        {
            maximal_values.erase(maximal_values.begin() + i);
            variances.erase(variances.begin() + i);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);

    std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;

    // estimate noise level column by column for TOCSY t1 noise belt identification!!
    for (int i = 0; i < ndata_frq; i++)
    {
        std::vector<float> col(ndata_frq_indirect);
        for (int j = 0; j < ndata_frq_indirect; j++)
        {
            col[j] = spect[j * ndata_frq + i];
        }
        std::vector<float> detrended_col(ndata_frq_indirect);
        fid_2d_math::detrend_slice_1d(col.data(), detrended_col.data(), ndata_frq_indirect);

        std::vector<float> scores;
        scores.reserve(ndata_frq_indirect);
        for (int j = 0; j < ndata_frq_indirect; j++)
        {
            scores.push_back(std::abs(detrended_col[j]));
        }
        sort(scores.begin(), scores.end());
        noise_level_columns.push_back(scores[ndata_frq_indirect / 3] * 1.4826);
    }

    // estimate noise level row by row
    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        std::vector<float> row(ndata_frq);
        for (int i = 0; i < ndata_frq; i++)
        {
            row[i] = spect[j * ndata_frq + i];
        }
        std::vector<float> detrended_row(ndata_frq);
        fid_2d_math::detrend_slice_1d(row.data(), detrended_row.data(), ndata_frq);

        std::vector<float> scores;
        scores.reserve(ndata_frq);
        for (int i = 0; i < ndata_frq; i++)
        {
            scores.push_back(std::abs(detrended_row[i]));
        }
        sort(scores.begin(), scores.end());
        noise_level_rows.push_back(scores[ndata_frq / 3] * 1.4826);
    }

    return;
}

#ifdef WEBASSEMBLY
/**
 * Exposed functions
*/
EMSCRIPTEN_BINDINGS(dp_module_fid_2d) {

    register_vector<unsigned char>("VectorUChar");

    /**
     * Note: base class function init is exposed in spectrum_pick_1d.cpp, so it is not included here.
     * fid_base is included in spectrum_pick_1d.cpp, so it is not included here.
    */

    class_<fid_2d,base<fid_base>>("fid_2d")
        .constructor()
        .function("set_aqseq", &fid_2d::set_aqseq)
        .function("set_negative", &fid_2d::set_negative)
        .function("set_first_only", &fid_2d::set_first_only)
        .function("set_up_apodization_from_string", &fid_2d::set_up_apodization_from_string)
        .function("run_zf", &fid_2d::run_zf)
        .function("extract_region", &fid_2d::extract_region)
        .function("extract_region_ppm", &fid_2d::extract_region_ppm)
        .function("read_phase_correction_from_string", &fid_2d::read_phase_correction_from_string)
        .function("read_nus_list_from_string", &fid_2d::read_nus_list_from_string)
        .function("read_bruker_files_as_strings", &fid_2d::read_bruker_files_as_strings)
        .function("read_bruker_fid_data", &fid_2d::read_bruker_fid_data)
        .function("read_bruker_fid_data_bytes", &fid_2d::read_bruker_fid_data_bytes)
        .function("read_nmrpipe_file_from_buffer", &fid_2d::read_nmrpipe_file_from_buffer) //from base class fid_2d
        .function("direct_only_process", &fid_2d::direct_only_process)
        .function("indirect_only_process", &fid_2d::indirect_only_process)
        .function("full_process", &fid_2d::full_process)
        .function("other_process", &fid_2d::other_process)
        .function("water_suppression", &fid_2d::water_suppression)
        .function("polynorminal_baseline", &fid_2d::polynorminal_baseline)
        .function("write_json_as_string", &fid_2d::write_json_as_string)
        .function("write_pseudo3d_json_as_string", &fid_2d::write_pseudo3d_json_as_string)
        .function("write_nmrpipe_ft2_to_buffer", &fid_2d::write_nmrpipe_ft2_to_buffer)
        .function("write_nmrpipe_ft2_to_buffer_index", &fid_2d::write_nmrpipe_ft2_to_buffer_index)
        .function("get_ft2_size_in_float32", &fid_2d::get_ft2_size_in_float32)
        .function("get_nspectra", &fid_2d::get_nspectra)
        /**
         * Below are used to extract ft2 data from fid_2d object, used by web assembly code.
         */
        .function("get_ndata_frq", &fid_2d::get_ndata_frq)
        .function("get_ndata_frq_indirect", &fid_2d::get_ndata_frq_indirect)
        .function("get_inner_dim", &fid_2d::get_inner_dim)
        .function("get_outer_dim", &fid_2d::get_outer_dim)
        .function("prepare_header_for_nmrpipe", &fid_2d::prepare_header_for_nmrpipe)
        .function("get_nmrpipe_header_data", &fid_2d::get_nmrpipe_header_data)
        .function("get_data_of_rr", &fid_2d::get_data_of_rr)
        .function("get_data_of_ri", &fid_2d::get_data_of_ri)
        .function("get_data_of_ir", &fid_2d::get_data_of_ir)
        .function("get_data_of_ii", &fid_2d::get_data_of_ii)
        ;

    
    /**
     * float vector registration is also done in spectrum_pick_1d.cpp, not here.
    */
}
#endif // WEBASSEMBLY