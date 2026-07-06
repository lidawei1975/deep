#include "fid_3d.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <queue>
#include <cstring>
#include <cmath>
#include <climits>
#include <chrono>
#include <utility>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

#include <Eigen/Dense>

#include "kiss_fft.h"
#include "lmminimizer.h"

#ifdef WEBASSEMBLY
#include <emscripten/bind.h>
using namespace emscripten;
#endif

#ifndef WEBASS_DEBUG
#define WEBASS_DEBUG 1
#endif

static constexpr float FwhhFacctor = 3.0f;
static constexpr float FwhhFacctorSeperation = 5.0f;
static constexpr float FwhhFacctorFitting = 2.0f;
static constexpr bool kDebugDumpEmRoundPeaks = false;
static constexpr int kDebugDumpEmRoundInterval = 1;
static constexpr bool kUseWatershedFittingRegion = true;

namespace fid_3d_aux
{

#if WEBASS_DEBUG
    void debug_print_random_samples(const std::vector<float> &data,
                                    const std::string &label,
                                    size_t sample_count = 5)
    {
        if (data.empty())
        {
            std::cout << "[WEBASS_DEBUG] " << label << " is empty" << std::endl;
            return;
        }

        const size_t fixed_indices[] = {
            static_cast<size_t>(211754),
            static_cast<size_t>(330868),
            static_cast<size_t>(170482),
            static_cast<size_t>(296336),
            static_cast<size_t>(449375)};

        const size_t requested = sizeof(fixed_indices) / sizeof(fixed_indices[0]);
        const size_t n_samples = std::min(sample_count, requested);

        std::cout << "[WEBASS_DEBUG] " << label << " fixed samples:";
        for (size_t i = 0; i < n_samples; ++i)
        {
            const size_t idx = fixed_indices[i];
            if (idx < data.size())
            {
                std::cout << " (" << idx << ":" << data[idx] << ")";
            }
            else
            {
                std::cout << " (" << idx << ":out_of_range)";
            }
        }
        std::cout << std::endl;
    }
#endif

    bool read_nus_pairs_from_string(const std::string &nus_text, std::vector<std::array<int, 2>> &nus_pairs)
    {
        std::istringstream iss(nus_text);
        std::string line;
        nus_pairs.clear();

        while (std::getline(iss, line))
        {
            std::istringstream line_stream(line);
            int y = 0;
            int z = 0;
            if (!(line_stream >> y >> z))
            {
                continue;
            }
            if (y < 0 || z < 0)
            {
                std::cerr << "Error: nuslist coordinates must be non-negative." << std::endl;
                return false;
            }
            nus_pairs.push_back({y, z});
        }

        return !nus_pairs.empty();
    }

    bool expand_nus_raw_buffer(const std::vector<float> &sampled_raw,
                               int n_direct_raw,
                               int n_indirect1_raw,
                               int n_indirect2_raw,
                               const std::vector<std::array<int, 2>> &nus_pairs,
                               std::vector<float> &expanded_raw)
    {
        if (n_direct_raw <= 0 || n_indirect1_raw <= 0 || n_indirect2_raw <= 0)
        {
            return false;
        }

        const size_t trace_stride = static_cast<size_t>(n_direct_raw);
        const size_t expanded_total = static_cast<size_t>(n_direct_raw) * static_cast<size_t>(n_indirect1_raw) * static_cast<size_t>(n_indirect2_raw);
        expanded_raw.assign(expanded_total, 0.0f);

        if (nus_pairs.empty())
        {
            return false;
        }

        const int logical_y_size = n_indirect1_raw / 2;
        const int logical_z_size = n_indirect2_raw / 2;
        const int qy_size = (logical_y_size > 0) ? (n_indirect1_raw / logical_y_size) : 0;
        const int qz_size = (logical_z_size > 0) ? (n_indirect2_raw / logical_z_size) : 0;

        if (qy_size <= 0 || qz_size <= 0)
        {
            std::cerr << "Error: invalid indirect quadrature factors in NUS expansion." << std::endl;
            return false;
        }

        const size_t traces_per_sample = static_cast<size_t>(qy_size) * static_cast<size_t>(qz_size);
        const size_t expected_trace_count = static_cast<size_t>(nus_pairs.size()) * traces_per_sample;
        if (sampled_raw.size() != expected_trace_count * trace_stride)
        {
            std::cerr << "Error: nus raw payload size does not match sampleCount*qy*qz ("
                      << nus_pairs.size() << "*" << qy_size << "*" << qz_size << ")." << std::endl;
            return false;
        }

        for (size_t i = 0; i < nus_pairs.size(); ++i)
        {
            const int y = nus_pairs[i][0];
            const int z = nus_pairs[i][1];
            if (y >= logical_y_size || z >= logical_z_size)
            {
                std::cerr << "Error: nuslist coordinate out of range: y=" << y << ", z=" << z << std::endl;
                return false;
            }

            for (int izq = 0; izq < qz_size; ++izq)
            {
                for (int iyq = 0; iyq < qy_size; ++iyq)
                {
                    // Match nusExpand.tcl aqORD=0 source layout for expand mode:
                    // per sample loc, iyq changes fastest then izq.
                    const size_t trace_in_sample = static_cast<size_t>(izq) * static_cast<size_t>(qy_size) + static_cast<size_t>(iyq);
                    const size_t source_trace_index = i * traces_per_sample + trace_in_sample;
                    const size_t source_offset = source_trace_index * trace_stride;

                    const size_t target_offset =
                        (static_cast<size_t>(z * qz_size + izq) * static_cast<size_t>(n_indirect1_raw) +
                         static_cast<size_t>(y * qy_size + iyq)) *
                        trace_stride;

                    std::memcpy(expanded_raw.data() + target_offset,
                                sampled_raw.data() + source_offset,
                                trace_stride * sizeof(float));
                }
            }
        }

        return true;
    }


    bool moving_mean_1d(const float *data, float *new_data, int ndata, int nwindow)
    {
        if (nwindow <= 0 || nwindow > ndata)
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

            float sum = 0.0f;
            for (int j = n1; j <= n2; j++)
            {
                sum += data[j];
            }
            new_data[i] = sum / static_cast<float>(n2 - n1 + 1);
        }

        return true;
    }

    void subtract_signal_1d(float *data, const float *baseline, int n)
    {
        for (int i = 0; i < n; i++)
        {
            data[i] -= baseline[i];
        }
    }

    bool polynomial_baseline_1d(const float *data, int ndata, int order, std::vector<float> &baseline)
    {
        if (order < 0 || order > 4 || ndata <= 0)
        {
            return false;
        }

        const int n_terms = order + 1;
        Eigen::MatrixXd design(ndata, n_terms);
        Eigen::VectorXd values(ndata);

        for (int i = 0; i < ndata; i++)
        {
            const double x = (ndata > 1) ? (2.0 * static_cast<double>(i) / static_cast<double>(ndata - 1) - 1.0) : 0.0;
            double term = 1.0;
            values(i) = static_cast<double>(data[i]);
            for (int j = 0; j < n_terms; j++)
            {
                design(i, j) = term;
                term *= x;
            }
        }

        const Eigen::VectorXd solution = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);
        baseline.assign(ndata, 0.0f);
        for (int i = 0; i < ndata; i++)
        {
            const double x = (ndata > 1) ? (2.0 * static_cast<double>(i) / static_cast<double>(ndata - 1) - 1.0) : 0.0;
            double term = 1.0;
            double fitted = 0.0;
            for (int j = 0; j < n_terms; j++)
            {
                fitted += solution(j) * term;
                term *= x;
            }
            baseline[i] = static_cast<float>(fitted);
        }

        return true;
    }

    double estimate_frq_polynomial_noise_threshold(const float *data,
                                                   int trace_len,
                                                   int segment_size)
    {
        if (data == nullptr || trace_len <= 0 || segment_size <= 0)
        {
            return 0.0;
        }

        const float zero_eps = std::numeric_limits<float>::epsilon() * 10.0f;
        bool all_zero = true;
        for (int k = 0; k < trace_len; ++k)
        {
            if (std::abs(data[k]) > zero_eps)
            {
                all_zero = false;
                break;
            }
        }
        if (all_zero)
        {
            return 0.0;
        }

        std::vector<double> segment_variances;
        for (int i = 0; i + segment_size <= trace_len; i += segment_size)
        {
                const int seg_len = segment_size;
                if (seg_len >= 3)
                {
                    // Fit 2nd-order polynomial (1, x, x^2) on normalized x in [-1,1]
                    Eigen::MatrixXd design(seg_len, 3);
                    Eigen::VectorXd vals(seg_len);
                    for (int j = 0; j < seg_len; ++j)
                    {
                        const double x = (seg_len > 1) ? (2.0 * static_cast<double>(j) / static_cast<double>(seg_len - 1) - 1.0) : 0.0;
                        design(j, 0) = 1.0;
                        design(j, 1) = x;
                        design(j, 2) = x * x;
                        vals(j) = static_cast<double>(data[i + j]);
                    }
                    const Eigen::VectorXd coeffs = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vals);
                    double sq_sum_res = 0.0;
                    for (int j = 0; j < seg_len; ++j)
                    {
                        const double x = (seg_len > 1) ? (2.0 * static_cast<double>(j) / static_cast<double>(seg_len - 1) - 1.0) : 0.0;
                        const double fitted = coeffs(0) + coeffs(1) * x + coeffs(2) * x * x;
                        const double resid = static_cast<double>(data[i + j]) - fitted;
                        sq_sum_res += resid * resid;
                    }
                    const double variance = sq_sum_res / static_cast<double>(seg_len);
                    segment_variances.push_back(std::max(0.0, variance));
                }
                else
                {
                    double sum = 0.0;
                    double sq_sum = 0.0;
                    for (int j = 0; j < segment_size; ++j)
                    {
                        const double val = static_cast<double>(data[i + j]);
                        sum += val;
                        sq_sum += val * val;
                    }

                    const double mean = sum / static_cast<double>(segment_size);
                    const double variance = (sq_sum / static_cast<double>(segment_size)) - (mean * mean);
                    segment_variances.push_back(std::max(0.0, variance));
                }
        }

        if (segment_variances.empty())
        {
            return 0.0;
        }

        std::nth_element(segment_variances.begin(),
                         segment_variances.begin() + segment_variances.size() / 2,
                         segment_variances.end());
        const double median_variance = segment_variances[segment_variances.size() / 2];
        return 4.0 * std::sqrt(median_variance);
    }

    double estimate_frq_polynomial_noise_threshold(const std::vector<const std::vector<float> *> &components,
                                                   int trace_len,
                                                   int segment_size)
    {
        if (trace_len <= 0 || segment_size <= 0)
        {
            return 0.0;
        }

        std::vector<double> segment_variances;
        for (const std::vector<float> *component : components)
        {
            if (component == nullptr || component->empty())
            {
                continue;
            }
            if (component->size() % static_cast<size_t>(trace_len) != 0)
            {
                return 0.0;
            }

            const size_t n_traces = component->size() / static_cast<size_t>(trace_len);
            for (size_t t = 0; t < n_traces; ++t)
            {
                const float *trace = component->data() + t * static_cast<size_t>(trace_len);
                const float zero_eps = std::numeric_limits<float>::epsilon() * 10.0f;
                bool all_zero = true;
                for (int k = 0; k < trace_len; ++k)
                {
                    if (std::abs(trace[k]) > zero_eps)
                    {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero)
                {
                    continue;
                }

                for (int i = 0; i + segment_size <= trace_len; i += segment_size)
                {
                    const int seg_len = segment_size;
                    if (seg_len >= 3)
                    {
                        Eigen::MatrixXd design(seg_len, 3);
                        Eigen::VectorXd vals(seg_len);
                        for (int j = 0; j < seg_len; ++j)
                        {
                            const double x = (seg_len > 1) ? (2.0 * static_cast<double>(j) / static_cast<double>(seg_len - 1) - 1.0) : 0.0;
                            design(j, 0) = 1.0;
                            design(j, 1) = x;
                            design(j, 2) = x * x;
                            vals(j) = static_cast<double>(trace[i + j]);
                        }
                        const Eigen::VectorXd coeffs = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(vals);
                        double sq_sum_res = 0.0;
                        for (int j = 0; j < seg_len; ++j)
                        {
                            const double x = (seg_len > 1) ? (2.0 * static_cast<double>(j) / static_cast<double>(seg_len - 1) - 1.0) : 0.0;
                            const double fitted = coeffs(0) + coeffs(1) * x + coeffs(2) * x * x;
                            const double resid = static_cast<double>(trace[i + j]) - fitted;
                            sq_sum_res += resid * resid;
                        }
                        const double variance = sq_sum_res / static_cast<double>(seg_len);
                        segment_variances.push_back(std::max(0.0, variance));
                    }
                    else
                    {
                        double sum = 0.0;
                        double sq_sum = 0.0;
                        for (int j = 0; j < segment_size; ++j)
                        {
                            const double val = static_cast<double>(trace[i + j]);
                            sum += val;
                            sq_sum += val * val;
                        }

                        const double mean = sum / static_cast<double>(segment_size);
                        const double variance = (sq_sum / static_cast<double>(segment_size)) - (mean * mean);
                        segment_variances.push_back(std::max(0.0, variance));
                    }
                }
            }
        }

        if (segment_variances.empty())
        {
            return 0.0;
        }

        std::nth_element(segment_variances.begin(),
                         segment_variances.begin() + segment_variances.size() / 2,
                         segment_variances.end());
        const double median_variance = segment_variances[segment_variances.size() / 2];
        std::cout << "Estimated noise variance for frequency-domain polynomial baseline fitting: "
                  << std::sqrt(median_variance) << std::endl;
        return 4.0 * std::sqrt(median_variance);
    }

    bool compute_frq_polynomial_baseline_1d(const float *data,
                                            int ndata,
                                            int order,
                                            std::vector<float> &baseline,
                                            double threshold,
                                            int segment_size,
                                            std::vector<char> *selected_mask = nullptr)
    {
        if (order < 0 || order > 4 || ndata <= 0 || segment_size <= 0)
        {
            return false;
        }

        if (threshold < 0.0)
        {
            threshold = estimate_frq_polynomial_noise_threshold(data, ndata, segment_size);
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
                const double resid = static_cast<double>(data[start + r]) - mean;
                sq_sum += resid * resid;
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

        if (valid_indices.size() < static_cast<size_t>(order + 1))
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

        if (valid_indices.size() < static_cast<size_t>(order + 1))
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

        const int n_terms = order + 1;
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

        const Eigen::VectorXd solution = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);

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

    bool apply_frq_polynomial_baseline_to_buffer(std::vector<float> &component,
                                                 int trace_len,
                                                 int order,
                                                 double threshold,
                                                 int segment_size)
    {
        if (component.empty() || trace_len <= 0)
        {
            return true;
        }
        if (component.size() % static_cast<size_t>(trace_len) != 0)
        {
            return false;
        }

        std::vector<float> baseline;
        for (size_t offset = 0; offset < component.size(); offset += static_cast<size_t>(trace_len))
        {
            const float zero_eps = std::numeric_limits<float>::epsilon() * 10.0f;
            bool all_zero = true;
            for (int i = 0; i < trace_len; ++i)
            {
                if (std::abs(component[offset + static_cast<size_t>(i)]) > zero_eps)
                {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero)
            {
                continue;
            }

            if (!compute_frq_polynomial_baseline_1d(component.data() + offset,
                                                    trace_len,
                                                    order,
                                                    baseline,
                                                    threshold,
                                                    segment_size,
                                                    nullptr))
            {
                return false;
            }
            subtract_signal_1d(component.data() + offset, baseline.data(), trace_len);
        }

        return true;
    }

    int next_power_of_2_int(int n)
    {
        int p = 1;
        while (p < n)
        {
            p *= 2;
        }
        return p;
    }

    bool parse_indexed_filename_range(const std::string &first_file,
                                      const std::string &last_file,
                                      std::string &path_name,
                                      std::string &prefix,
                                      std::string &extension,
                                      int &start_idx,
                                      int &end_idx,
                                      int &digit_width)
    {
        std::string path1, file1, ext1;
        std::string path2, file2, ext2;
        if (!ldw_math_spectrum_1d::SplitFilename(first_file, path1, file1, ext1) || !ldw_math_spectrum_1d::SplitFilename(last_file, path2, file2, ext2))
        {
            return false;
        }

        if (path1 != path2 || ext1 != ext2)
        {
            std::cerr << "Error: first and last filenames must share the same path and extension." << std::endl;
            return false;
        }

        auto split_prefix_and_index = [](const std::string &name, std::string &out_prefix, std::string &out_digits) -> bool
        {
            const size_t last_nondigit = name.find_last_not_of("0123456789");
            if (last_nondigit == std::string::npos)
            {
                out_prefix = "";
                out_digits = name;
            }
            else
            {
                out_prefix = name.substr(0, last_nondigit + 1);
                out_digits = name.substr(last_nondigit + 1);
            }
            return !out_digits.empty();
        };

        std::string prefix1, digits1, prefix2, digits2;
        if (!split_prefix_and_index(file1, prefix1, digits1) || !split_prefix_and_index(file2, prefix2, digits2))
        {
            std::cerr << "Error: cannot parse numeric suffix from filenames." << std::endl;
            return false;
        }
        if (prefix1 != prefix2)
        {
            std::cerr << "Error: first and last filenames must share the same prefix." << std::endl;
            return false;
        }

        path_name = path1;
        prefix = prefix1;
        extension = ext1;
        start_idx = std::stoi(digits1);
        end_idx = std::stoi(digits2);
        digit_width = static_cast<int>(digits1.size());

        if (end_idx < start_idx)
        {
            std::swap(start_idx, end_idx);
        }

        return true;
    }

    bool build_indexed_filename(const std::string &path_name,
                                const std::string &prefix,
                                const std::string &extension,
                                int idx,
                                int digit_width,
                                std::string &fname)
    {
        std::ostringstream oss;
        oss << path_name << "/" << prefix << std::setw(digit_width) << std::setfill('0') << idx;
        if (!extension.empty())
        {
            oss << "." << extension;
        }
        fname = oss.str();
        return true;
    }

    bool fft_1d_trace(int n_time,
                      int n_freq,
                      const float *in_real,
                      const float *in_imag,
                      float *out_real,
                      float *out_imag,
                      bool b_swap,
                      bool b_remove_filter = false,
                      double grpdly = -1.0)
    {
        kiss_fft_cpx *cx_in = new kiss_fft_cpx[n_freq];
        kiss_fft_cpx *cx_out = new kiss_fft_cpx[n_freq];

        for (int i = 0; i < n_time; i++)
        {
            if (b_swap && (i % 2 == 1))
            {
                cx_in[i].r = -in_real[i];
                cx_in[i].i = -in_imag[i];
            }
            else
            {
                cx_in[i].r = in_real[i];
                cx_in[i].i = in_imag[i];
            }
        }
        for (int i = n_time; i < n_freq; i++)
        {
            cx_in[i].r = 0.0f;
            cx_in[i].i = 0.0f;
        }

        if (kiss_fft_cfg cfg = kiss_fft_alloc(n_freq, 0, NULL, NULL))
        {
            kiss_fft(cfg, cx_in, cx_out);
            free(cfg);
        }
        else
        {
            std::cerr << "Error: cannot allocate kiss_fft configuration." << std::endl;
            delete[] cx_in;
            delete[] cx_out;
            return false;
        }

        std::vector<float> tmp_real(n_freq, 0.0f);
        std::vector<float> tmp_imag(n_freq, 0.0f);
        const float scale = std::sqrt(static_cast<float>(n_freq));
        for (int i = 0; i < n_freq; i++)
        {
            tmp_real[i] = cx_out[i].r / scale;
            tmp_imag[i] = cx_out[i].i / scale;
        }

        if (b_remove_filter && grpdly > 0.0)
        {
            for (int i = 0; i < n_freq; i++)
            {
                const float phase = static_cast<float>(2.0 * M_PI * grpdly * i / n_freq);
                const float cos_phase = std::cos(phase);
                const float sin_phase = std::sin(phase);

                const float real = tmp_real[i];
                const float imag = tmp_imag[i];

                tmp_real[i] = real * cos_phase - imag * sin_phase;
                tmp_imag[i] = real * sin_phase + imag * cos_phase;
            }
        }

        std::vector<float> swapped_real(n_freq, 0.0f);
        std::vector<float> swapped_imag(n_freq, 0.0f);
        for (int i = 1; i <= n_freq; i++)
        {
            swapped_real[i - 1] = tmp_real[(i + n_freq / 2) % n_freq];
            swapped_imag[i - 1] = tmp_imag[(i + n_freq / 2) % n_freq];
        }

        for (int i = 0; i < n_freq; i++)
        {
            out_real[i] = swapped_real[n_freq - i - 1];
            out_imag[i] = swapped_imag[n_freq - i - 1];
        }

        delete[] cx_in;
        delete[] cx_out;
        return true;
    }

    void phase_correction_trace(int n_freq, float *spectrum_real, float *spectrum_imag, double p0, double p1)
    {
        for (int j = 0; j < n_freq; j++)
        {
            float phase = p0 + p1 * j / n_freq;
            float cos_phase = std::cos(phase * M_PI / 180.0);
            float sin_phase = std::sin(phase * M_PI / 180.0);

            float real = spectrum_real[j];
            float imag = spectrum_imag[j];

            spectrum_real[j] = real * cos_phase - imag * sin_phase;
            spectrum_imag[j] = real * sin_phase + imag * cos_phase;
        }
    }

    float estimate_noise_mad(const std::vector<float> &data)
    {
        if (data.empty())
        {
            return 0.0f;
        }

        std::vector<float> abs_data(data.size(), 0.0f);
        for (size_t i = 0; i < data.size(); i++)
        {
            abs_data[i] = std::abs(data[i]);
        }
        const size_t mid = abs_data.size() / 2;
        std::nth_element(abs_data.begin(), abs_data.begin() + mid, abs_data.end());
        return abs_data[mid] * 1.4826f;
    }

    template <typename T>
    std::vector<T> transpose_buffer_3d(const std::vector<T> &src,
                                       int dim0,
                                       int dim1,
                                       int dim2,
                                       int axis_a,
                                       int axis_b)
    {
        std::vector<T> dst(src.size());
        if (src.empty())
        {
            return dst;
        }

        if (axis_a == axis_b || axis_a < 0 || axis_a > 2 || axis_b < 0 || axis_b > 2)
        {
            return src;
        }

        const int dims[3] = {dim0, dim1, dim2};
        if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0)
        {
            return src;
        }

        const int new_dims[3] = {
            (axis_a == 0 ? dims[axis_b] : axis_b == 0 ? dims[axis_a]
                                                      : dims[0]),
            (axis_a == 1 ? dims[axis_b] : axis_b == 1 ? dims[axis_a]
                                                      : dims[1]),
            (axis_a == 2 ? dims[axis_b] : axis_b == 2 ? dims[axis_a]
                                                      : dims[2])};

        const size_t plane_stride = static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]);
        for (size_t idx = 0; idx < src.size(); ++idx)
        {
            const int x = static_cast<int>(idx % dims[0]);
            const int y = static_cast<int>((idx / dims[0]) % dims[1]);
            const int z = static_cast<int>(idx / plane_stride);

            int coords[3] = {x, y, z};
            std::swap(coords[axis_a], coords[axis_b]);

            const size_t new_idx = static_cast<size_t>(coords[0]) +
                                   static_cast<size_t>(new_dims[0]) *
                                       (static_cast<size_t>(coords[1]) + static_cast<size_t>(new_dims[1]) * static_cast<size_t>(coords[2]));
            dst[new_idx] = src[idx];
        }

        return dst;
    }

    bool read_numeric_file_as_float(const std::string &fname,
                                    FID_DATA_TYPE data_type,
                                    size_t expected_points,
                                    std::vector<float> &out)
    {
        FILE *fp = fopen(fname.c_str(), "rb");
        if (!fp)
        {
            std::cerr << "Error: cannot open file " << fname << std::endl;
            return false;
        }

        out.clear();
        if (data_type == FID_DATA_TYPE_INT32)
        {
            std::vector<int32_t> temp(expected_points, 0);
            const size_t n_read = fread(temp.data(), sizeof(int32_t), expected_points, fp);
            if (n_read != expected_points)
            {
                fclose(fp);
                std::cerr << "Error: short read in file " << fname
                          << ". Expected " << expected_points << " int32 values, got " << n_read << std::endl;
                return false;
            }
            out.resize(expected_points, 0.0f);
            for (size_t i = 0; i < expected_points; i++)
            {
                out[i] = static_cast<float>(temp[i]);
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT64)
        {
            std::vector<double> temp(expected_points, 0.0);
            const size_t n_read = fread(temp.data(), sizeof(double), expected_points, fp);
            if (n_read != expected_points)
            {
                fclose(fp);
                std::cerr << "Error: short read in file " << fname
                          << ". Expected " << expected_points << " float64 values, got " << n_read << std::endl;
                return false;
            }
            out.resize(expected_points, 0.0f);
            for (size_t i = 0; i < expected_points; i++)
            {
                out[i] = static_cast<float>(temp[i]);
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT32)
        {
            out.resize(expected_points, 0.0f);
            const size_t n_read = fread(out.data(), sizeof(float), expected_points, fp);
            if (n_read != expected_points)
            {
                fclose(fp);
                std::cerr << "Error: short read in file " << fname
                          << ". Expected " << expected_points << " float32 values, got " << n_read << std::endl;
                return false;
            }
        }
        else
        {
            fclose(fp);
            std::cerr << "Error: unsupported Bruker data type" << std::endl;
            return false;
        }

        fclose(fp);
        return true;
    }

    void apply_fnmode6_trace_pairs(std::vector<float> &raw,
                                   int n_direct,
                                   int n_indirect1,
                                   int n_indirect2)
    {
        if (n_direct <= 0 || n_indirect1 <= 1 || n_indirect2 <= 0)
        {
            return;
        }

        std::vector<float> diff(n_direct, 0.0f);
        std::vector<float> sum(n_direct, 0.0f);
        std::vector<float> rotated(n_direct, 0.0f);
        const size_t trace_stride = static_cast<size_t>(n_direct);

        for (int iz = 0; iz < n_indirect2; ++iz)
        {
            for (int iy = 0; iy + 1 < n_indirect1; iy += 2)
            {
                float *trace0 = raw.data() + (static_cast<size_t>(iz) * n_indirect1 + static_cast<size_t>(iy)) * trace_stride;
                float *trace1 = trace0 + trace_stride;

                for (int ix = 0; ix < n_direct; ++ix)
                {
                    diff[ix] = 0.5f * (trace0[ix] - trace1[ix]);
                    sum[ix] = 0.5f * (trace0[ix] + trace1[ix]);
                }

                for (int ix = 0; ix < n_direct; ix += 2)
                {
                    if (ix + 1 < n_direct)
                    {
                        rotated[ix] = -sum[ix + 1];
                        rotated[ix + 1] = sum[ix];
                    }
                }

                std::memcpy(trace0, diff.data(), static_cast<size_t>(n_direct) * sizeof(float));
                std::memcpy(trace1, rotated.data(), static_cast<size_t>(n_direct) * sizeof(float));
            }
        }
    }

    void apply_fnmode6_plane_pairs(std::vector<float> &raw,
                                   int n_direct,
                                   int n_indirect1,
                                   int n_indirect2)
    {
        if (n_direct <= 0 || n_indirect1 <= 0 || n_indirect2 <= 1)
        {
            return;
        }

        std::vector<float> diff(n_direct, 0.0f);
        std::vector<float> sum(n_direct, 0.0f);
        std::vector<float> rotated(n_direct, 0.0f);
        const size_t plane_stride = static_cast<size_t>(n_indirect1) * static_cast<size_t>(n_direct);

        for (int iz = 0; iz + 1 < n_indirect2; iz += 2)
        {
            float *plane0 = raw.data() + static_cast<size_t>(iz) * plane_stride;
            float *plane1 = plane0 + plane_stride;

            for (int iy = 0; iy < n_indirect1; ++iy)
            {
                float *trace0 = plane0 + static_cast<size_t>(iy) * n_direct;
                float *trace1 = plane1 + static_cast<size_t>(iy) * n_direct;

                for (int ix = 0; ix < n_direct; ++ix)
                {
                    diff[ix] = 0.5f * (trace0[ix] - trace1[ix]);
                    sum[ix] = 0.5f * (trace0[ix] + trace1[ix]);
                }

                for (int ix = 0; ix < n_direct; ix += 2)
                {
                    if (ix + 1 < n_direct)
                    {
                        rotated[ix] = -sum[ix + 1];
                        rotated[ix + 1] = sum[ix];
                    }
                }

                std::memcpy(trace0, diff.data(), static_cast<size_t>(n_direct) * sizeof(float));
                std::memcpy(trace1, rotated.data(), static_cast<size_t>(n_direct) * sizeof(float));
            }
        }
    }

    bool unpack_bruker_raw_to_components(const std::vector<float> &raw,
                                         int n_direct_raw,
                                         int n_indirect1_raw,
                                         int n_indirect2_raw,
                                         std::vector<float> &spectrum_rrr,
                                         std::vector<float> &spectrum_rri,
                                         std::vector<float> &spectrum_rir,
                                         std::vector<float> &spectrum_rii,
                                         std::vector<float> &spectrum_irr,
                                         std::vector<float> &spectrum_iri,
                                         std::vector<float> &spectrum_iir,
                                         std::vector<float> &spectrum_iii)
    {
        if ((n_direct_raw % 2) != 0 || (n_indirect1_raw % 2) != 0 || (n_indirect2_raw % 2) != 0)
        {
            std::cerr << "Error: Bruker raw dimensions must be even for complex 3D data." << std::endl;
            return false;
        }

        const int n_direct = n_direct_raw / 2;
        const int n_indirect1 = n_indirect1_raw / 2;
        const int n_indirect2 = n_indirect2_raw / 2;
        const size_t total_points = static_cast<size_t>(n_direct) * static_cast<size_t>(n_indirect1) * static_cast<size_t>(n_indirect2);
        const size_t expected_raw = static_cast<size_t>(n_direct_raw) * static_cast<size_t>(n_indirect1_raw) * static_cast<size_t>(n_indirect2_raw);

        if (raw.size() != expected_raw)
        {
            std::cerr << "Error: unexpected Bruker raw payload size. Expected " << expected_raw
                      << " points, got " << raw.size() << std::endl;
            return false;
        }

        spectrum_rrr.assign(total_points, 0.0f);
        spectrum_rri.assign(total_points, 0.0f);
        spectrum_rir.assign(total_points, 0.0f);
        spectrum_rii.assign(total_points, 0.0f);
        spectrum_irr.assign(total_points, 0.0f);
        spectrum_iri.assign(total_points, 0.0f);
        spectrum_iir.assign(total_points, 0.0f);
        spectrum_iii.assign(total_points, 0.0f);

        const size_t raw_plane_stride = static_cast<size_t>(n_indirect1_raw) * static_cast<size_t>(n_direct_raw);
        const size_t raw_trace_stride = static_cast<size_t>(n_direct_raw);
        const size_t out_trace_stride = static_cast<size_t>(n_direct);

        for (int iz = 0; iz < n_indirect2; ++iz)
        {
            for (int iy = 0; iy < n_indirect1; ++iy)
            {
                const size_t raw_base = (static_cast<size_t>(2 * iz) * n_indirect1_raw + static_cast<size_t>(2 * iy)) * raw_trace_stride;
                const size_t out_base = (static_cast<size_t>(iz) * n_indirect1 + static_cast<size_t>(iy)) * out_trace_stride;

                for (int ix = 0; ix < n_direct; ++ix)
                {
                    const size_t src = raw_base + static_cast<size_t>(2 * ix);
                    const size_t dst = out_base + static_cast<size_t>(ix);

                    spectrum_rrr[dst] = raw[src];
                    spectrum_rri[dst] = raw[src + 1];
                    spectrum_rir[dst] = raw[src + raw_trace_stride];
                    spectrum_rii[dst] = raw[src + raw_trace_stride + 1];
                    spectrum_irr[dst] = raw[src + raw_plane_stride];
                    spectrum_iri[dst] = raw[src + raw_plane_stride + 1];
                    spectrum_iir[dst] = raw[src + raw_plane_stride + raw_trace_stride];
                    spectrum_iii[dst] = raw[src + raw_plane_stride + raw_trace_stride + 1];
                }
            }
        }

        return true;
    }

} // namespace fid_3d_aux

bool fid_3d::apply_direct_extraction_after_direct_fft()
{
    if (current_axis_order[0] != 0)
    {
        std::cerr << "Error: direct extraction requires direct axis in inner position." << std::endl;
        return false;
    }

    if (extraction_use_ppm)
    {
        const double stop_ppm = origin_frequency / observed_frequency;
        double begin_ppm = stop_ppm + spectral_width / observed_frequency;
        const double step_ppm = (stop_ppm - begin_ppm) / ndata_frq;
        begin_ppm += step_ppm;

        const int pos1 = static_cast<int>((extraction_from_ppm - begin_ppm) / step_ppm + 0.5);
        const int pos2 = static_cast<int>((extraction_to_ppm - begin_ppm) / step_ppm + 0.5);
        extraction_from_pos = std::min(pos1, pos2);
        extraction_to_pos = std::max(pos1, pos2);
    }
    else
    {
        extraction_from_pos = static_cast<int>(ndata_frq * extraction_from + 0.5);
        extraction_to_pos = static_cast<int>(ndata_frq * extraction_to + 0.5);
    }

    if (extraction_from_pos < 0)
    {
        extraction_from_pos = 0;
    }
    if (extraction_to_pos > ndata_frq)
    {
        extraction_to_pos = ndata_frq;
    }

    if (((extraction_to_pos - extraction_from_pos) % 2) != 0)
    {
        if (extraction_to_pos < ndata_frq)
        {
            extraction_to_pos += 1;
        }
        else if (extraction_from_pos > 0)
        {
            extraction_from_pos -= 1;
        }
    }

    if (extraction_from_pos >= extraction_to_pos)
    {
        std::cerr << "Error: invalid extraction range. index [" << extraction_from_pos << ", " << extraction_to_pos
                  << ") in [0, " << ndata_frq << ")." << std::endl;
        return false;
    }

    if (extraction_from_pos == 0 && extraction_to_pos == ndata_frq)
    {
        return true;
    }

    const int old_n_direct = ndata_frq;
    const int new_n_direct = extraction_to_pos - extraction_from_pos;
    const auto dims = current_position_lengths();
    const int n_indirect1 = dims[1];
    const int n_indirect2 = dims[2];
    const size_t n_traces = static_cast<size_t>(n_indirect1) * static_cast<size_t>(n_indirect2);

    auto crop_component = [&](std::vector<float> &component) -> bool
    {
        const size_t expected = n_traces * static_cast<size_t>(old_n_direct);
        if (component.size() != expected)
        {
            return false;
        }

        std::vector<float> cropped(n_traces * static_cast<size_t>(new_n_direct), 0.0f);
        for (size_t t = 0; t < n_traces; ++t)
        {
            const size_t src_base = t * static_cast<size_t>(old_n_direct) + static_cast<size_t>(extraction_from_pos);
            const size_t dst_base = t * static_cast<size_t>(new_n_direct);
            std::memcpy(cropped.data() + dst_base, component.data() + src_base, static_cast<size_t>(new_n_direct) * sizeof(float));
        }
        component.swap(cropped);
        return true;
    };

    if (!crop_component(spectrum_rrr) ||
        !crop_component(spectrum_rri) ||
        !crop_component(spectrum_rir) ||
        !crop_component(spectrum_rii) ||
        !crop_component(spectrum_irr) ||
        !crop_component(spectrum_iri) ||
        !crop_component(spectrum_iir) ||
        !crop_component(spectrum_iii))
    {
        std::cerr << "Error: direct extraction failed due to unexpected component size." << std::endl;
        return false;
    }

    const double before_sw = spectral_width;
    const double before_orig = origin_frequency;
    const int before_center = n_center;
    const double before_stop_ppm = before_orig / observed_frequency;
    double before_begin_ppm = before_stop_ppm + before_sw / observed_frequency;
    const double before_step_ppm = (before_stop_ppm - before_begin_ppm) / old_n_direct;
    before_begin_ppm += before_step_ppm;

    origin_frequency = carrier_frequency - spectral_width / old_n_direct * (extraction_to_pos - n_center);
    n_center -= extraction_from_pos;
    spectral_width = spectral_width * (static_cast<double>(new_n_direct) / static_cast<double>(old_n_direct));

    ndata_frq = new_n_direct;

    stop1 = origin_frequency / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency;
    step1 = (stop1 - begin1) / ndata_frq;
    begin1 += step1;

    const double after_sw = spectral_width;
    const double after_orig = origin_frequency;
    const int after_center = n_center;
    const double after_begin_ppm = begin1;
    const double after_stop_ppm = stop1;

    std::cout << "Extraction direct dimension: index [" << extraction_from_pos << ", " << extraction_to_pos
              << "), ndata_frq " << old_n_direct << " -> " << new_n_direct << std::endl;
    std::cout << "Direct metadata after extraction: "
              << "SW " << before_sw << " -> " << after_sw
              << ", ORIG " << before_orig << " -> " << after_orig
              << ", CENTER " << before_center << " -> " << after_center
              << ", ppm [" << before_begin_ppm << ", " << before_stop_ppm << "] -> ["
              << after_begin_ppm << ", " << after_stop_ppm << "]" << std::endl;
#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(spectrum_rrr, "spectrum_rrr after direct extraction");
#endif
    return true;
}

fid_3d::fid_3d()
{
    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = false;

    fnmode2 = 0;
    fnmode3 = 0;

    grpdly = -1.0;

    indirect1_p0 = 0.0;
    indirect1_p1 = 0.0;
    indirect2_p0 = 0.0;
    indirect2_p1 = 0.0;

    user_p0_direct = 0.0;
    user_p1_direct = 0.0;
    user_p0_indirect1 = 0.0;
    user_p1_indirect1 = 0.0;
    user_p0_indirect2 = 0.0;
    user_p1_indirect2 = 0.0;

    aqseq = "321";

    b_negative = false;

    b_frq = false;
    b_frq_indirect1 = false;
    b_frq_indirect2 = false;

    b_imaginary = true;
    b_imaginary_indirect = true;
    b_imaginary_indirect2 = true;
    keep_imag_direct_step = true;
    keep_imag_indirect1_step = true;
    keep_imag_indirect2_step = true;
    time_domain_polynorminal_order = {-1, -1, -1};
    frq_domain_polynorminal_order = {-1, -1, -1};
    final_frq_polynorminal_order = {-1, -1, -1};
    segment_size = 8;
    b_inverse_indirect1 = true;
    b_inverse_indirect2 = false;

    extraction_use_ppm = false;
    extraction_from = 0.0;
    extraction_to = 1.0;
    extraction_from_ppm = 0.0;
    extraction_to_ppm = 0.0;
    extraction_from_pos = 0;
    extraction_to_pos = 0;

    n_center = 0;
    n_center_indirect1 = 0;
    n_center_indirect2 = 0;

    apodization_code = 0;
    apodization_code_indirect1 = 0;
    apodization_code_indirect2 = 0;

    zf = 1;
    zf_indirect1 = 1;
    zf_indirect2 = 1;
    use_nus_input = false;
    nus_input_is_inflated = false;
    expected_bruker_raw_points = 0;
    bruker_nus_pairs.clear();
    simulate_nus_mode = false;
    simulate_nus_pairs.clear();
    current_axis_order = {0, 1, 2};

    apodization_direct = nullptr;
    apodization_indirect1 = nullptr;
    apodization_indirect2 = nullptr;

    noise_level = 0.0f;
    noise_level_for_nus = false;

    spect = nullptr;
}

fid_3d::~fid_3d()
{
}

int fid_3d::current_axis_length(int position) const
{
    if (position < 0 || position > 2)
    {
        return 0;
    }

    const int physical_axis = current_axis_order[position];
    switch (physical_axis)
    {
    case 0:
        return b_frq ? ndata_frq : ndata;
    case 1:
        return b_frq_indirect1 ? ndata_frq_indirect1 : ndata_indirect1;
    case 2:
        return b_frq_indirect2 ? ndata_frq_indirect2 : ndata_indirect2;
    default:
        return 0;
    }
}

std::array<int, 3> fid_3d::current_position_lengths() const
{
    return {current_axis_length(0), current_axis_length(1), current_axis_length(2)};
}

bool fid_3d::read_phase_correction(std::string fname)
{
    std::ifstream infile(fname);
    if (!infile.good())
    {
        std::cout << "Error: cannot open phase correction file: " << fname << std::endl;
        return false;
    }

    std::string contents = std::string((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
    infile.close();
    return read_phase_correction_from_string(contents);
}

bool fid_3d::read_phase_correction_from_string(const std::string &phase_correction_text)
{
    std::istringstream iss(phase_correction_text);
    if (!(iss >> user_p0_direct >> user_p1_direct >> user_p0_indirect1 >> user_p1_indirect1 >> user_p0_indirect2 >> user_p1_indirect2))
    {
        std::cout << "Error: phase correction string must contain 6 numbers" << std::endl;
        return false;
    }

    std::cout << "User provided phase correction values: "
              << user_p0_direct << " " << user_p1_direct << " "
              << user_p0_indirect1 << " " << user_p1_indirect1 << " "
              << user_p0_indirect2 << " " << user_p1_indirect2 << std::endl;
    return true;
}

bool fid_3d::extract_region(double from, double to)
{
    if (from < 0 || from >= 1 || to <= 0 || to > 1 || from >= to)
    {
        std::cerr << "Error: from and to must be in [0,1) and from<to." << std::endl;
        return false;
    }
    extraction_use_ppm = false;
    extraction_from = from;
    extraction_to = to;
    return true;
}

bool fid_3d::extract_region_ppm(double from_ppm, double to_ppm)
{
    if (std::isnan(from_ppm) || std::isnan(to_ppm) || std::isinf(from_ppm) || std::isinf(to_ppm) || from_ppm == to_ppm)
    {
        std::cerr << "Error: invalid ppm extraction range." << std::endl;
        return false;
    }
    extraction_use_ppm = true;
    extraction_from_ppm = from_ppm;
    extraction_to_ppm = to_ppm;
    return true;
}

bool fid_3d::apply_nus_simulation_mask(const std::vector<std::array<int, 2>> &nus_pairs_list)
{
    if (!simulate_nus_mode || nus_pairs_list.empty())
    {
        return true;
    }

    // Create a set for fast lookup of allowed (iy, iz) pairs
    std::set<std::pair<int, int>> allowed_pairs;
    for (const auto &pair : nus_pairs_list)
    {
        allowed_pairs.insert({pair[0], pair[1]});
    }

    if (spectrum_rrr.empty())
    {
        std::cerr << "Error: apply_nus_simulation_mask called before spectrum data was populated" << std::endl;
        return false;
    }

    const int n_direct = ndata;
    const int n_indirect1 = ndata_indirect1;
    const int n_indirect2 = ndata_indirect2;
    const size_t trace_stride = static_cast<size_t>(n_direct);

    const int total_positions = n_indirect1 * n_indirect2;
    int zeroed_positions = 0;

    auto zero_component = [&](std::vector<float> &component)
    {
        for (int iz = 0; iz < n_indirect2; ++iz)
        {
            for (int iy = 0; iy < n_indirect1; ++iy)
            {
                const bool is_allowed = allowed_pairs.count({iy, iz}) > 0;
                if (!is_allowed)
                {
                    const size_t base_offset = static_cast<size_t>(iz) * n_indirect1 * trace_stride +
                                               static_cast<size_t>(iy) * trace_stride;
                    std::fill(component.data() + base_offset,
                              component.data() + base_offset + trace_stride, 0.0f);
                }
            }
        }
    };

    // Count zeroed positions from first component
    for (int iz = 0; iz < n_indirect2; ++iz)
    {
        for (int iy = 0; iy < n_indirect1; ++iy)
        {
            if (allowed_pairs.count({iy, iz}) == 0)
            {
                zeroed_positions++;
            }
        }
    }

    // Apply mask to all 8 components
    zero_component(spectrum_rrr);
    zero_component(spectrum_rri);
    zero_component(spectrum_rir);
    zero_component(spectrum_rii);
    zero_component(spectrum_irr);
    zero_component(spectrum_iri);
    zero_component(spectrum_iir);
    zero_component(spectrum_iii);

    std::cout << "NUS simulation mask applied: kept " << allowed_pairs.size()
              << " positions, zeroed " << zeroed_positions
              << " positions out of " << total_positions << std::endl;

    return true;
}

// ============================================================================
// Private FID Processing Pipeline Helpers
// ============================================================================

bool fid_3d::read_time_domain_planes(const std::string &path_name,
                                     const std::string &prefix,
                                     const std::string &file_ext,
                                     int start_idx,
                                     int end_idx,
                                     int digit_width)
{
    const int n_planes = end_idx - start_idx + 1;
    const int n_direct = ndata;
    const int n_indirect1 = ndata_indirect1;
    const size_t plane_trace_stride = static_cast<size_t>(n_direct);
    const size_t plane_complex_block = static_cast<size_t>(n_indirect1) * n_direct;
    const size_t total_points = static_cast<size_t>(n_direct) * n_indirect1 * ndata_indirect2;

    // Allocate all 8 complex components
    spectrum_rrr.assign(total_points, 0.0f);
    spectrum_rri.assign(total_points, 0.0f);
    spectrum_rir.assign(total_points, 0.0f);
    spectrum_rii.assign(total_points, 0.0f);
    spectrum_irr.assign(total_points, 0.0f);
    spectrum_iri.assign(total_points, 0.0f);
    spectrum_iir.assign(total_points, 0.0f);
    spectrum_iii.assign(total_points, 0.0f);

    std::vector<float> trace_buffer(n_direct, 0.0f);

    for (int p = 0; p < n_planes; p++)
    {
        const int current_idx = start_idx + p;
        std::string current_fname;
        fid_3d_aux::build_indexed_filename(path_name, prefix, file_ext, current_idx, digit_width, current_fname);

        FILE *fp = fopen(current_fname.c_str(), "rb");
        if (!fp)
        {
            std::cerr << "Error: Cannot open " << current_fname << std::endl;
            return false;
        }

        fseek(fp, 512 * sizeof(float), SEEK_SET);

        bool z_imag_component = false;
        int iz = p;
        if (b_imaginary_indirect2)
        {
            // Interleaved order:
            // file1=Z1 real, file2=Z1 imag, file3=Z2 real, file4=Z2 imag, ...
            z_imag_component = (p % 2 == 1);
            iz = p / 2;
        }
        if (iz < 0 || iz >= ndata_indirect2)
        {
            fclose(fp);
            continue;
        }

        for (int iy = 0; iy < n_indirect1; iy++)
        {
            const size_t base_offset = static_cast<size_t>(iz) * plane_complex_block + static_cast<size_t>(iy) * plane_trace_stride;

            auto read_trace = [&](std::vector<float> &target) -> bool
            {
                const size_t got = fread(trace_buffer.data(), sizeof(float), n_direct, fp);
                if (got != static_cast<size_t>(n_direct))
                {
                    return false;
                }
                std::memcpy(target.data() + base_offset, trace_buffer.data(), n_direct * sizeof(float));
                return true;
            };

            if (!z_imag_component)
            {
                if (!read_trace(spectrum_rrr))
                {
                    fclose(fp);
                    std::cerr << "Error: short read of RRR trace in " << current_fname << std::endl;
                    return false;
                }
                if (b_imaginary)
                {
                    if (!read_trace(spectrum_rri))
                    {
                        fclose(fp);
                        std::cerr << "Error: short read of RRI trace in " << current_fname << std::endl;
                        return false;
                    }
                }
                if (b_imaginary_indirect)
                {
                    if (!read_trace(spectrum_rir))
                    {
                        fclose(fp);
                        std::cerr << "Error: short read of RIR trace in " << current_fname << std::endl;
                        return false;
                    }
                }
                if (b_imaginary && b_imaginary_indirect)
                {
                    if (!read_trace(spectrum_rii))
                    {
                        fclose(fp);
                        std::cerr << "Error: short read of RII trace in " << current_fname << std::endl;
                        return false;
                    }
                }
            }
            else
            {
                if (!read_trace(spectrum_irr))
                {
                    fclose(fp);
                    std::cerr << "Error: short read of IRR trace in " << current_fname << std::endl;
                    return false;
                }
                if (b_imaginary)
                {
                    if (!read_trace(spectrum_iri))
                    {
                        fclose(fp);
                        std::cerr << "Error: short read of IRI trace in " << current_fname << std::endl;
                        return false;
                    }
                }
                if (b_imaginary_indirect)
                {
                    if (!read_trace(spectrum_iir))
                    {
                        fclose(fp);
                        std::cerr << "Error: short read of IIR trace in " << current_fname << std::endl;
                        return false;
                    }
                }
                if (b_imaginary && b_imaginary_indirect)
                {
                    if (!read_trace(spectrum_iii))
                    {
                        fclose(fp);
                        std::cerr << "Error: short read of III trace in " << current_fname << std::endl;
                        return false;
                    }
                }
            }
        }

        fclose(fp);
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;

    return true;
}

bool fid_3d::read_bruker_folder(std::string folder_name)
{
    struct stat sb;
    if (stat(folder_name.c_str(), &sb) != 0 || !(sb.st_mode & S_IFDIR))
    {
        std::cerr << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }

    std::string fid_file_name = folder_name + "/ser";
    if (stat(fid_file_name.c_str(), &sb) != 0)
    {
        fid_file_name = folder_name + "/fid";
        if (stat(fid_file_name.c_str(), &sb) != 0)
        {
            std::cerr << "Error: cannot find ser or fid file in folder " << folder_name << std::endl;
            return false;
        }
    }

    const std::string acqus_file_name = folder_name + "/acqus";
    const std::string acqu2s_file_name = folder_name + "/acqu2s";
    const std::string acqu3s_file_name = folder_name + "/acqu3s";

    if (stat(acqus_file_name.c_str(), &sb) != 0)
    {
        std::cerr << "Error: cannot find acqus file in folder " << folder_name << std::endl;
        return false;
    }
    if (stat(acqu2s_file_name.c_str(), &sb) != 0)
    {
        std::cerr << "Error: cannot find acqu2s file in folder " << folder_name << std::endl;
        return false;
    }
    if (stat(acqu3s_file_name.c_str(), &sb) != 0)
    {
        std::cerr << "Error: cannot find acqu3s file in folder " << folder_name << std::endl;
        return false;
    }

    return read_bruker_files("", acqu2s_file_name, acqu3s_file_name, acqus_file_name, fid_file_name);
}

bool fid_3d::read_bruker_files(const std::string &pulse_program_name,
                               const std::string &acqus_file2_name,
                               const std::string &acqus_file3_name,
                               const std::string &acqus_file_name,
                               const std::string &fid_data_file_name,
                               const std::string &nuslist_contents,
                               bool simulate_nus,
                               bool nus_ser_inflated)
{
    simulate_nus_mode = simulate_nus;
    std::string contents_pulse;
    if (!pulse_program_name.empty())
    {
        std::ifstream pulse_infile(pulse_program_name);
        if (pulse_infile.good())
        {
            contents_pulse.assign((std::istreambuf_iterator<char>(pulse_infile)), std::istreambuf_iterator<char>());
        }
    }

    std::ifstream acqus_infile(acqus_file_name);
    std::ifstream acqu2s_infile(acqus_file2_name);
    std::ifstream acqu3s_infile(acqus_file3_name);
    if (!acqus_infile.good() || !acqu2s_infile.good() || !acqu3s_infile.good())
    {
        std::cerr << "Error: cannot read acqus/acqu2s/acqu3s file" << std::endl;
        return false;
    }

    const std::string contents_acqus((std::istreambuf_iterator<char>(acqus_infile)), std::istreambuf_iterator<char>());
    const std::string contents_acqus2((std::istreambuf_iterator<char>(acqu2s_infile)), std::istreambuf_iterator<char>());
    const std::string contents_acqus3((std::istreambuf_iterator<char>(acqu3s_infile)), std::istreambuf_iterator<char>());

    if (!read_bruker_files_as_strings(contents_pulse,
                                      contents_acqus,
                                      contents_acqus2,
                                      contents_acqus3,
                                      nuslist_contents,
                                      simulate_nus,
                                      nus_ser_inflated))
    {
        return false;
    }

    if (fid_data_file_name.empty())
    {
        std::cerr << "Error: missing Bruker raw data file" << std::endl;
        return false;
    }

    std::vector<float> raw_buffer;
    if (!fid_3d_aux::read_numeric_file_as_float(fid_data_file_name, data_type, expected_bruker_raw_points, raw_buffer))
    {
        return false;
    }

    return read_bruker_fid_data(raw_buffer);
}

bool fid_3d::read_bruker_files_as_strings(const std::string &contents_pulse,
                                          const std::string &contents_acqus,
                                          const std::string &contents_acqus2,
                                          const std::string &contents_acqus3,
                                          const std::string &contents_nuslist,
                                          bool simulate_nus,
                                          bool nus_ser_inflated)
{
    simulate_nus_mode = simulate_nus;
    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = false;
    b_frq = false;
    b_frq_indirect1 = false;
    b_frq_indirect2 = false;

    udict_acqus_direct.clear();
    udict_acqus_indirect1.clear();
    udict_acqus_indirect2.clear();
    indirect1_p0 = 0.0;
    indirect1_p1 = 0.0;
    indirect2_p0 = 0.0;
    indirect2_p1 = 0.0;

    use_nus_input = false;
    nus_input_is_inflated = nus_ser_inflated;
    expected_bruker_raw_points = 0;
    simulate_nus_pairs.clear();
    bruker_nus_pairs.clear();

    // IMPORTANT: NUS input is enabled only for the real NUS path.
    // When simulate_nus is true, we still parse the nuslist contents, but use_nus_input stays FALSE
    // so the loader reads the data as fully sampled and applies masking only after unpacking.
    if (!contents_nuslist.empty() && contents_nuslist != "none" && contents_nuslist != "null" && contents_nuslist != "no")
    {
        std::vector<std::array<int, 2>> parsed_nus_pairs;
        if (!fid_3d_aux::read_nus_pairs_from_string(contents_nuslist, parsed_nus_pairs))
        {
            std::cerr << "Error: failed to parse NUS list content" << std::endl;
            return false;
        }

        if (simulate_nus)
        {
            simulate_nus_pairs = std::move(parsed_nus_pairs);
            std::cout << "Read " << simulate_nus_pairs.size() << " 3D NUS points for simulation from string payload" << std::endl;
        }
        else
        {
            bruker_nus_pairs = std::move(parsed_nus_pairs);
            use_nus_input = true;
            std::cout << "Read " << bruker_nus_pairs.size() << " 3D NUS points from string payload" << std::endl;
        }
    }

    std::istringstream pulse_stream(contents_pulse);
    std::string line;
    while (std::getline(pulse_stream, line))
    {
        if (line.find(";PHC0(F2):") != std::string::npos)
        {
            indirect1_p0 = std::stod(line.substr(10));
        }
        else if (line.find(";PHC1(F2):") != std::string::npos)
        {
            indirect1_p1 = std::stod(line.substr(10));
        }
        else if (line.find(";PHC0(F1):") != std::string::npos)
        {
            indirect2_p0 = std::stod(line.substr(10));
        }
        else if (line.find(";PHC1(F1):") != std::string::npos)
        {
            indirect2_p1 = std::stod(line.substr(10));
        }
    }

    if (!process_jcamp_as_string(contents_acqus, udict_acqus_direct) ||
        !process_jcamp_as_string(contents_acqus2, udict_acqus_indirect1) ||
        !process_jcamp_as_string(contents_acqus3, udict_acqus_indirect2))
    {
        std::cerr << "Error: cannot parse acqus/acqu2s/acqu3s content" << std::endl;
        return false;
    }

    if (udict_acqus_direct.find("AQ_mod") == udict_acqus_direct.end() ||
        (udict_acqus_direct["AQ_mod"] != "1" && udict_acqus_direct["AQ_mod"] != "3"))
    {
        std::cerr << "Error: Bruker 3D loader requires complex direct data" << std::endl;
        return false;
    }

    data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    if (udict_acqus_direct.find("DTYPA") != udict_acqus_direct.end() && udict_acqus_direct["DTYPA"] == "2")
    {
        data_type = FID_DATA_TYPE_FLOAT64;
    }
    else
    {
        data_type = FID_DATA_TYPE_INT32;
    }

    std::string indirect_size_key = "TD";
    if (use_nus_input)
    {
        if (udict_acqus_indirect1.find("NUSTD") != udict_acqus_indirect1.end() &&
            udict_acqus_indirect2.find("NUSTD") != udict_acqus_indirect2.end())
        {
            indirect_size_key = "NUSTD";
        }
        else if (udict_acqus_indirect1.find("NusTD") != udict_acqus_indirect1.end() &&
                 udict_acqus_indirect2.find("NusTD") != udict_acqus_indirect2.end())
        {
            indirect_size_key = "NusTD";
        }
        else
        {
            std::cerr << "Error: cannot find NUSTD (or NusTD) in acqu2s/acqu3s while NUS is enabled" << std::endl;
            return false;
        }
    }

    if (udict_acqus_direct.find("TD") == udict_acqus_direct.end() ||
        udict_acqus_indirect1.find(indirect_size_key) == udict_acqus_indirect1.end() ||
        udict_acqus_indirect2.find(indirect_size_key) == udict_acqus_indirect2.end())
    {
        std::cerr << "Error: cannot find " << indirect_size_key << " in one of acqus/acqu2s/acqu3s" << std::endl;
        return false;
    }

    ndata_bruker = std::stoi(udict_acqus_direct["TD"]);
    const int indirect1_header_size = std::stoi(udict_acqus_indirect1[indirect_size_key]);
    const int indirect2_header_size = std::stoi(udict_acqus_indirect2[indirect_size_key]);

    if (ndata_bruker <= 0 || indirect1_header_size <= 0 || indirect2_header_size <= 0)
    {
        std::cerr << "Error: invalid Bruker size values in headers" << std::endl;
        return false;
    }

    if (data_type == FID_DATA_TYPE_INT32)
    {
        ndata_bruker = static_cast<int>(std::ceil(static_cast<double>(ndata_bruker) / 256.0)) * 256;
    }
    else
    {
        ndata_bruker = static_cast<int>(std::ceil(static_cast<double>(ndata_bruker) / 128.0)) * 128;
    }

    if (udict_acqus_direct.find("GRPDLY") != udict_acqus_direct.end())
    {
        grpdly = std::stod(udict_acqus_direct["GRPDLY"]);
        if (grpdly <= 0.0)
        {
            grpdly = -1.0;
        }
    }

    if (udict_acqus_direct.find("SW_h") == udict_acqus_direct.end() ||
        udict_acqus_direct.find("SFO1") == udict_acqus_direct.end() ||
        udict_acqus_direct.find("O1") == udict_acqus_direct.end() ||
        udict_acqus_indirect1.find("SW_h") == udict_acqus_indirect1.end() ||
        udict_acqus_indirect1.find("SFO1") == udict_acqus_indirect1.end() ||
        udict_acqus_indirect1.find("O1") == udict_acqus_indirect1.end() ||
        udict_acqus_indirect2.find("SW_h") == udict_acqus_indirect2.end() ||
        udict_acqus_indirect2.find("SFO1") == udict_acqus_indirect2.end() ||
        udict_acqus_indirect2.find("O1") == udict_acqus_indirect2.end())
    {
        std::cerr << "Error: missing SW_h/SFO1/O1 in one of the Bruker JCAMP files" << std::endl;
        return false;
    }

    spectral_width = std::stod(udict_acqus_direct["SW_h"]);
    observed_frequency = std::stod(udict_acqus_direct["SFO1"]);
    carrier_frequency = std::stod(udict_acqus_direct["O1"]);

    spectral_width_indirect1 = std::stod(udict_acqus_indirect1["SW_h"]);
    observed_frequency_indirect1 = std::stod(udict_acqus_indirect1["SFO1"]);
    carrier_frequency_indirect1 = std::stod(udict_acqus_indirect1["O1"]);

    spectral_width_indirect2 = std::stod(udict_acqus_indirect2["SW_h"]);
    observed_frequency_indirect2 = std::stod(udict_acqus_indirect2["SFO1"]);
    carrier_frequency_indirect2 = std::stod(udict_acqus_indirect2["O1"]);

    fnmode2 = 0;
    fnmode3 = 0;
    if (udict_acqus_indirect1.find("FnMODE") != udict_acqus_indirect1.end())
    {
        fnmode2 = std::stoi(udict_acqus_indirect1["FnMODE"]);
    }
    if (udict_acqus_indirect2.find("FnMODE") != udict_acqus_indirect2.end())
    {
        fnmode3 = std::stoi(udict_acqus_indirect2["FnMODE"]);
    }

    ndata_bruker_indirect1 = indirect1_header_size;
    ndata_bruker_indirect2 = indirect2_header_size;

    if ((ndata_bruker % 2) != 0 || (ndata_bruker_indirect1 % 2) != 0 || (ndata_bruker_indirect2 % 2) != 0)
    {
        std::cerr << "Error: Bruker raw dimensions must be even for complex 3D data" << std::endl;
        return false;
    }

    ndata = ndata_bruker / 2;
    ndata_indirect1 = ndata_bruker_indirect1 / 2;
    ndata_indirect2 = ndata_bruker_indirect2 / 2;

    if (use_nus_input)
    {
        for (const auto &pair : bruker_nus_pairs)
        {
            if (pair[0] < 0 || pair[1] < 0 || pair[0] >= ndata_indirect1 || pair[1] >= ndata_indirect2)
            {
                std::cerr << "Error: nuslist coordinate out of inflated NusTD bounds: y="
                          << pair[0] << ", z=" << pair[1]
                          << " for grid " << ndata_indirect1 << "x" << ndata_indirect2 << std::endl;
                return false;
            }
        }
    }

    n_center = static_cast<int>(ndata / 2.0f) + 1;
    n_center_indirect1 = static_cast<int>(ndata_indirect1 / 2.0f) + 1;
    n_center_indirect2 = static_cast<int>(ndata_indirect2 / 2.0f) + 1;

    nucleus = udict_acqus_direct["NUC1"];
    nucleus_indirect1 = udict_acqus_indirect1["NUC1"];
    nucleus_indirect2 = udict_acqus_indirect2["NUC1"];

    origin_frequency = carrier_frequency - spectral_width * (ndata - (ndata / 2.0f + 1.0f)) / ndata;
    origin_frequency_indirect1 = carrier_frequency_indirect1 - spectral_width_indirect1 * (ndata_indirect1 - (ndata_indirect1 / 2.0f + 1.0f)) / ndata_indirect1;
    origin_frequency_indirect2 = carrier_frequency_indirect2 - spectral_width_indirect2 * (ndata_indirect2 - (ndata_indirect2 / 2.0f + 1.0f)) / ndata_indirect2;

    ndata_power_of_2 = fid_3d_aux::next_power_of_2_int(ndata);
    ndata_power_of_2_indirect1 = fid_3d_aux::next_power_of_2_int(ndata_indirect1);
    ndata_power_of_2_indirect2 = fid_3d_aux::next_power_of_2_int(ndata_indirect2);

    b_imaginary = true;
    b_imaginary_indirect = true;
    b_imaginary_indirect2 = true;
    current_axis_order = {0, 1, 2};

    if (use_nus_input && !nus_input_is_inflated)
    {
        expected_bruker_raw_points = static_cast<size_t>(ndata_bruker) *
                                     bruker_nus_pairs.size() *
                                     static_cast<size_t>(ndata_bruker_indirect1 / ndata_indirect1) *
                                     static_cast<size_t>(ndata_bruker_indirect2 / ndata_indirect2);
    }
    else
    {
        expected_bruker_raw_points = static_cast<size_t>(ndata_bruker) *
                                     static_cast<size_t>(ndata_bruker_indirect1) *
                                     static_cast<size_t>(ndata_bruker_indirect2);
    }

    return true;
}

bool fid_3d::read_bruker_fid_data(const std::vector<float> &fid_data_float_in)
{
    if (ndata_bruker <= 0 || ndata_bruker_indirect1 <= 0 || ndata_bruker_indirect2 <= 0 || expected_bruker_raw_points == 0)
    {
        std::cerr << "Error: invalid Bruker dimensions. Call read_bruker_files_as_strings first." << std::endl;
        return false;
    }

    if (fid_data_float_in.empty() || fid_data_float_in.size() != expected_bruker_raw_points)
    {
        std::cerr << "Error: Bruker float payload size does not match expected dimensions (expected "
                  << expected_bruker_raw_points << ", got " << fid_data_float_in.size() << ")." << std::endl;
        return false;
    }

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(fid_data_float_in, "bruker fid input");
#endif

    std::vector<float> expanded_raw_buffer;
    if (use_nus_input && !nus_input_is_inflated)
    {
        if (!fid_3d_aux::expand_nus_raw_buffer(fid_data_float_in,
                       ndata_bruker,
                       ndata_bruker_indirect1,
                       ndata_bruker_indirect2,
                       bruker_nus_pairs,
                       expanded_raw_buffer))
        {
            return false;
        }
    }
    else
    {
        expanded_raw_buffer = fid_data_float_in;
    }

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(expanded_raw_buffer, "bruker fid expanded raw");
#endif

    if (fnmode2 == 6)
    {
        fid_3d_aux::apply_fnmode6_trace_pairs(expanded_raw_buffer, ndata_bruker, ndata_bruker_indirect1, ndata_bruker_indirect2);
    }
    if (fnmode3 == 6)
    {
        fid_3d_aux::apply_fnmode6_plane_pairs(expanded_raw_buffer, ndata_bruker, ndata_bruker_indirect1, ndata_bruker_indirect2);
    }

    if (!fid_3d_aux::unpack_bruker_raw_to_components(expanded_raw_buffer,
                                         ndata_bruker,
                                         ndata_bruker_indirect1,
                                         ndata_bruker_indirect2,
                                         spectrum_rrr,
                                         spectrum_rri,
                                         spectrum_rir,
                                         spectrum_rii,
                                         spectrum_irr,
                                         spectrum_iri,
                                         spectrum_iir,
                                         spectrum_iii))
    {
        return false;
    }

    if (simulate_nus_mode && !simulate_nus_pairs.empty())
    {
        std::cout << "Simulate-NUS enabled: fully sampled read completed, now zeroing traces not in the nuslist" << std::endl;
        if (!apply_nus_simulation_mask(simulate_nus_pairs))
        {
            return false;
        }
    }

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(spectrum_rrr, "bruker fid spectrum_rrr");
#endif

    b_read_bruker_acqus_and_fid = true;
    b_read_nmrpipe_fid = false;
    b_frq = false;
    b_frq_indirect1 = false;
    b_frq_indirect2 = false;
    spect = spectrum_rrr.data();
    print_axis_summary();
    return true;
}

bool fid_3d::read_bruker_fid_data_bytes(const std::vector<unsigned char> &fid_data_bytes)
{
    if (fid_data_bytes.empty())
    {
        std::cerr << "Error: empty Bruker FID payload" << std::endl;
        return false;
    }

    if (expected_bruker_raw_points == 0)
    {
        std::cerr << "Error: invalid Bruker dimensions. Call read_bruker_files_as_strings first." << std::endl;
        return false;
    }

    std::vector<float> fid_data_float_in;
    fid_data_float_in.resize(expected_bruker_raw_points, 0.0f);

    if (data_type == FID_DATA_TYPE_INT32)
    {
        const size_t expected_bytes = expected_bruker_raw_points * sizeof(int32_t);
        if (fid_data_bytes.size() != expected_bytes)
        {
            std::cerr << "Error: Bruker INT32 payload size mismatch (expected "
                      << expected_bytes << " bytes, got " << fid_data_bytes.size() << ")." << std::endl;
            return false;
        }

        for (size_t i = 0; i < expected_bruker_raw_points; ++i)
        {
            int32_t v = 0;
            std::memcpy(&v, fid_data_bytes.data() + i * sizeof(int32_t), sizeof(int32_t));
            fid_data_float_in[i] = static_cast<float>(v);
        }
    }
    else if (data_type == FID_DATA_TYPE_FLOAT64)
    {
        const size_t expected_bytes = expected_bruker_raw_points * sizeof(double);
        if (fid_data_bytes.size() != expected_bytes)
        {
            std::cerr << "Error: Bruker FLOAT64 payload size mismatch (expected "
                      << expected_bytes << " bytes, got " << fid_data_bytes.size() << ")." << std::endl;
            return false;
        }

        for (size_t i = 0; i < expected_bruker_raw_points; ++i)
        {
            double v = 0.0;
            std::memcpy(&v, fid_data_bytes.data() + i * sizeof(double), sizeof(double));
            fid_data_float_in[i] = static_cast<float>(v);
        }
    }
    else
    {
        std::cerr << "Error: unsupported Bruker data type" << std::endl;
        return false;
    }

    return read_bruker_fid_data(fid_data_float_in);
}

bool fid_3d::read_ft3_from_buffer(const std::vector<unsigned char> &ft3_bytes)
{
    const size_t header_bytes = 512 * sizeof(float);
    if (ft3_bytes.size() < header_bytes)
    {
        std::cerr << "Error: FT3 payload is too small to contain a valid nmrPipe header." << std::endl;
        return false;
    }

    nmrpipe_header_data.assign(512, 0.0f);
    std::memcpy(nmrpipe_header_data.data(), ft3_bytes.data(), header_bytes);

    nmrPipe::nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
    if (!set_varibles_from_nmrpipe_dictionary())
    {
        return false;
    }

    const bool is_frq_data = b_frq && b_frq_indirect1 && b_frq_indirect2;
    const bool is_fid_data = !b_frq && !b_frq_indirect1 && !b_frq_indirect2;
    const bool is_mixed_stage = !is_frq_data && !is_fid_data;
    if (!is_frq_data && !is_mixed_stage)
    {
        std::cerr << "Error: read_ft3_from_buffer expects stacked FT3 data (frequency-domain or mixed-stage)." << std::endl;
        return false;
    }

    const auto sizes = current_position_lengths();
    const int n0 = sizes[0];
    const int n1 = sizes[1];
    const int n2 = sizes[2];
    if (n0 <= 0 || n1 <= 0 || n2 <= 0)
    {
        std::cerr << "Error: invalid dimensions while reading FT3 buffer." << std::endl;
        return false;
    }

    const bool axis_complex[3] = {b_imaginary, b_imaginary_indirect, b_imaginary_indirect2};
    const int c0 = axis_complex[current_axis_order[0]] ? 2 : 1;
    const int c1 = axis_complex[current_axis_order[1]] ? 2 : 1;
    const int c2 = axis_complex[current_axis_order[2]] ? 2 : 1;

    const size_t logical_total = static_cast<size_t>(n0) * static_cast<size_t>(n1) * static_cast<size_t>(n2);
    spectrum_rrr.assign(logical_total, 0.0f);
    spectrum_rri.assign(logical_total, 0.0f);
    spectrum_rir.assign(logical_total, 0.0f);
    spectrum_rii.assign(logical_total, 0.0f);
    spectrum_irr.assign(logical_total, 0.0f);
    spectrum_iri.assign(logical_total, 0.0f);
    spectrum_iir.assign(logical_total, 0.0f);
    spectrum_iii.assign(logical_total, 0.0f);

    auto get_component = [&](int imag_f2, int imag_f1, int imag_f3) -> std::vector<float> *
    {
        const int idx = imag_f2 + 2 * imag_f1 + 4 * imag_f3;
        switch (idx)
        {
        case 0:
            return &spectrum_rrr;
        case 1:
            return &spectrum_rri;
        case 2:
            return &spectrum_rir;
        case 3:
            return &spectrum_rii;
        case 4:
            return &spectrum_irr;
        case 5:
            return &spectrum_iri;
        case 6:
            return &spectrum_iir;
        case 7:
            return &spectrum_iii;
        default:
            return nullptr;
        }
    };

    const size_t trace_bytes = static_cast<size_t>(n0) * sizeof(float);
    const size_t expected_payload_bytes =
        static_cast<size_t>(n0) * static_cast<size_t>(n1) * static_cast<size_t>(n2) *
        static_cast<size_t>(c0) * static_cast<size_t>(c1) * static_cast<size_t>(c2) * sizeof(float);
    if (ft3_bytes.size() != header_bytes + expected_payload_bytes)
    {
        std::cerr << "Error: FT3 payload size mismatch (expected "
                  << (header_bytes + expected_payload_bytes) << " bytes, got "
                  << ft3_bytes.size() << ")." << std::endl;
        return false;
    }

    size_t payload_offset = header_bytes;
    std::vector<float> trace_data(static_cast<size_t>(n0), 0.0f);
    for (int i2 = 0; i2 < n2; ++i2)
    {
        for (int i1 = 0; i1 < n1; ++i1)
        {
            const size_t base = (static_cast<size_t>(i2) * static_cast<size_t>(n1) + static_cast<size_t>(i1)) * static_cast<size_t>(n0);
            for (int q2 = 0; q2 < c2; ++q2)
            {
                for (int q1 = 0; q1 < c1; ++q1)
                {
                    for (int q0 = 0; q0 < c0; ++q0)
                    {
                        std::memcpy(trace_data.data(), ft3_bytes.data() + payload_offset, trace_bytes);
                        payload_offset += trace_bytes;

                        int imag_by_axis[3] = {0, 0, 0};
                        imag_by_axis[current_axis_order[0]] = q0;
                        imag_by_axis[current_axis_order[1]] = q1;
                        imag_by_axis[current_axis_order[2]] = q2;

                        std::vector<float> *component = get_component(
                            imag_by_axis[0],
                            imag_by_axis[1],
                            imag_by_axis[2]);
                        if (component == nullptr)
                        {
                            std::cerr << "Error: invalid component mapping while reading FT3 buffer." << std::endl;
                            return false;
                        }

                        std::memcpy(component->data() + base, trace_data.data(), trace_bytes);
                    }
                }
            }
        }
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;
    spect = spectrum_rrr.data();
    cleanup_unused_components();

    if(noise_level_for_nus == true)
    {
        /**
         * Get maximal abs value of spectrum_rrr and set noise level to 1% of it.
         */
        noise_level = calculate_noise_level_for_nus(); //return 0.0f if spectrum_rrr is empty or all zeros
    }
    
    if(noise_level == 0.0f){
        noise_level = estimate_final_rrr_segment_noise();
    }
    std::cout << "Estimated 3D Noise Level: " << noise_level << std::endl;
    print_axis_summary();

    return true;
}

bool fid_3d::postprocess_loaded_ft3()
{
    if (spectrum_rrr.empty())
    {
        std::cerr << "Error: no FT3 data loaded for post-processing." << std::endl;
        return false;
    }

    auto get_component = [&](int imag_f2, int imag_f1, int imag_f3) -> std::vector<float> *
    {
        const int idx = imag_f2 + 2 * imag_f1 + 4 * imag_f3;
        switch (idx)
        {
        case 0:
            return &spectrum_rrr;
        case 1:
            return &spectrum_rri;
        case 2:
            return &spectrum_rir;
        case 3:
            return &spectrum_rii;
        case 4:
            return &spectrum_irr;
        case 5:
            return &spectrum_iri;
        case 6:
            return &spectrum_iir;
        case 7:
            return &spectrum_iii;
        default:
            return nullptr;
        }
    };

    auto apply_direct_phase_if_needed = [&]() -> bool
    {
        if (!b_frq || !b_imaginary)
        {
            return true;
        }

        int pos = -1;
        for (int p = 0; p < 3; ++p)
        {
            if (current_axis_order[p] == 0)
            {
                pos = p;
                break;
            }
        }
        if (pos == -1)
        {
            std::cerr << "Error: direct axis mapping corrupted while applying FT3 phase correction." << std::endl;
            return false;
        }

        if (pos != 0 && !transpose_dimensions(0, pos))
        {
            return false;
        }

        const auto sizes = current_position_lengths();
        const int n_direct = sizes[0];
        const int n_indirect1 = sizes[1];
        const int n_indirect2 = sizes[2];
        const size_t trace_count = static_cast<size_t>(n_indirect1) * static_cast<size_t>(n_indirect2);

        auto phase_pair = [&](int imag_f1, int imag_f3) -> bool
        {
            std::vector<float> *real_comp = get_component(0, imag_f1, imag_f3);
            std::vector<float> *imag_comp = get_component(1, imag_f1, imag_f3);
            if (real_comp == nullptr || imag_comp == nullptr || real_comp->empty() || imag_comp->empty())
            {
                return true;
            }
            if (real_comp->size() != imag_comp->size())
            {
                return false;
            }

            for (size_t trace = 0; trace < trace_count; ++trace)
            {
                float *trace_real = real_comp->data() + trace * static_cast<size_t>(n_direct);
                float *trace_imag = imag_comp->data() + trace * static_cast<size_t>(n_direct);
                fid_3d_aux::phase_correction_trace(n_direct, trace_real, trace_imag, user_p0_direct, user_p1_direct);
            }
            return true;
        };

        if (!phase_pair(0, 0) || !phase_pair(0, 1) || !phase_pair(1, 0) || !phase_pair(1, 1))
        {
            return false;
        }

        if (pos != 0 && !transpose_dimensions(0, pos))
        {
            return false;
        }

        return true;
    };

    auto apply_axis_frq_baseline = [&](int phys_axis) -> bool
    {
        const int order = frq_domain_polynorminal_order[phys_axis];
        const bool axis_has_frq = (phys_axis == 0) ? b_frq : (phys_axis == 1 ? b_frq_indirect1 : b_frq_indirect2);
        if (order < 0 || !axis_has_frq)
        {
            return true;
        }

        int pos = -1;
        for (int p = 0; p < 3; ++p)
        {
            if (current_axis_order[p] == phys_axis)
            {
                pos = p;
                break;
            }
        }
        if (pos == -1)
        {
            std::cerr << "Error: axis mapping corrupted while applying FT3 frequency baseline." << std::endl;
            return false;
        }

        if (pos != 0 && !transpose_dimensions(0, pos))
        {
            return false;
        }

        const auto sizes = current_position_lengths();
        const int trace_len = sizes[0];
        const size_t trace_count = static_cast<size_t>(sizes[1]) * static_cast<size_t>(sizes[2]);

        int other_axes[2];
        int other_count = 0;
        for (int axis = 0; axis < 3; ++axis)
        {
            if (axis != phys_axis)
            {
                other_axes[other_count++] = axis;
            }
        }

        for (int q0 = 0; q0 < 2; ++q0)
        {
            for (int q1 = 0; q1 < 2; ++q1)
            {
                int bits_real[3] = {0, 0, 0};
                bits_real[phys_axis] = 0;
                bits_real[other_axes[0]] = q0;
                bits_real[other_axes[1]] = q1;

                int bits_imag[3] = {bits_real[0], bits_real[1], bits_real[2]};
                bits_imag[phys_axis] = 1;

                std::vector<float> *comp_real = get_component(bits_real[0], bits_real[1], bits_real[2]);
                std::vector<float> *comp_imag = get_component(bits_imag[0], bits_imag[1], bits_imag[2]);
                if (comp_real == nullptr || comp_imag == nullptr || comp_real->empty() || comp_imag->empty())
                {
                    continue;
                }
                if (comp_real->size() != comp_imag->size())
                {
                    return false;
                }

                for (size_t trace = 0; trace < trace_count; ++trace)
                {
                    float *trace_real = comp_real->data() + trace * static_cast<size_t>(trace_len);
                    float *trace_imag = comp_imag->data() + trace * static_cast<size_t>(trace_len);
                    if (!apply_frq_domain_polynorminal_baseline(trace_real, trace_imag, trace_len, phys_axis))
                    {
                        return false;
                    }
                }
            }
        }

        if (pos != 0 && !transpose_dimensions(0, pos))
        {
            return false;
        }

        return true;
    };

    if (!apply_direct_phase_if_needed())
    {
        return false;
    }

    if (!apply_axis_frq_baseline(0) ||
        !apply_axis_frq_baseline(1) ||
        !apply_axis_frq_baseline(2))
    {
        return false;
    }

    spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();

    if (!apply_final_frq_polynomial_baseline())
    {
        return false;
    }

    if (!keep_imag_direct_step) b_imaginary = false;
    if (!keep_imag_indirect1_step) b_imaginary_indirect = false;
    if (!keep_imag_indirect2_step) b_imaginary_indirect2 = false;
    cleanup_unused_components();

    if(noise_level_for_nus == true)
    {
        /**
         * Get maximal abs value of spectrum_rrr and set noise level to 1% of it.
         */
        noise_level = calculate_noise_level_for_nus(); //return 0.0f if spectrum_rrr is empty or all zeros
    }
    
    if(noise_level == 0.0f){
        noise_level = estimate_final_rrr_segment_noise();
    }
    std::cout << "Estimated 3D Noise Level: " << noise_level << std::endl;

    spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
    return true;
}

bool fid_3d::apply_time_domain_polynorminal_baseline_axis(int axis_id)
{
    if (axis_id < 0 || axis_id > 2)
    {
        return false;
    }

    const int order = time_domain_polynorminal_order[axis_id];
    if (order < 0)
    {
        return true;
    }

    if (order > 4)
    {
        std::cerr << "Error: invalid time-domain polynorminal order=" << order
                  << " for axis " << axis_id << ". Expected -1 or [0,4]." << std::endl;
        return false;
    }

    if (current_axis_order[0] != axis_id)
    {
        std::cerr << "Error: time-domain polynorminal baseline requested on axis " << axis_id
                  << " but current inner axis is " << current_axis_order[0] << std::endl;
        return false;
    }

    const auto dims = current_position_lengths();
    const int n_axis = dims[0];
    const size_t n_traces = static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]);
    const size_t total = static_cast<size_t>(n_axis) * n_traces;

    const int start_index = (axis_id == 0) ? std::max(0, std::min(n_axis - 1, static_cast<int>(grpdly + 0.5))) : 0;
    const int n_effective = n_axis - start_index;

    if (n_effective <= order)
    {
        std::cerr << "Error: time-domain polynorminal order " << order
                  << " is too high for effective axis length " << n_effective
                  << " (axis length " << n_axis << ", start index " << start_index << ")"
                  << std::endl;
        return false;
    }

    const int nwindow = std::min(32, n_effective);
    std::vector<float> smooth_trace(n_effective, 0.0f);
    std::vector<float> baseline(n_effective, 0.0f);
    auto apply_component = [&](std::vector<float> &component) -> bool
    {
        if (component.empty())
        {
            return true;
        }
        if (component.size() != total)
        {
            return false;
        }
        for (size_t t = 0; t < n_traces; ++t)
        {
            float *trace = component.data() + t * static_cast<size_t>(n_axis);
            float *fit_trace = trace + static_cast<size_t>(start_index);
            if (!fid_3d_aux::moving_mean_1d(fit_trace, smooth_trace.data(), n_effective, nwindow))
            {
                return false;
            }
            if (!fid_3d_aux::polynomial_baseline_1d(smooth_trace.data(), n_effective, order, baseline))
            {
                return false;
            }
            fid_3d_aux::subtract_signal_1d(fit_trace, baseline.data(), n_effective);
        }
        return true;
    };

    if (!apply_component(spectrum_rrr) ||
        !apply_component(spectrum_rri) ||
        !apply_component(spectrum_rir) ||
        !apply_component(spectrum_rii) ||
        !apply_component(spectrum_irr) ||
        !apply_component(spectrum_iri) ||
        !apply_component(spectrum_iir) ||
        !apply_component(spectrum_iii))
    {
        std::cerr << "Error: time-domain polynorminal baseline correction failed." << std::endl;
        return false;
    }

    const char *axis_name[3] = {"F2", "F1", "F3"};
    std::cout << "Applied time-domain polynorminal baseline: axis=" << axis_name[axis_id]
              << ", order=" << order
              << ", trace_length=" << n_axis
              << ", start_index=" << start_index
              << ", effective_length=" << n_effective
              << ", moving_mean_window=" << nwindow
              << ", n_traces=" << n_traces << std::endl;

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(spectrum_rrr, std::string("spectrum_rrr after time-domain baseline ") + axis_name[axis_id]);
#endif

    return true;
}

bool fid_3d::apply_frq_domain_polynorminal_baseline(float *trace_real,
                                                    float *trace_imag,
                                                    int n_freq,
                                                    int axis_id,
                                                    double threshold) const
{
    if (trace_real == nullptr || trace_imag == nullptr || n_freq <= 0 || axis_id < 0 || axis_id > 2)
    {
        return false;
    }

    const int order = frq_domain_polynorminal_order[axis_id];
    if (order < 0)
    {
        return true;
    }

    if (order > 4 || n_freq <= order)
    {
        return false;
    }

    std::vector<float> baseline_real(n_freq, 0.0f);
    if (!fid_3d_aux::compute_frq_polynomial_baseline_1d(trace_real, n_freq, order, baseline_real, threshold, segment_size, nullptr))
    {
        return false;
    }

    // Apply baseline subtraction to the real channel only (matches 2D behavior).
    fid_3d_aux::subtract_signal_1d(trace_real, baseline_real.data(), n_freq);
    return true;
}

bool fid_3d::process_direct_dimension()
{
    if (current_axis_order[0] != 0)
    {
        std::cerr << "Error: direct dimension is not in the inner position." << std::endl;
        return false;
    }

    const auto position_lengths = current_position_lengths();
    const int n_direct = position_lengths[0];
    const int n_indirect1 = position_lengths[1];
    const int n_indirect2 = position_lengths[2];

    // Calculate FFT sizes
    ndata_power_of_2 = fid_3d_aux::next_power_of_2_int(ndata);
    ndata_frq = ndata_power_of_2 * zf;
    n_center = static_cast<int>(ndata_frq / 2.0f) + 1;
    std::cout << std::endl;
    std::cout << "Processing direct dimension (F2): size before=" << n_direct
              << ", size after FFT=" << ndata_frq << std::endl;
    if (frq_domain_polynorminal_order[0] >= 0)
    {
        std::cout << "Frequency-domain polynorminal baseline configured on F2 with order "
                  << frq_domain_polynorminal_order[0] << std::endl;
    }
    const size_t total_after_direct = static_cast<size_t>(ndata_frq) * n_indirect1 * n_indirect2;

    // Allocate output buffers for all 8 components
    std::vector<float> d_rrr(total_after_direct, 0.0f), d_rri(total_after_direct, 0.0f),
        d_rir(total_after_direct, 0.0f), d_rii(total_after_direct, 0.0f);
    std::vector<float> d_irr(total_after_direct, 0.0f), d_iri(total_after_direct, 0.0f),
        d_iir(total_after_direct, 0.0f), d_iii(total_after_direct, 0.0f);

    // Setup apodization
    apodization_direct->set_sw(spectral_width);
    apodization_direct->set_n(ndata);

    // Working buffers
    std::vector<float> in_r(ndata, 0.0f), in_i(ndata, 0.0f),
        out_r(ndata_frq, 0.0f), out_i(ndata_frq, 0.0f);
    // Process direct dimension for all (iz, iy) positions with all 4 real/imag pairs
    for (int iz = 0; iz < n_indirect2; iz++)
    {
        for (int iy = 0; iy < n_indirect1; iy++)
        {
            const size_t in_base = (static_cast<size_t>(iz) * n_indirect1 + iy) * n_direct;
            const size_t out_base = (static_cast<size_t>(iz) * n_indirect1 + iy) * ndata_frq;

            // Process all 4 component pairs: (RRR,RRI), (RIR,RII), (IRR,IRI), (IIR,III)
            auto process_component_pair = [&](std::vector<float> &in_comp_r,
                                              std::vector<float> &in_comp_i,
                                              std::vector<float> &out_comp_r,
                                              std::vector<float> &out_comp_i) -> bool
            {
                std::memcpy(in_r.data(), in_comp_r.data() + in_base, n_direct * sizeof(float));
                std::memcpy(in_i.data(), in_comp_i.data() + in_base, n_direct * sizeof(float));
                apodization_direct->run_apodization(in_r.data(), n_direct, false);
                apodization_direct->run_apodization(in_i.data(), n_direct, false);
                fid_3d_aux::fft_1d_trace(n_direct, ndata_frq, in_r.data(), in_i.data(),
                             out_r.data(), out_i.data(), false,
                             true, grpdly);
                fid_3d_aux::phase_correction_trace(ndata_frq, out_r.data(), out_i.data(),
                                       user_p0_direct, user_p1_direct);
                std::memcpy(out_comp_r.data() + out_base, out_r.data(), ndata_frq * sizeof(float));
                std::memcpy(out_comp_i.data() + out_base, out_i.data(), ndata_frq * sizeof(float));
                return true;
            };

            if (!process_component_pair(spectrum_rrr, spectrum_rri, d_rrr, d_rri) ||
                !process_component_pair(spectrum_rir, spectrum_rii, d_rir, d_rii) ||
                !process_component_pair(spectrum_irr, spectrum_iri, d_irr, d_iri) ||
                !process_component_pair(spectrum_iir, spectrum_iii, d_iir, d_iii))
            {
                return false;
            }
        }
    }

    // Swap buffers
    spectrum_rrr.swap(d_rrr);
    spectrum_rri.swap(d_rri);
    spectrum_rir.swap(d_rir);
    spectrum_rii.swap(d_rii);
    spectrum_irr.swap(d_irr);
    spectrum_iri.swap(d_iri);
    spectrum_iir.swap(d_iir);
    spectrum_iii.swap(d_iii);

    if (frq_domain_polynorminal_order[0] >= 0)
    {
        const double direct_threshold = fid_3d_aux::estimate_frq_polynomial_noise_threshold(
            std::vector<const std::vector<float> *>{&spectrum_rrr, &spectrum_rir, &spectrum_irr, &spectrum_iir},
            ndata_frq,
            segment_size);
        if (direct_threshold <= 0.0)
        {
            std::cerr << "Error: direct frequency-domain polynomial baseline threshold estimation failed." << std::endl;
            return false;
        }
        if (!fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rrr, ndata_frq, frq_domain_polynorminal_order[0], direct_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rir, ndata_frq, frq_domain_polynorminal_order[0], direct_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_irr, ndata_frq, frq_domain_polynorminal_order[0], direct_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_iir, ndata_frq, frq_domain_polynorminal_order[0], direct_threshold, segment_size))
        {
            std::cerr << "Error: direct frequency-domain polynomial baseline correction failed." << std::endl;
            return false;
        }
    }
    b_frq = true;

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(spectrum_rrr, "spectrum_rrr after direct FFT+phase");
#endif

    if (!apply_direct_extraction_after_direct_fft())
    {
        return false;
    }

    apodization_code = 1;

    {
        const auto sizes = current_position_lengths();
        const char *axis_name[3] = {"F2", "F1", "F3"};
        std::cout << "Finished direct dimension (F2): size before=" << n_direct
                  << ", size after=" << sizes[0] << std::endl;
        std::cout << "Current dimension sizes after direct FFT: "
                  << sizes[0] << " " << sizes[1] << " " << sizes[2]
                  << " | order=" << axis_name[current_axis_order[0]] << ","
                  << axis_name[current_axis_order[1]] << ","
                  << axis_name[current_axis_order[2]] << std::endl;
        std::cout << "Phase correction after direct FFT (F2, user only): "
                  << "p0=" << user_p0_direct << ", p1=" << user_p1_direct
                  << std::endl;
        // Print grpdly-based P1 contribution if applied
        if (grpdly > 0.0)
        {
            const double grpdly_p1 = 360.0 * grpdly;
            std::cout << "Phase correction from grpdly (F2): "
                      << "p1=" << grpdly_p1
                      << std::endl;
        }
    }

    return true;
}

bool fid_3d::process_indirect1_dimension()
{
    if (current_axis_order[0] != 1)
    {
        std::cerr << "Error: indirect1 dimension is not in the inner position." << std::endl;
        return false;
    }

    const auto position_lengths = current_position_lengths();
    const int n_indirect1 = position_lengths[0];
    const int n_indirect2 = position_lengths[1];
    const int n_direct = position_lengths[2];

    // Calculate FFT sizes
    ndata_power_of_2_indirect1 = fid_3d_aux::next_power_of_2_int(ndata_indirect1);
    ndata_frq_indirect1 = ndata_power_of_2_indirect1 * zf_indirect1;
    std::cout << std::endl;
    std::cout << "Processing indirect1 dimension (F1): size before=" << n_indirect1
              << ", size after FFT=" << ndata_frq_indirect1 << std::endl;
    if (frq_domain_polynorminal_order[1] >= 0)
    {
        std::cout << "Frequency-domain polynorminal baseline configured on F1 with order "
                  << frq_domain_polynorminal_order[1] << std::endl;
    }
    const size_t total_after_indirect1 = static_cast<size_t>(n_direct) * ndata_frq_indirect1 * n_indirect2;

    // Allocate output buffers for all 8 components
    std::vector<float> i1_rrr(total_after_indirect1, 0.0f), i1_rri(total_after_indirect1, 0.0f),
        i1_rir(total_after_indirect1, 0.0f), i1_rii(total_after_indirect1, 0.0f);
    std::vector<float> i1_irr(total_after_indirect1, 0.0f), i1_iri(total_after_indirect1, 0.0f),
        i1_iir(total_after_indirect1, 0.0f), i1_iii(total_after_indirect1, 0.0f);

    // Setup apodization
    apodization_indirect1->set_sw(spectral_width_indirect1);
    apodization_indirect1->set_n(n_indirect1);
    apodization_indirect1->set_first_point(std::abs(user_p1_indirect1) < 2.0 ? 0.5f : 1.0f);

    const size_t total_before_indirect1 = static_cast<size_t>(n_direct) * static_cast<size_t>(n_indirect1) * static_cast<size_t>(n_indirect2);

    // Working buffers
    std::vector<float> y_r(n_indirect1, 0.0f), y_i(n_indirect1, 0.0f),
        y_out_r(ndata_frq_indirect1, 0.0f), y_out_i(ndata_frq_indirect1, 0.0f);

    const bool have_direct_imag = b_imaginary;
    const bool have_indirect1_imag = b_imaginary_indirect;
    const bool have_indirect2_imag = b_imaginary_indirect2;

    // Enforce policy: if direct imaginary was removed, do not carry or process any ??i components.
    if (!have_direct_imag)
    {
        spectrum_rri.clear();
        spectrum_rii.clear();
        spectrum_iri.clear();
        spectrum_iii.clear();
    }

    // Process indirect1 dimension - for each (iz, ix) gather indirect1 traces, FFT, scatter
    for (int iz = 0; iz < n_indirect2; iz++)
    {
        for (int ix = 0; ix < n_direct; ix++)
        {
            const size_t in_base = (static_cast<size_t>(iz) * n_direct + ix) * n_indirect1;
            const size_t out_base = (static_cast<size_t>(iz) * n_direct + ix) * ndata_frq_indirect1;

            auto process_component_pair = [&](const std::vector<float> &in_comp_r,
                                              const std::vector<float> *in_comp_i,
                                              std::vector<float> &out_comp_r,
                                              std::vector<float> *out_comp_i,
                                              bool write_imag) -> bool
            {
                if (in_comp_r.size() != total_before_indirect1)
                {
                    return false;
                }
                std::memcpy(y_r.data(), in_comp_r.data() + in_base, n_indirect1 * sizeof(float));
                if (in_comp_i != nullptr)
                {
                    if (in_comp_i->size() != total_before_indirect1)
                    {
                        return false;
                    }
                    std::memcpy(y_i.data(), in_comp_i->data() + in_base, n_indirect1 * sizeof(float));
                }
                else
                {
                    std::fill(y_i.begin(), y_i.end(), 0.0f);
                }

                // Optional F1 imaginary sign inversion to match acquisition convention.
                if (b_inverse_indirect1)
                {
                    for (int i = 0; i < n_indirect1; ++i)
                    {
                        y_i[i] = -y_i[i];
                    }
                }

                apodization_indirect1->run_apodization(y_r.data(), n_indirect1, false);
                apodization_indirect1->run_apodization(y_i.data(), n_indirect1, false);
                fid_3d_aux::fft_1d_trace(n_indirect1, ndata_frq_indirect1, y_r.data(), y_i.data(),
                             y_out_r.data(), y_out_i.data(), fnmode2 == 5);
                fid_3d_aux::phase_correction_trace(ndata_frq_indirect1, y_out_r.data(), y_out_i.data(),
                                       user_p0_indirect1, user_p1_indirect1);
                std::memcpy(out_comp_r.data() + out_base, y_out_r.data(), ndata_frq_indirect1 * sizeof(float));
                if (write_imag && out_comp_i != nullptr)
                {
                    std::memcpy(out_comp_i->data() + out_base, y_out_i.data(), ndata_frq_indirect1 * sizeof(float));
                }
                return true;
            };

            if (!process_component_pair(spectrum_rrr,
                                        have_indirect1_imag ? &spectrum_rir : nullptr,
                                        i1_rrr,
                                        &i1_rir,
                                        have_indirect1_imag))
            {
                std::cerr << "Error: indirect1 processing failed for RRR/RIR pair." << std::endl;
                return false;
            }

            if (have_direct_imag)
            {
                if (!process_component_pair(spectrum_rri,
                                            have_indirect1_imag ? &spectrum_rii : nullptr,
                                            i1_rri,
                                            &i1_rii,
                                            have_indirect1_imag))
                {
                    std::cerr << "Error: indirect1 processing failed for RRI/RII pair." << std::endl;
                    return false;
                }
            }

            if (have_indirect2_imag)
            {
                if (!process_component_pair(spectrum_irr,
                                            have_indirect1_imag ? &spectrum_iir : nullptr,
                                            i1_irr,
                                            &i1_iir,
                                            have_indirect1_imag))
                {
                    std::cerr << "Error: indirect1 processing failed for IRR/IIR pair." << std::endl;
                    return false;
                }
            }

            if (have_direct_imag && have_indirect2_imag)
            {
                if (!process_component_pair(spectrum_iri,
                                            have_indirect1_imag ? &spectrum_iii : nullptr,
                                            i1_iri,
                                            &i1_iii,
                                            have_indirect1_imag))
                {
                    std::cerr << "Error: indirect1 processing failed for IRI/III pair." << std::endl;
                    return false;
                }
            }
        }
    }

    // Swap/clear buffers according to currently retained imaginary channels.
    spectrum_rrr.swap(i1_rrr);
    if (have_direct_imag)
    {
        spectrum_rri.swap(i1_rri);
    }
    else
    {
        spectrum_rri.clear();
    }
    if (have_indirect1_imag)
    {
        spectrum_rir.swap(i1_rir);
    }
    else
    {
        spectrum_rir.clear();
    }
    if (have_direct_imag && have_indirect1_imag)
    {
        spectrum_rii.swap(i1_rii);
    }
    else
    {
        spectrum_rii.clear();
    }
    if (have_indirect2_imag)
    {
        spectrum_irr.swap(i1_irr);
    }
    else
    {
        spectrum_irr.clear();
    }
    if (have_direct_imag && have_indirect2_imag)
    {
        spectrum_iri.swap(i1_iri);
    }
    else
    {
        spectrum_iri.clear();
    }
    if (have_indirect1_imag && have_indirect2_imag)
    {
        spectrum_iir.swap(i1_iir);
    }
    else
    {
        spectrum_iir.clear();
    }
    if (have_direct_imag && have_indirect1_imag && have_indirect2_imag)
    {
        spectrum_iii.swap(i1_iii);
    }
    else
    {
        spectrum_iii.clear();
    }

    b_frq_indirect1 = true;
    apodization_code_indirect1 = 1;

    if (frq_domain_polynorminal_order[1] >= 0)
    {
        const double indirect1_threshold = fid_3d_aux::estimate_frq_polynomial_noise_threshold(
            std::vector<const std::vector<float> *>{&spectrum_rrr, &spectrum_rri, &spectrum_irr, &spectrum_iri},
            ndata_frq_indirect1,
            segment_size);
        if (indirect1_threshold <= 0.0)
        {
            std::cerr << "Error: indirect1 frequency-domain polynomial baseline threshold estimation failed." << std::endl;
            return false;
        }
        if (!fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rrr, ndata_frq_indirect1, frq_domain_polynorminal_order[1], indirect1_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rri, ndata_frq_indirect1, frq_domain_polynorminal_order[1], indirect1_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_irr, ndata_frq_indirect1, frq_domain_polynorminal_order[1], indirect1_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_iri, ndata_frq_indirect1, frq_domain_polynorminal_order[1], indirect1_threshold, segment_size))
        {
            std::cerr << "Error: indirect1 frequency-domain polynomial baseline correction failed." << std::endl;
            return false;
        }
    }

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(spectrum_rrr, "spectrum_rrr after indirect1 FFT+phase");
#endif

    {
        const auto sizes = current_position_lengths();
        const char *axis_name[3] = {"F2", "F1", "F3"};
        std::cout << "Finished indirect1 dimension (F1): size before=" << n_indirect1
                  << ", size after=" << sizes[0] << std::endl;
        std::cout << "Current dimension sizes after indirect1 FFT: "
                  << sizes[0] << " " << sizes[1] << " " << sizes[2]
                  << " | order=" << axis_name[current_axis_order[0]] << ","
                  << axis_name[current_axis_order[1]] << ","
                  << axis_name[current_axis_order[2]] << std::endl;
        std::cout << "Phase correction after indirect1 FFT (F1): "
                  << "p0=" << user_p0_indirect1 << ", p1=" << user_p1_indirect1 << std::endl;
    }

    return true;
}

bool fid_3d::process_indirect2_dimension()
{
    if (current_axis_order[0] != 2)
    {
        std::cerr << "Error: indirect2 dimension is not in the inner position." << std::endl;
        return false;
    }

    const auto position_lengths = current_position_lengths();
    const int n_indirect2 = position_lengths[0];
    const int n_indirect1 = position_lengths[1];
    const int n_direct = position_lengths[2];

    // Calculate FFT sizes
    ndata_power_of_2_indirect2 = fid_3d_aux::next_power_of_2_int(n_indirect2);
    ndata_frq_indirect2 = ndata_power_of_2_indirect2 * zf_indirect2;
    std::cout << std::endl;
    std::cout << "Processing indirect2 dimension (F3): size before=" << n_indirect2
              << ", size after FFT=" << ndata_frq_indirect2 << std::endl;
    if (frq_domain_polynorminal_order[2] >= 0)
    {
        std::cout << "Frequency-domain polynorminal baseline configured on F3 with order "
                  << frq_domain_polynorminal_order[2] << std::endl;
    }
    const size_t total_after_indirect2 = static_cast<size_t>(n_direct) * n_indirect1 * ndata_frq_indirect2;

    // Allocate output buffers for all 8 components
    std::vector<float> i2_rrr(total_after_indirect2, 0.0f), i2_rri(total_after_indirect2, 0.0f),
        i2_rir(total_after_indirect2, 0.0f), i2_rii(total_after_indirect2, 0.0f);
    std::vector<float> i2_irr(total_after_indirect2, 0.0f), i2_iri(total_after_indirect2, 0.0f),
        i2_iir(total_after_indirect2, 0.0f), i2_iii(total_after_indirect2, 0.0f);

    // Setup apodization
    apodization_indirect2->set_sw(spectral_width_indirect2);
    apodization_indirect2->set_n(n_indirect2);
    apodization_indirect2->set_first_point(std::abs(user_p1_indirect2) < 2.0 ? 0.5f : 1.0f);

    const size_t total_before_indirect2 = static_cast<size_t>(n_direct) * static_cast<size_t>(n_indirect1) * static_cast<size_t>(n_indirect2);

    // Working buffers
    std::vector<float> z_r(n_indirect2, 0.0f), z_i(n_indirect2, 0.0f),
        z_out_r(ndata_frq_indirect2, 0.0f), z_out_i(ndata_frq_indirect2, 0.0f);

    const bool have_direct_imag = b_imaginary;
    const bool have_indirect1_imag = b_imaginary_indirect;
    const bool have_indirect2_imag = b_imaginary_indirect2;

    // Enforce policy: if direct imaginary was removed, do not carry or process any ??i components.
    if (!have_direct_imag)
    {
        spectrum_rri.clear();
        spectrum_rii.clear();
        spectrum_iri.clear();
        spectrum_iii.clear();
    }

    // Process indirect2 dimension - for each (iy, ix) gather indirect2 traces, FFT, scatter
    for (int iy = 0; iy < n_indirect1; iy++)
    {
        for (int ix = 0; ix < n_direct; ix++)
        {
            const size_t trace_base = (static_cast<size_t>(iy) * n_direct + ix) * n_indirect2;
            const size_t out_base = (static_cast<size_t>(iy) * n_direct + ix) * ndata_frq_indirect2;

            auto process_component_pair = [&](const std::vector<float> &in_comp_r,
                                              const std::vector<float> *in_comp_i,
                                              std::vector<float> &out_comp_r,
                                              std::vector<float> *out_comp_i,
                                              bool write_imag) -> bool
            {
                if (in_comp_r.size() != total_before_indirect2)
                {
                    return false;
                }
                std::memcpy(z_r.data(), in_comp_r.data() + trace_base, n_indirect2 * sizeof(float));
                if (in_comp_i != nullptr)
                {
                    if (in_comp_i->size() != total_before_indirect2)
                    {
                        return false;
                    }
                    std::memcpy(z_i.data(), in_comp_i->data() + trace_base, n_indirect2 * sizeof(float));
                }
                else
                {
                    std::fill(z_i.begin(), z_i.end(), 0.0f);
                }
                // Optional F3 imaginary sign inversion to match acquisition convention.
                if (b_inverse_indirect2)
                {
                    for (int i = 0; i < n_indirect2; ++i)
                    {
                        z_i[i] = -z_i[i];
                    }
                }
                apodization_indirect2->run_apodization(z_r.data(), n_indirect2, false);
                apodization_indirect2->run_apodization(z_i.data(), n_indirect2, false);
                fid_3d_aux::fft_1d_trace(n_indirect2, ndata_frq_indirect2, z_r.data(), z_i.data(),
                             z_out_r.data(), z_out_i.data(), fnmode3 == 5);
                fid_3d_aux::phase_correction_trace(ndata_frq_indirect2, z_out_r.data(), z_out_i.data(),
                                       user_p0_indirect2, user_p1_indirect2);
                std::memcpy(out_comp_r.data() + out_base, z_out_r.data(), ndata_frq_indirect2 * sizeof(float));
                if (write_imag && out_comp_i != nullptr)
                {
                    std::memcpy(out_comp_i->data() + out_base, z_out_i.data(), ndata_frq_indirect2 * sizeof(float));
                }
                return true;
            };

            if (!process_component_pair(spectrum_rrr,
                                        have_indirect2_imag ? &spectrum_irr : nullptr,
                                        i2_rrr,
                                        &i2_irr,
                                        have_indirect2_imag))
            {
                std::cerr << "Error: indirect2 processing failed for RRR/IRR pair." << std::endl;
                return false;
            }

            if (have_direct_imag)
            {
                if (!process_component_pair(spectrum_rri,
                                            have_indirect2_imag ? &spectrum_iri : nullptr,
                                            i2_rri,
                                            &i2_iri,
                                            have_indirect2_imag))
                {
                    std::cerr << "Error: indirect2 processing failed for RRI/IRI pair." << std::endl;
                    return false;
                }
            }

            if (have_indirect1_imag)
            {
                if (!process_component_pair(spectrum_rir,
                                            have_indirect2_imag ? &spectrum_iir : nullptr,
                                            i2_rir,
                                            &i2_iir,
                                            have_indirect2_imag))
                {
                    std::cerr << "Error: indirect2 processing failed for RIR/IIR pair." << std::endl;
                    return false;
                }
            }

            if (have_direct_imag && have_indirect1_imag)
            {
                if (!process_component_pair(spectrum_rii,
                                            have_indirect2_imag ? &spectrum_iii : nullptr,
                                            i2_rii,
                                            &i2_iii,
                                            have_indirect2_imag))
                {
                    std::cerr << "Error: indirect2 processing failed for RII/III pair." << std::endl;
                    return false;
                }
            }
        }
    }

    // Swap/clear buffers according to currently retained imaginary channels.
    spectrum_rrr.swap(i2_rrr);
    if (have_direct_imag)
    {
        spectrum_rri.swap(i2_rri);
    }
    else
    {
        spectrum_rri.clear();
    }
    if (have_indirect1_imag)
    {
        spectrum_rir.swap(i2_rir);
    }
    else
    {
        spectrum_rir.clear();
    }
    if (have_direct_imag && have_indirect1_imag)
    {
        spectrum_rii.swap(i2_rii);
    }
    else
    {
        spectrum_rii.clear();
    }
    if (have_indirect2_imag)
    {
        spectrum_irr.swap(i2_irr);
    }
    else
    {
        spectrum_irr.clear();
    }
    if (have_direct_imag && have_indirect2_imag)
    {
        spectrum_iri.swap(i2_iri);
    }
    else
    {
        spectrum_iri.clear();
    }
    if (have_indirect1_imag && have_indirect2_imag)
    {
        spectrum_iir.swap(i2_iir);
    }
    else
    {
        spectrum_iir.clear();
    }
    if (have_direct_imag && have_indirect1_imag && have_indirect2_imag)
    {
        spectrum_iii.swap(i2_iii);
    }
    else
    {
        spectrum_iii.clear();
    }

    b_frq_indirect2 = true;
    apodization_code_indirect2 = 1;

    if (frq_domain_polynorminal_order[2] >= 0)
    {
        const double indirect2_threshold = fid_3d_aux::estimate_frq_polynomial_noise_threshold(
            std::vector<const std::vector<float> *>{&spectrum_rrr, &spectrum_rri, &spectrum_rir, &spectrum_rii},
            ndata_frq_indirect2,
            segment_size);
        if (indirect2_threshold <= 0.0)
        {
            std::cerr << "Error: indirect2 frequency-domain polynomial baseline threshold estimation failed." << std::endl;
            return false;
        }
        if (!fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rrr, ndata_frq_indirect2, frq_domain_polynorminal_order[2], indirect2_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rri, ndata_frq_indirect2, frq_domain_polynorminal_order[2], indirect2_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rir, ndata_frq_indirect2, frq_domain_polynorminal_order[2], indirect2_threshold, segment_size) ||
            !fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rii, ndata_frq_indirect2, frq_domain_polynorminal_order[2], indirect2_threshold, segment_size))
        {
            std::cerr << "Error: indirect2 frequency-domain polynomial baseline correction failed." << std::endl;
            return false;
        }
    }

#if WEBASS_DEBUG
    fid_3d_aux::debug_print_random_samples(spectrum_rrr, "spectrum_rrr after indirect2 FFT+phase");
#endif

    {
        const auto sizes = current_position_lengths();
        const char *axis_name[3] = {"F2", "F1", "F3"};
        std::cout << "Finished indirect2 dimension (F3): size before=" << n_indirect2
                  << ", size after=" << sizes[0] << std::endl;
        std::cout << "Current dimension sizes after indirect2 FFT: "
                  << sizes[0] << " " << sizes[1] << " " << sizes[2]
                  << " | order=" << axis_name[current_axis_order[0]] << ","
                  << axis_name[current_axis_order[1]] << ","
                  << axis_name[current_axis_order[2]] << std::endl;
        std::cout << "Phase correction after indirect2 FFT (F3): "
                  << "p0=" << user_p0_indirect2 << ", p1=" << user_p1_indirect2 << std::endl;
    }

    return true;
}

void fid_3d::cleanup_unused_components()
{
    if (b_imaginary == false)
    {
        spectrum_rri.clear();
        spectrum_rii.clear();
        spectrum_iri.clear();
        spectrum_iii.clear();
    }
    if (b_imaginary_indirect == false)
    {
        spectrum_rir.clear();
        spectrum_rii.clear();
        spectrum_iir.clear();
        spectrum_iii.clear();
    }
    if (b_imaginary_indirect2 == false)
    {
        spectrum_irr.clear();
        spectrum_iri.clear();
        spectrum_iir.clear();
        spectrum_iii.clear();
    }
}

bool fid_3d::transpose_dimensions(int axis_a, int axis_b)
{
    /**
     * @brief Transpose two storage axes of the processed 3D buffers.
     *
     * This routine rearranges the internal storage layout for all eight
     * frequency-domain components (rrr, rri, rir, rii, irr, iri, iir, iii)
     * by swapping the two axis positions `axis_a` and `axis_b` in the
     * current buffer ordering. Axis ids refer to the positions within the
     * current buffer ordering (0..2) rather than fixed physical axes.
     *
     * The function validates inputs, computes the transposed buffers via
     * `fid_3d_aux::transpose_buffer_3d`, replaces the component buffers
     * with their transposed counterparts, updates `current_axis_order`,
     * and refreshes the `spect` pointer to `spectrum_rrr` data.
     *
     * Parameters:
     *  - axis_a: first axis index in [0,2]
     *  - axis_b: second axis index in [0,2]
     *
     * Returns:
     *  - true on success, false on invalid input or on buffer-size mismatch.
     */
    if (axis_a == axis_b || axis_a < 0 || axis_a > 2 || axis_b < 0 || axis_b > 2)
    {
        std::cerr << "Error: transpose_dimensions requires two distinct axis ids in [0, 2]." << std::endl;
        return false;
    }

    const auto dims = current_position_lengths();
    const size_t total = static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]);

    if (total == 0)
    {
        return true;
    }

    auto prepare_transposed = [&](const std::vector<float> &component, std::vector<float> &prepared)
    {
        if (component.empty())
        {
            prepared.clear();
            return true;
        }
        if (component.size() != total)
        {
            return false;
        }
        prepared = fid_3d_aux::transpose_buffer_3d(component, dims[0], dims[1], dims[2], axis_a, axis_b);
        return true;
    };

    std::vector<float> t_rrr, t_rri, t_rir, t_rii, t_irr, t_iri, t_iir, t_iii;
    if (!prepare_transposed(spectrum_rrr, t_rrr) ||
        !prepare_transposed(spectrum_rri, t_rri) ||
        !prepare_transposed(spectrum_rir, t_rir) ||
        !prepare_transposed(spectrum_rii, t_rii) ||
        !prepare_transposed(spectrum_irr, t_irr) ||
        !prepare_transposed(spectrum_iri, t_iri) ||
        !prepare_transposed(spectrum_iir, t_iir) ||
        !prepare_transposed(spectrum_iii, t_iii))
    {
        std::cerr << "Error: transpose_dimensions failed because one or more buffers have unexpected size." << std::endl;
        return false;
    }

    spectrum_rrr.swap(t_rrr);
    spectrum_rri.swap(t_rri);
    spectrum_rir.swap(t_rir);
    spectrum_rii.swap(t_rii);
    spectrum_irr.swap(t_irr);
    spectrum_iri.swap(t_iri);
    spectrum_iir.swap(t_iir);
    spectrum_iii.swap(t_iii);

    std::swap(current_axis_order[axis_a], current_axis_order[axis_b]);

    spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
    return true;
}

bool fid_3d::transpose_dimensions(int axis0, int axis1, int axis2)
{
    /**
     * @brief Apply a 3-axis permutation to the processed 3D buffers.
     *
     * Currently this overload supports only two permutations:
     *  - identity (0,1,2) which is a no-op
     *  - reverse (2,1,0) which is implemented by swapping axis 0 and 2
     *
     * The parameters `axis0, axis1, axis2` describe the desired storage
     * order (inner, middle, outer). The function returns true if the
     * requested permutation is supported and was applied successfully.
     */
    if (axis0 == 0 && axis1 == 1 && axis2 == 2)
    {
        return true;
    }

    if (axis0 == 2 && axis1 == 1 && axis2 == 0)
    {
        return transpose_dimensions(0, 2);
    }

    std::cerr << "Error: transpose_dimensions permutation currently supports only identity or reverse order." << std::endl;
    return false;
}

bool fid_3d::apply_final_frq_polynomial_baseline()
{
    // If no final orders requested, nothing to do.
    if (final_frq_polynorminal_order[0] < 0 && final_frq_polynorminal_order[1] < 0 && final_frq_polynorminal_order[2] < 0)
    {
        return true;
    }

    double threshold =  estimate_final_rrr_segment_noise();
    std::cout <<" in final baseline, estimated noise threshold for RRR component is "<<threshold<<std::endl;
    threshold = 4.0 * threshold; // Apply a safety factor to avoid overfitting noise.


    const auto orig_order = current_axis_order;
    std::cout << "[final baseline] original axis order: ";
    print_current_axis_order();

    for (int phys_axis = 0; phys_axis < 3; ++phys_axis)
    {
        const int order = final_frq_polynorminal_order[phys_axis];
        std::cout<<"Working on physical axis "<<phys_axis<<" with polynomial order "<<order<<std::endl;
        if (order < 0)
            continue;

        // Find the position in the current buffer that corresponds to this physical axis
        int pos = -1;
        for (int p = 0; p < 3; ++p)
        {
            if (current_axis_order[p] == phys_axis)
            {
                pos = p;
                break;
            }
        }
        if (pos == -1)
        {
            std::cerr << "Error: internal axis mapping corrupted while applying final baseline." << std::endl;
            return false;
        }

        // Move the target axis to inner-most (position 0) if needed
        if (pos != 0)
        {
            /**
             * This function will transpose data and update current_axis_order accordingly. 
             */
            if (!transpose_dimensions(0, pos))
                return false;
        }

        print_current_axis_order();
        const int trace_len = current_axis_length(0);
        std::cout<<"Applying final frequency-domain polynomial baseline correction on physical axis "<<phys_axis
                 <<" (current position "<<0<<", trace length "<<trace_len<<") with order "<<order
                 <<" and threshold "<<threshold<<std::endl;

#if WEBASS_DEBUG
        if (phys_axis == 1)
        {
            const auto trace_sizes = current_position_lengths();
            const size_t debug_trace_index = 318*256+142;
            const size_t trace_count = static_cast<size_t>(trace_sizes[1]) * static_cast<size_t>(trace_sizes[2]);
            if (debug_trace_index < trace_count)
            {
                const float *trace_data = spectrum_rrr.data() + debug_trace_index * static_cast<size_t>(trace_len);
                std::vector<float> debug_baseline(trace_len, 0.0f);
                std::vector<char> selected_mask;
                std::cout<<"Running debug code, threshold="<<threshold<<" on trace index "<<debug_trace_index<<" along physical axis "<<phys_axis<<std::endl;
                if (!fid_3d_aux::compute_frq_polynomial_baseline_1d(trace_data,
                                                                   trace_len,
                                                                   order,
                                                                   debug_baseline,
                                                                   threshold,
                                                                   segment_size,
                                                                   &selected_mask))
                {
                    std::cerr << "Error: debug baseline fit failed for trace " << debug_trace_index << std::endl;
                    return false;
                }

                std::ofstream debug_out("final_baseline_trace.txt");
                if (debug_out.good())
                {
                    debug_out << "# trace_index=" << debug_trace_index
                              << " trace_len=" << trace_len
                              << " order=" << order
                              << " threshold=" << threshold << '\n';
                    debug_out << "# columns: point original selected baseline\n";
                    for (int i = 0; i < trace_len; ++i)
                    {
                        debug_out << i << ' '
                                  << trace_data[i] << ' '
                                  << (selected_mask[i] ? 1 : 0) << ' '
                                  << debug_baseline[static_cast<size_t>(i)] << '\n';
                    }
                }
            }
        }
#endif

        if (!fid_3d_aux::apply_frq_polynomial_baseline_to_buffer(spectrum_rrr, trace_len, order, threshold, segment_size))
        {
            std::cerr << "Error: applying final polynomial baseline failed for axis " << phys_axis << std::endl;
            return false;
        }

        // Restore original axis ordering for further iterations
        if (pos != 0)
        {
            if (!transpose_dimensions(0, pos))
                return false;
        }
    }

    // Ensure we restored original order
    if (current_axis_order != orig_order)
    {
        std::cerr << "Error: failed to restore axis order after final baseline." << std::endl;
        std::cerr << " current: ";
        print_current_axis_order();
        std::cerr << " expected: ";
        for (int p = 0; p < 3; ++p)
            std::cerr << orig_order[p] << (p < 2 ? "," : "\n");
        return false;
    }

    std::cout << "[final baseline] restored axis order: ";
    print_current_axis_order();

    return true;
}

bool fid_3d::create_nmrpipe_dictionary(std::map<std::string, std::string> &dict_string,
                                       std::map<std::string, float> &dict_float,
                                       bool b_init) const
{
    if (b_init)
    {
        nmrPipe::create_default_nmrpipe_dictionary(dict_string, dict_float);
    }

    const auto sizes = current_position_lengths();
    // Storage position -> nmrPipe physical-dimension label mapping.
    // Internal physical-axis ids: 0=F2, 1=F1, 2=F3.
    // nmrPipe labels: F2->2, F1->1, F3->3.
    const int storage_to_pipe_dim[3] = {2, 1, 3};

    dict_float["FDPIPEFLAG"] = 1.0; // Indicate that this is an nmrPipe-format dictionary. why not used in 2D case?? TODO: check if this can be used in 2D case as well.

    dict_float["FDDIMCOUNT"] = 3.0f;
    dict_float["FDDIMORDER1"] = static_cast<float>(storage_to_pipe_dim[current_axis_order[0]]);
    dict_float["FDDIMORDER2"] = static_cast<float>(storage_to_pipe_dim[current_axis_order[1]]);
    dict_float["FDDIMORDER3"] = static_cast<float>(storage_to_pipe_dim[current_axis_order[2]]);
    dict_float["FDTRANSPOSED"] = 0.0f;

    // Update spectrum sizes for current storage order (inner, middle, outer).
    dict_float["FDSIZE"] = static_cast<float>(sizes[0]);
    dict_float["FDSPECNUM"] = static_cast<float>(sizes[1]);
    dict_float["FDF3SIZE"] = static_cast<float>(sizes[2]);

    // nmrPipe convention: when storage dim0 and dim1 are both complex,
    // FDSPECNUM is stored as doubled. The read path reverses this rule.
    bool b1, b2;
    switch (current_axis_order[0])
    {
    case 0:
        b1 = b_imaginary;
        break;
    case 1:
        b1 = b_imaginary_indirect;
        break;
    case 2:
        b1 = b_imaginary_indirect2;
        break;
    default:
        b1 = false;
        break;
    }
    switch (current_axis_order[1])
    {
    case 0:
        b2 = b_imaginary;
        break;
    case 1:
        b2 = b_imaginary_indirect;
        break;
    case 2:
        b2 = b_imaginary_indirect2;
        break;
    default:
        b2 = false;
        break;
    }

    if (b1 && b2)
    {
        dict_float["FDSPECNUM"] = static_cast<float>(sizes[1] * 2);
    }

    dict_string["FDF2LABEL"] = nucleus;
    dict_string["FDF1LABEL"] = nucleus_indirect1;
    dict_string["FDF3LABEL"] = nucleus_indirect2;

    dict_float["FDF2SW"] = spectral_width;
    dict_float["FDF2OBS"] = observed_frequency;
    dict_float["FDF2CAR"] = observed_frequency > 0.0 ? static_cast<float>(carrier_frequency / observed_frequency) : 0.0f;

    dict_float["FDF1SW"] = spectral_width_indirect1;
    dict_float["FDF1OBS"] = observed_frequency_indirect1;
    dict_float["FDF1CAR"] = observed_frequency_indirect1 > 0.0 ? static_cast<float>(carrier_frequency_indirect1 / observed_frequency_indirect1) : 0.0f;

    dict_float["FDF3SW"] = spectral_width_indirect2;
    dict_float["FDF3OBS"] = observed_frequency_indirect2;
    dict_float["FDF3CAR"] = observed_frequency_indirect2 > 0.0 ? static_cast<float>(carrier_frequency_indirect2 / observed_frequency_indirect2) : 0.0f;

    dict_float["FDF2ORIG"] = static_cast<float>(origin_frequency);
    dict_float["FDF1ORIG"] = static_cast<float>(origin_frequency_indirect1);
    dict_float["FDF3ORIG"] = static_cast<float>(origin_frequency_indirect2);

    dict_float["FDF2CENTER"] = static_cast<float>(n_center);
    dict_float["FDF1CENTER"] = static_cast<float>(n_center_indirect1);
    dict_float["FDF3CENTER"] = static_cast<float>(n_center_indirect2);

    // Persist Bruker FnMODE values across intermediate nmrPipe files.
    dict_float["FDUSER1"] = static_cast<float>(fnmode2);
    dict_float["FDUSER2"] = static_cast<float>(fnmode3);

    dict_float["FDF2TDSIZE"] = static_cast<float>(ndata);
    dict_float["FDF1TDSIZE"] = static_cast<float>(ndata_indirect1);
    dict_float["FDF3TDSIZE"] = static_cast<float>(ndata_indirect2);

    const int extraction_span = std::max(1, extraction_to_pos - extraction_from_pos);
    dict_float["FDF2APOD"] = static_cast<float>(extraction_span / std::max(1, zf));
    dict_float["FDF1APOD"] = static_cast<float>(ndata_indirect1);
    dict_float["FDF3APOD"] = static_cast<float>(ndata_indirect2);

    dict_float["FDF2APODCODE"] = static_cast<float>(apodization_code);
    dict_float["FDF1APODCODE"] = static_cast<float>(apodization_code_indirect1);
    dict_float["FDF3APODCODE"] = static_cast<float>(apodization_code_indirect2);

    dict_float["FDF2P0"] = static_cast<float>(user_p0_direct);
    dict_float["FDF2P1"] = static_cast<float>(user_p1_direct);
    dict_float["FDF1P0"] = static_cast<float>(user_p0_indirect1);
    dict_float["FDF1P1"] = static_cast<float>(user_p1_indirect1);
    dict_float["FDF3P0"] = static_cast<float>(user_p0_indirect2);
    dict_float["FDF3P1"] = static_cast<float>(user_p1_indirect2);

    if (b_frq && b_frq_indirect1 && b_frq_indirect2)
    {
        dict_float["FDF2FTFLAG"] = 1.0f;
        dict_float["FDF1FTFLAG"] = 1.0f;
        dict_float["FDF3FTFLAG"] = 1.0f;

        dict_float["FDF2FTSIZE"] = static_cast<float>(ndata_frq);
        dict_float["FDF1FTSIZE"] = static_cast<float>(ndata_frq_indirect1);
        dict_float["FDF3FTSIZE"] = static_cast<float>(ndata_frq_indirect2);

        dict_float["FDF2ZF"] = static_cast<float>(-ndata_frq);
        dict_float["FDF1ZF"] = static_cast<float>(-ndata_frq_indirect1);
        dict_float["FDF3ZF"] = static_cast<float>(-ndata_frq_indirect2);
    }
    else
    {
        dict_float["FDF2FTFLAG"] = b_frq ? 1.0f : 0.0f;
        dict_float["FDF1FTFLAG"] = b_frq_indirect1 ? 1.0f : 0.0f;
        dict_float["FDF3FTFLAG"] = b_frq_indirect2 ? 1.0f : 0.0f;

        dict_float["FDF2FTSIZE"] = b_frq ? static_cast<float>(ndata_frq) : 0.0f;
        dict_float["FDF1FTSIZE"] = b_frq_indirect1 ? static_cast<float>(ndata_frq_indirect1) : 0.0f;
        dict_float["FDF3FTSIZE"] = b_frq_indirect2 ? static_cast<float>(ndata_frq_indirect2) : 0.0f;

        dict_float["FDF2ZF"] = b_frq ? static_cast<float>(-ndata_frq) : 0.0f;
        dict_float["FDF1ZF"] = b_frq_indirect1 ? static_cast<float>(-ndata_frq_indirect1) : 0.0f;
        dict_float["FDF3ZF"] = b_frq_indirect2 ? static_cast<float>(-ndata_frq_indirect2) : 0.0f;
    }

    // Quadrature flags are physical-axis properties (F2/F1/F3), independent of storage order.
    dict_float["FDF2QUADFLAG"] = b_imaginary ? 0.0f : 1.0f;
    dict_float["FDF1QUADFLAG"] = b_imaginary_indirect ? 0.0f : 1.0f;
    dict_float["FDF3QUADFLAG"] = b_imaginary_indirect2 ? 0.0f : 1.0f;
    dict_float["FDQUADFLAG"] = (b_imaginary || b_imaginary_indirect || b_imaginary_indirect2) ? 0.0f : 1.0f;

    return true;
}

void fid_3d::print_current_axis_order() const
{
    const char *axis_name[3] = {"F2", "F1", "F3"};
    for (int pos = 0; pos < 3; ++pos)
    {
        const int phys = current_axis_order[pos];
        if (phys >= 0 && phys <= 2)
        {
            std::cout << "pos" << pos << "->" << axis_name[phys];
        }
        else
        {
            std::cout << "pos" << pos << "->?";
        }
        if (pos < 2)
            std::cout << " ";
        else
            std::cout << std::endl;
    }
}

float fid_3d::estimate_final_rrr_segment_noise() const
{
    if (spectrum_rrr.empty())
        return 0.0f;

    const auto sizes = current_position_lengths();
    const int n0 = sizes[0];
    const int n1 = sizes[1];
    const int n2 = sizes[2];

    const int seg = segment_size;
    if (seg <= 0)
    {
        return fid_3d_aux::estimate_noise_mad(spectrum_rrr);
    }
    if (n0 < seg || n1 < seg || n2 < seg)
    {
        return fid_3d_aux::estimate_noise_mad(spectrum_rrr);
    }

    std::vector<float> rms_values;
    const size_t segment_points = static_cast<size_t>(seg) * static_cast<size_t>(seg) * static_cast<size_t>(seg);
    // const bool use_quadratic_fit = segment_points >= 10;

    const bool use_quadratic_fit = false; //debug code. 

    std::vector<double> coord_x;
    std::vector<double> coord_y;
    std::vector<double> coord_z;
    if (use_quadratic_fit)
    {
        coord_x.resize(static_cast<size_t>(seg));
        coord_y.resize(static_cast<size_t>(seg));
        coord_z.resize(static_cast<size_t>(seg));
        for (int i = 0; i < seg; ++i)
        {
            const double coord = (seg > 1) ? (2.0 * static_cast<double>(i) / static_cast<double>(seg - 1) - 1.0) : 0.0;
            coord_x[static_cast<size_t>(i)] = coord;
            coord_y[static_cast<size_t>(i)] = coord;
            coord_z[static_cast<size_t>(i)] = coord;
        }
    }

    Eigen::MatrixXd design;
    Eigen::VectorXd values;
    if (use_quadratic_fit)
    {
        design.resize(static_cast<Eigen::Index>(segment_points), 10);
        values.resize(static_cast<Eigen::Index>(segment_points));
    }

    for (int z = 0; z + seg <= n2; z += seg)
    {
        for (int y = 0; y + seg <= n1; y += seg)
        {
            for (int x = 0; x + seg <= n0; x += seg)
            {
                if (use_quadratic_fit)
                {
                    size_t row = 0;
                    for (int kz = 0; kz < seg; ++kz)
                    {
                        const double z_coord = coord_z[static_cast<size_t>(kz)];
                        for (int ky = 0; ky < seg; ++ky)
                        {
                            const double y_coord = coord_y[static_cast<size_t>(ky)];
                            const size_t base = static_cast<size_t>(x) + static_cast<size_t>(n0) * (static_cast<size_t>(y + ky) + static_cast<size_t>(n1) * static_cast<size_t>(z + kz));
                            for (int kx = 0; kx < seg; ++kx)
                            {
                                const double x_coord = coord_x[static_cast<size_t>(kx)];
                                const double value = static_cast<double>(spectrum_rrr[base + static_cast<size_t>(kx)]);

                                values(static_cast<Eigen::Index>(row)) = value;
                                design(static_cast<Eigen::Index>(row), 0) = 1.0;
                                design(static_cast<Eigen::Index>(row), 1) = x_coord;
                                design(static_cast<Eigen::Index>(row), 2) = y_coord;
                                design(static_cast<Eigen::Index>(row), 3) = z_coord;
                                design(static_cast<Eigen::Index>(row), 4) = x_coord * x_coord;
                                design(static_cast<Eigen::Index>(row), 5) = y_coord * y_coord;
                                design(static_cast<Eigen::Index>(row), 6) = z_coord * z_coord;
                                design(static_cast<Eigen::Index>(row), 7) = x_coord * y_coord;
                                design(static_cast<Eigen::Index>(row), 8) = x_coord * z_coord;
                                design(static_cast<Eigen::Index>(row), 9) = y_coord * z_coord;
                                ++row;
                            }
                        }
                    }

                    if (row != segment_points)
                    {
                        continue;
                    }

                    const Eigen::VectorXd coeffs = design.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(values);

                    double sq_sum = 0.0;
                    for (int kz = 0; kz < seg; ++kz)
                    {
                        const double z_coord = coord_z[static_cast<size_t>(kz)];
                        for (int ky = 0; ky < seg; ++ky)
                        {
                            const double y_coord = coord_y[static_cast<size_t>(ky)];
                            const size_t base = static_cast<size_t>(x) + static_cast<size_t>(n0) * (static_cast<size_t>(y + ky) + static_cast<size_t>(n1) * static_cast<size_t>(z + kz));
                            for (int kx = 0; kx < seg; ++kx)
                            {
                                const double x_coord = coord_x[static_cast<size_t>(kx)];
                                const double fitted =
                                    coeffs(0) +
                                    coeffs(1) * x_coord +
                                    coeffs(2) * y_coord +
                                    coeffs(3) * z_coord +
                                    coeffs(4) * x_coord * x_coord +
                                    coeffs(5) * y_coord * y_coord +
                                    coeffs(6) * z_coord * z_coord +
                                    coeffs(7) * x_coord * y_coord +
                                    coeffs(8) * x_coord * z_coord +
                                    coeffs(9) * y_coord * z_coord;
                                const double resid = static_cast<double>(spectrum_rrr[base + static_cast<size_t>(kx)]) - fitted;
                                sq_sum += resid * resid;
                            }
                        }
                    }

                    const double rmsd = std::sqrt(sq_sum / static_cast<double>(segment_points));
                    rms_values.push_back(static_cast<float>(rmsd));
                }
                else
                {
                    double sum = 0.0;
                    int count = 0;
                    for (int kz = 0; kz < seg; ++kz)
                    {
                        for (int ky = 0; ky < seg; ++ky)
                        {
                            const size_t base = static_cast<size_t>(x) + static_cast<size_t>(n0) * (static_cast<size_t>(y + ky) + static_cast<size_t>(n1) * static_cast<size_t>(z + kz));
                            for (int kx = 0; kx < seg; ++kx)
                            {
                                const float v = spectrum_rrr[base + static_cast<size_t>(kx)];
                                sum += static_cast<double>(v);
                                ++count;
                            }
                        }
                    }
                    if (count == 0)
                        continue;
                    const double mean = sum / static_cast<double>(count);

                    double sq_sum = 0.0;
                    for (int kz = 0; kz < seg; ++kz)
                    {
                        for (int ky = 0; ky < seg; ++ky)
                        {
                            const size_t base = static_cast<size_t>(x) + static_cast<size_t>(n0) * (static_cast<size_t>(y + ky) + static_cast<size_t>(n1) * static_cast<size_t>(z + kz));
                            for (int kx = 0; kx < seg; ++kx)
                            {
                                const float v = spectrum_rrr[base + static_cast<size_t>(kx)];
                                const double diff = static_cast<double>(v) - mean;
                                sq_sum += diff * diff;
                            }
                        }
                    }

                    const double rmsd = std::sqrt(sq_sum / static_cast<double>(count));
                    rms_values.push_back(static_cast<float>(rmsd));
                }
            }
        }
    }

    if (rms_values.empty())
    {
        return fid_3d_aux::estimate_noise_mad(spectrum_rrr);
    }

    /**
     * Debug output of all segments RMS values to a file, for analysis of noise distribution and choice of threshold.
     */
#ifdef FID_3D_DEBUG
    {
        std::ofstream debug_out("final_baseline_segment_rms.txt");
        if (debug_out.good())
        {
            debug_out << "# segment_size=" << segment_size << " total_segments=" << rms_values.size() << '\n';
            debug_out << "# columns: segment_index rms_value\n";
            for (size_t i = 0; i < rms_values.size(); ++i)
            {
                debug_out << i << ' ' << rms_values[i] << '\n';
            }
        }
        debug_out.close();
    }
#endif

    const size_t n = rms_values.size();
    const size_t mid = n / 2;
    std::nth_element(rms_values.begin(), rms_values.begin() + mid, rms_values.end());
    float median = rms_values[mid];
    if ((n % 2) == 0)
    {
        // find max in lower half
        const auto it = std::max_element(rms_values.begin(), rms_values.begin() + mid);
        median = 0.5f * (median + *it);
    }

    return median;
}

bool fid_3d::prepare_header_for_nmrpipe()
{
    const bool init_from_default = b_read_bruker_acqus_and_fid;
    if (!create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, init_from_default))
    {
        return false;
    }

    if (nmrpipe_header_data.size() != 512)
    {
        nmrpipe_header_data.assign(512, 0.0f);
    }

    return nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
}

bool fid_3d::serialize_ft3_bytes(std::vector<unsigned char> &out_bytes)
{
    const auto sizes = current_position_lengths();
    const int n0 = sizes[0];
    const int n1 = sizes[1];
    const int n2 = sizes[2];
    if (n0 <= 0 || n1 <= 0 || n2 <= 0)
    {
        std::cerr << "Error: invalid dimensions in write_ft3 serialization." << std::endl;
        return false;
    }

    const size_t logical_total = static_cast<size_t>(n0) * static_cast<size_t>(n1) * static_cast<size_t>(n2);
    auto check_component_size = [&](const std::vector<float> &component, const char *name) -> bool
    {
        if (component.size() != logical_total)
        {
            std::cerr << "Error: component " << name << " has unexpected size in write_ft3 serialization." << std::endl;
            return false;
        }
        return true;
    };

    std::map<std::string, std::string> dict_string;
    std::map<std::string, float> dict_float;
    if (!create_nmrpipe_dictionary(dict_string, dict_float, true))
    {
        return false;
    }

    nmrpipe_dict_string = dict_string;
    nmrpipe_dict_float = dict_float;
    if (nmrpipe_header_data.size() != 512)
    {
        nmrpipe_header_data.assign(512, 0.0f);
    }
    if (!nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float))
    {
        std::cerr << "Error: failed to build nmrPipe header in write_ft3 serialization." << std::endl;
        return false;
    }

    // Physical-axis complexity flags.
    // axis_complex[0]=F2, axis_complex[1]=F1, axis_complex[2]=F3.
    const bool axis_complex[3] = {b_imaginary, b_imaginary_indirect, b_imaginary_indirect2};
    // Number of quadrature channels to write along each current storage position.
    // c*=2 for complex axis, c*=1 for real-only axis.
    const int c0 = axis_complex[current_axis_order[0]] ? 2 : 1;
    const int c1 = axis_complex[current_axis_order[1]] ? 2 : 1;
    const int c2 = axis_complex[current_axis_order[2]] ? 2 : 1;

    auto get_component = [&](int imag_f2, int imag_f1, int imag_f3) -> const std::vector<float> *
    {
        // Component index is encoded by physical-axis imag bits:
        // idx = imag_f2 + 2*imag_f1 + 4*imag_f3.
        // This keeps component selection independent of current storage order.
        const int idx = imag_f2 + 2 * imag_f1 + 4 * imag_f3;
        switch (idx)
        {
        case 0:
            return &spectrum_rrr;
        case 1:
            return &spectrum_rri;
        case 2:
            return &spectrum_rir;
        case 3:
            return &spectrum_rii;
        case 4:
            return &spectrum_irr;
        case 5:
            return &spectrum_iri;
        case 6:
            return &spectrum_iir;
        case 7:
            return &spectrum_iii;
        default:
            return nullptr;
        }
    };

    const size_t header_bytes = 512 * sizeof(float);
    const size_t trace_bytes = static_cast<size_t>(n0) * sizeof(float);
    const size_t n_traces = static_cast<size_t>(n1) * static_cast<size_t>(n2);
    const size_t n_components_per_trace = static_cast<size_t>(c0) * static_cast<size_t>(c1) * static_cast<size_t>(c2);

    out_bytes.clear();
    out_bytes.reserve(header_bytes + trace_bytes * n_traces * n_components_per_trace);
    out_bytes.resize(header_bytes);
    std::memcpy(out_bytes.data(), nmrpipe_header_data.data(), header_bytes);

    auto append_trace = [&](const float *src)
    {
        const size_t old_size = out_bytes.size();
        out_bytes.resize(old_size + trace_bytes);
        std::memcpy(out_bytes.data() + old_size, src, trace_bytes);
    };

    for (int i2 = 0; i2 < n2; ++i2)
    {
        for (int i1 = 0; i1 < n1; ++i1)
        {
            const size_t base = (static_cast<size_t>(i2) * static_cast<size_t>(n1) + static_cast<size_t>(i1)) * static_cast<size_t>(n0);

            for (int q2 = 0; q2 < c2; ++q2)
            {
                for (int q1 = 0; q1 < c1; ++q1)
                {
                    for (int q0 = 0; q0 < c0; ++q0)
                    {
                        // q0/q1/q2 are imag selectors in storage-position space.
                        // Convert them back to physical-axis imag flags (F2/F1/F3)
                        // before choosing one of {rrr,rri,rir,rii,irr,iri,iir,iii}.
                        int imag_by_axis[3] = {0, 0, 0};
                        imag_by_axis[current_axis_order[0]] = q0;
                        imag_by_axis[current_axis_order[1]] = q1;
                        imag_by_axis[current_axis_order[2]] = q2;

                        const std::vector<float> *component = get_component(
                            imag_by_axis[0],
                            imag_by_axis[1],
                            imag_by_axis[2]);
                        if (component == nullptr || !check_component_size(*component, "component"))
                        {
                            return false;
                        }

                        append_trace(component->data() + base);
                    }
                }
            }
        }
    }

    return true;
}

bool fid_3d::write_ft3(const std::string &out_fname)
{
    if (out_fname.empty())
    {
        std::cerr << "Error: write_ft3 requires a non-empty output filename." << std::endl;
        return false;
    }

    std::vector<unsigned char> out_bytes;
    if (!serialize_ft3_bytes(out_bytes))
    {
        return false;
    }

    std::ofstream ofs(out_fname, std::ios::binary);
    if (!ofs.good())
    {
        std::cerr << "Error: cannot open output file " << out_fname << std::endl;
        return false;
    }

    ofs.write(reinterpret_cast<const char *>(out_bytes.data()), static_cast<std::streamsize>(out_bytes.size()));

    if (!ofs.good())
    {
        std::cerr << "Error: failed writing output file " << out_fname << std::endl;
        return false;
    }

    if (!b_imaginary && b_imaginary_indirect && b_imaginary_indirect2)
    {
        std::cout << "Wrote direct-only intermediate FT3 file: " << out_fname
                  << " (direct imag removed; indirect imag kept)" << std::endl;
    }
    else
    {
        std::cout << "Wrote stacked FT3 file: " << out_fname << std::endl;
    }

    return true;
}

bool fid_3d::write_ft3_to_buffer(std::vector<unsigned char> &out_bytes)
{
    return serialize_ft3_bytes(out_bytes);
}

bool fid_3d::write_ft2_series(const std::string &first_plane_fname)
{
    if (spect == nullptr || first_plane_fname.empty())
    {
        std::cerr << "Error: write_ft2_series invalid parameters" << std::endl;
        return false;
    }

    // Parse indexed filename
    std::string path_name;
    std::string prefix;
    std::string extension;
    int start_idx = 0;
    int digit_width = 0;

    if (!fid_3d_aux::parse_indexed_filename_range(first_plane_fname, first_plane_fname, path_name, prefix, extension,
                                      start_idx, start_idx, digit_width))
    {
        std::cerr << "Error: -out-ft2-first must contain a numeric suffix, e.g. proc001.ft2" << std::endl;
        return false;
    }

    if (extension != "ft2")
    {
        std::cerr << "Error: -out-ft2-first extension must be .ft2" << std::endl;
        return false;
    }

    if (!prepare_header_for_nmrpipe())
    {
        std::cerr << "Error: failed to prepare nmrPipe header in write_ft2_series" << std::endl;
        return false;
    }

    // Write each plane
    const auto sizes = current_position_lengths();
    const size_t plane_size = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
    for (int iz = 0; iz < sizes[2]; iz++)
    {
        std::string plane_name;
        fid_3d_aux::build_indexed_filename(path_name, prefix, extension, start_idx + iz, digit_width, plane_name);

        std::ofstream ofs(plane_name, std::ios::binary);
        if (!ofs.good())
        {
            std::cerr << "Error: cannot open output plane " << plane_name << std::endl;
            return false;
        }

        ofs.write(reinterpret_cast<const char *>(nmrpipe_header_data.data()), 512 * sizeof(float));
        ofs.write(reinterpret_cast<const char *>(spect + static_cast<size_t>(iz) * plane_size),
                  plane_size * sizeof(float));

        if (!ofs.good())
        {
            std::cerr << "Error: failed writing output plane " << plane_name << std::endl;
            return false;
        }
    }

    std::cout << "Wrote FT2 plane series starting at: " << first_plane_fname << std::endl;
    return true;
}

bool fid_3d::full_process()
{
    if (b_frq && b_frq_indirect1 && b_frq_indirect2)
    {
        spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
        cleanup_unused_components();
        return true;
    }

    // Initialize apodization windows if they were not set by the caller.
    if (apodization_direct == nullptr)
    {
        apodization_direct = new apodization("none");
    }
    if (apodization_indirect1 == nullptr)
    {
        apodization_indirect1 = new apodization("none");
    }
    if (apodization_indirect2 == nullptr)
    {
        apodization_indirect2 = new apodization("none");
    }

    if (!apply_time_domain_polynorminal_baseline_axis(0))
    {
        return false;
    }

    if (!process_direct_dimension())
    {
        return false;
    }

    if (!keep_imag_direct_step)
    {
        b_imaginary = false;
        cleanup_unused_components();
    }

    if (!transpose_dimensions(0, 2))
    {
        return false;
    }
    if (!apply_time_domain_polynorminal_baseline_axis(2))
    {
        return false;
    }
    if (!process_indirect2_dimension())
    {
        return false;
    }

    if (!keep_imag_indirect2_step)
    {
        b_imaginary_indirect2 = false;
        cleanup_unused_components();
    }

    if (!transpose_dimensions(0, 1))
    {
        return false;
    }
    if (!apply_time_domain_polynorminal_baseline_axis(1))
    {
        return false;
    }
    if (!process_indirect1_dimension())
    {
        return false;
    }

    if (!keep_imag_indirect1_step)
    {
        b_imaginary_indirect = false;
        cleanup_unused_components();
    }
    // Final storage order for output: xzy (inner=x, middle=z, outer=y).
    // Only log when we actually need to reorder data before writing.
    const std::array<int, 3> desired_output_order = {0, 2, 1};
    if (current_axis_order != desired_output_order)
    {
        const char *axis_name[3] = {"F2", "F1", "F3"};
        std::cout << std::endl;
        std::cout << "Transposing processed data before write: order="
                  << axis_name[current_axis_order[0]] << ","
                  << axis_name[current_axis_order[1]] << ","
                  << axis_name[current_axis_order[2]]
                  << " -> F2,F3,F1" << std::endl;
        if (!transpose_dimensions(0, 2))
        {
            return false;
        }
    }

    if (current_axis_order != desired_output_order)
    {
        return false;
    }

    spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
    // Apply optional final frequency-domain polynomial baseline (trace-by-trace on spectrum_rrr)
    if (!apply_final_frq_polynomial_baseline())
    {
        std::cerr << "Error: final frequency-domain polynomial baseline failed." << std::endl;
        return false;
    }

    noise_level = estimate_final_rrr_segment_noise();
    std::cout << "Estimated 3D Noise Level: " << noise_level << std::endl;

    cleanup_unused_components();
    return true;
}

bool fid_3d::direct_only_process()
{
    if (b_frq && b_frq_indirect1 && b_frq_indirect2)
    {
        std::cerr << "Error: direct_only_process cannot run on fully processed 3D frequency-domain data." << std::endl;
        return false;
    }

    // Initialize apodization windows if they were not set by the caller.
    if (apodization_direct == nullptr)
    {
        apodization_direct = new apodization("none");
    }
    if (apodization_indirect1 == nullptr)
    {
        apodization_indirect1 = new apodization("none");
    }
    if (apodization_indirect2 == nullptr)
    {
        apodization_indirect2 = new apodization("none");
    }

    if (!b_frq)
    {
        if (!apply_time_domain_polynorminal_baseline_axis(0))
        {
            return false;
        }

        if (!process_direct_dimension())
        {
            return false;
        }
    }
    else if (b_frq_indirect1 || b_frq_indirect2)
    {
        std::cerr << "Error: direct_only_process does not support mixed-stage inputs where an indirect axis is already in frequency domain." << std::endl;
        return false;
    }

    // Reorder to y,z,x (indirect1, indirect2, direct), i.e. move direct axis to the end.
    const std::array<int, 3> desired_order = {1, 2, 0};
    while (current_axis_order != desired_order)
    {
        bool changed = false;
        for (int pos = 0; pos < 3; ++pos)
        {
            if (current_axis_order[pos] == desired_order[pos])
            {
                continue;
            }

            int swap_pos = pos + 1;
            while (swap_pos < 3 && current_axis_order[swap_pos] != desired_order[pos])
            {
                ++swap_pos;
            }

            if (swap_pos >= 3)
            {
                std::cerr << "Error: failed to find target axis while reordering to y,z,x." << std::endl;
                return false;
            }

            if (!transpose_dimensions(pos, swap_pos))
            {
                return false;
            }

            changed = true;
            break;
        }

        if (!changed)
        {
            break;
        }
    }

    if (current_axis_order != desired_order)
    {
        std::cerr << "Error: failed to reorder direct-only intermediate data to y,z,x." << std::endl;
        return false;
    }

    // Respect user keep/remove policies in direct-only mode as well.
    // Keep flags only remove channels; they cannot create missing channels.
    const bool keep_direct = keep_imag_direct_step;
    const bool keep_indirect1 = keep_imag_indirect1_step;
    const bool keep_indirect2 = keep_imag_indirect2_step;

    b_imaginary = b_imaginary && keep_direct;
    b_imaginary_indirect = b_imaginary_indirect && keep_indirect1;
    b_imaginary_indirect2 = b_imaginary_indirect2 && keep_indirect2;
    cleanup_unused_components();

    std::cout << "Direct-only imaginary policy applied: "
              << "direct=" << (b_imaginary ? "keep" : "remove") << ", "
              << "indirect1=" << (b_imaginary_indirect ? "keep" : "remove") << ", "
              << "indirect2=" << (b_imaginary_indirect2 ? "keep" : "remove")
              << std::endl;

    spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
    return true;
}

bool fid_3d::indirect_only_process()
{
    if (b_frq && b_frq_indirect1 && b_frq_indirect2)
    {
        spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
        cleanup_unused_components();
        return true;
    }

    // This mode expects data after direct processing only (F2 in frequency domain,
    // both indirect dimensions still in time domain).
    if (!(b_frq && !b_frq_indirect1 && !b_frq_indirect2))
    {
        std::cerr << "Error: indirect_only_process requires post-direct/pre-indirect stage input." << std::endl;
        return false;
    }

    if (apodization_direct == nullptr)
    {
        apodization_direct = new apodization("none");
    }
    if (apodization_indirect1 == nullptr)
    {
        apodization_indirect1 = new apodization("none");
    }
    if (apodization_indirect2 == nullptr)
    {
        apodization_indirect2 = new apodization("none");
    }

    // Keep flags only remove channels; they cannot recreate missing channels.
    b_imaginary = b_imaginary && keep_imag_direct_step;
    cleanup_unused_components();

    auto reorder_to = [&](const std::array<int, 3> &desired_order) -> bool
    {
        while (current_axis_order != desired_order)
        {
            bool changed = false;
            for (int pos = 0; pos < 3; ++pos)
            {
                if (current_axis_order[pos] == desired_order[pos])
                {
                    continue;
                }

                int swap_pos = pos + 1;
                while (swap_pos < 3 && current_axis_order[swap_pos] != desired_order[pos])
                {
                    ++swap_pos;
                }

                if (swap_pos >= 3)
                {
                    return false;
                }

                if (!transpose_dimensions(pos, swap_pos))
                {
                    return false;
                }

                changed = true;
                break;
            }

            if (!changed)
            {
                break;
            }
        }

        return current_axis_order == desired_order;
    };

    // Match full_process ordering for remaining dimensions.
    if (!reorder_to({2, 1, 0}))
    {
        std::cerr << "Error: failed to reorder data before indirect2 processing." << std::endl;
        return false;
    }
    if (!apply_time_domain_polynorminal_baseline_axis(2))
    {
        return false;
    }
    if (!process_indirect2_dimension())
    {
        return false;
    }
    if (!keep_imag_indirect2_step)
    {
        b_imaginary_indirect2 = false;
        cleanup_unused_components();
    }

    if (!reorder_to({1, 2, 0}))
    {
        std::cerr << "Error: failed to reorder data before indirect1 processing." << std::endl;
        return false;
    }
    if (!apply_time_domain_polynorminal_baseline_axis(1))
    {
        return false;
    }
    if (!process_indirect1_dimension())
    {
        return false;
    }
    if (!keep_imag_indirect1_step)
    {
        b_imaginary_indirect = false;
        cleanup_unused_components();
    }

    // Final storage order for output: xzy (inner=x, middle=z, outer=y),
    // matching full_process() final write order.
    const std::array<int, 3> desired_output_order = {0, 2, 1};
    if (current_axis_order != desired_output_order)
    {
        const char *axis_name[3] = {"F2", "F1", "F3"};
        std::cout << std::endl;
        std::cout << "Transposing processed data before write: order="
                  << axis_name[current_axis_order[0]] << ","
                  << axis_name[current_axis_order[1]] << ","
                  << axis_name[current_axis_order[2]]
                  << " -> F2,F3,F1" << std::endl;
        if (!reorder_to(desired_output_order))
        {
            std::cerr << "Error: failed to reorder indirect-only result to final output order." << std::endl;
            return false;
        }
    }

    if (current_axis_order != desired_output_order)
    {
        return false;
    }

    // Apply optional final frequency-domain polynomial baseline (trace-by-trace on spectrum_rrr)
    if (!apply_final_frq_polynomial_baseline())
    {
        std::cerr << "Error: final frequency-domain polynomial baseline failed." << std::endl;
        return false;
    }

    spect = spectrum_rrr.empty() ? nullptr : spectrum_rrr.data();
    noise_level = estimate_final_rrr_segment_noise();
    std::cout << "Estimated 3D Noise Level: " << noise_level << std::endl;
    cleanup_unused_components();
    return true;
}

bool fid_3d::write(const std::string &out_ft3, const std::string &out_ft2_first)
{
    auto is_requested = [](const std::string &name)
    {
        return !name.empty() && name != "none" && name != "null" && name != "no";
    };

    if (is_requested(out_ft3))
    {
        if (!write_ft3(out_ft3))
        {
            return false;
        }
    }

    if (is_requested(out_ft2_first))
    {
        if (!write_ft2_series(out_ft2_first))
        {
            return false;
        }
    }

    return true;
}

bool fid_3d::read(const std::string &input_file)
{
    return read(input_file, input_file);
}

bool fid_3d::read(const std::string &first_file, const std::string &last_file)
{
    // ========================================================================
    // Stage 1: Parse filename range and read header
    // ========================================================================
    std::string path_name;
    std::string prefix;
    std::string file_ext;
    int start_idx = 0;
    int end_idx = 0;
    int digit_width = 0;
    const bool has_indexed_range = fid_3d_aux::parse_indexed_filename_range(first_file, last_file, path_name, prefix, file_ext,
                                                                start_idx, end_idx, digit_width);
    const bool single_file_request = (first_file == last_file);
    if (!has_indexed_range && !single_file_request)
    {
        return false;
    }

    const int n_planes = has_indexed_range ? (end_idx - start_idx + 1) : 1;
    if (n_planes <= 0)
    {
        std::cerr << "Error: Invalid file range." << std::endl;
        return false;
    }

    if (has_indexed_range)
    {
        std::cout << "Reading " << n_planes << " planes from " << first_file << " to " << last_file << std::endl;
    }
    else
    {
        std::cout << "Reading single input file " << first_file << std::endl;
    }

    // Read first file header to setup dimensions and verify flags
    nmrpipe_header_data.clear();
    nmrpipe_header_data.resize(512, 0.0f);

    FILE *fp = fopen(first_file.c_str(), "rb");
    if (!fp)
    {
        std::cerr << "Error: Cannot open " << first_file << std::endl;
        return false;
    }
    size_t n_read = fread(nmrpipe_header_data.data(), sizeof(float), 512, fp);
    fclose(fp);

    if (n_read != 512)
    {
        std::cerr << "Error: header read failed (" << n_read << " floats)" << std::endl;
        return false;
    }

    // Map header to dictionaries
    nmrPipe::nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    // Set variables from nmrPipe dictionary
    if (!set_varibles_from_nmrpipe_dictionary())
    {
        return false;
    }

    auto read_single_stacked_file = [&](const std::string &stacked_fname) -> bool
    {
        const auto sizes = current_position_lengths();
        const int n0 = sizes[0];
        const int n1 = sizes[1];
        const int n2 = sizes[2];
        if (n0 <= 0 || n1 <= 0 || n2 <= 0)
        {
            std::cerr << "Error: invalid dimensions while reading stacked file." << std::endl;
            return false;
        }

        // Mirror write_ft3 mapping exactly: same complexity flags,
        // same storage-order loops, same physical-axis component decode.
        const bool axis_complex[3] = {b_imaginary, b_imaginary_indirect, b_imaginary_indirect2};
        const int c0 = axis_complex[current_axis_order[0]] ? 2 : 1;
        const int c1 = axis_complex[current_axis_order[1]] ? 2 : 1;
        const int c2 = axis_complex[current_axis_order[2]] ? 2 : 1;

        const size_t logical_total = static_cast<size_t>(n0) * static_cast<size_t>(n1) * static_cast<size_t>(n2);
        spectrum_rrr.assign(logical_total, 0.0f);
        spectrum_rri.assign(logical_total, 0.0f);
        spectrum_rir.assign(logical_total, 0.0f);
        spectrum_rii.assign(logical_total, 0.0f);
        spectrum_irr.assign(logical_total, 0.0f);
        spectrum_iri.assign(logical_total, 0.0f);
        spectrum_iir.assign(logical_total, 0.0f);
        spectrum_iii.assign(logical_total, 0.0f);

        auto get_component = [&](int imag_f2, int imag_f1, int imag_f3) -> std::vector<float> *
        {
            // Same physical-axis imag-bit encoding used by write_ft3.
            const int idx = imag_f2 + 2 * imag_f1 + 4 * imag_f3;
            switch (idx)
            {
            case 0:
                return &spectrum_rrr;
            case 1:
                return &spectrum_rri;
            case 2:
                return &spectrum_rir;
            case 3:
                return &spectrum_rii;
            case 4:
                return &spectrum_irr;
            case 5:
                return &spectrum_iri;
            case 6:
                return &spectrum_iir;
            case 7:
                return &spectrum_iii;
            default:
                return nullptr;
            }
        };

        FILE *stacked_fp = fopen(stacked_fname.c_str(), "rb");
        if (!stacked_fp)
        {
            std::cerr << "Error: Cannot open " << stacked_fname << std::endl;
            return false;
        }

        fseek(stacked_fp, 512 * sizeof(float), SEEK_SET);

        std::vector<float> trace_data(static_cast<size_t>(n0), 0.0f);
        for (int i2 = 0; i2 < n2; ++i2)
        {
            for (int i1 = 0; i1 < n1; ++i1)
            {
                const size_t base = (static_cast<size_t>(i2) * static_cast<size_t>(n1) + static_cast<size_t>(i1)) * static_cast<size_t>(n0);
                for (int q2 = 0; q2 < c2; ++q2)
                {
                    for (int q1 = 0; q1 < c1; ++q1)
                    {
                        for (int q0 = 0; q0 < c0; ++q0)
                        {
                            const size_t got = fread(trace_data.data(), sizeof(float), static_cast<size_t>(n0), stacked_fp);
                            if (got != static_cast<size_t>(n0))
                            {
                                fclose(stacked_fp);
                                std::cerr << "Error: short read in stacked file " << stacked_fname << std::endl;
                                return false;
                            }

                            // Convert storage-position imag selectors to physical-axis
                            // imag bits before selecting target component buffer.
                            int imag_by_axis[3] = {0, 0, 0};
                            imag_by_axis[current_axis_order[0]] = q0;
                            imag_by_axis[current_axis_order[1]] = q1;
                            imag_by_axis[current_axis_order[2]] = q2;

                            std::vector<float> *component = get_component(
                                imag_by_axis[0],
                                imag_by_axis[1],
                                imag_by_axis[2]);
                            if (component == nullptr)
                            {
                                fclose(stacked_fp);
                                std::cerr << "Error: invalid component mapping while reading " << stacked_fname << std::endl;
                                return false;
                            }

                            std::memcpy(component->data() + base, trace_data.data(), static_cast<size_t>(n0) * sizeof(float));
                        }
                    }
                }
            }
        }

        fclose(stacked_fp);

        spect = spectrum_rrr.data();
        cleanup_unused_components();
            
        if(noise_level_for_nus == true)
        {
            /**
             * Get maximal abs value of spectrum_rrr and set noise level to 1% of it.
             */
            noise_level = calculate_noise_level_for_nus(); //return 0.0f if spectrum_rrr is empty or all zeros
        }
    
        if(noise_level == 0.0f)
        {
            noise_level = estimate_final_rrr_segment_noise();
        }
        std::cout << "Estimated 3D Noise Level: " << noise_level << std::endl;
        print_axis_summary();
        return true;
    };

    // ========================================================================
    // Stage 2: Determine processing stage from header
    // ========================================================================
    const bool is_frq_data = b_frq && b_frq_indirect1 && b_frq_indirect2;
    const bool is_fid_data = !b_frq && !b_frq_indirect1 && !b_frq_indirect2;
    const bool is_mixed_stage = !is_frq_data && !is_fid_data;

    // ========================================================================
    // Path A: Frequency domain data (already processed)
    // ========================================================================
    if (is_frq_data)
    {
        if (n_planes > 1)
        {
            std::cout << "Warning: stacked frequency-domain read expects one file; using first file "
                      << first_file << " and ignoring remaining range." << std::endl;
        }
        return read_single_stacked_file(first_file);
    }

    // ========================================================================
    // Path B: Mixed-stage stacked data (e.g. direct-only intermediate)
    // ========================================================================
    if (is_mixed_stage)
    {
        if (n_planes > 1)
        {
            std::cout << "Warning: stacked mixed-stage read expects one file; using first file "
                      << first_file << " and ignoring remaining range." << std::endl;
        }
        return read_single_stacked_file(first_file);
    }

    // ========================================================================
    // Path C: Time domain data - full 3D FID processing pipeline
    // ========================================================================

    if (!has_indexed_range)
    {
        std::cerr << "Error: time-domain multi-plane input requires indexed first/last filenames." << std::endl;
        return false;
    }

    // Initialize apodization windows
    if (apodization_direct == nullptr)
    {
        apodization_direct = new apodization("none");
    }
    if (apodization_indirect1 == nullptr)
    {
        apodization_indirect1 = new apodization("none");
    }
    if (apodization_indirect2 == nullptr)
    {
        apodization_indirect2 = new apodization("none");
    }

    // Validate file count against header-derived Z size.
    const int expected_z = ndata_indirect2;
    const int required_planes = b_imaginary_indirect2 ? (2 * expected_z) : expected_z;
    if (required_planes <= 0)
    {
        std::cerr << "Error: invalid indirect2 size in header (" << expected_z << ")." << std::endl;
        return false;
    }
    if (n_planes < required_planes)
    {
        std::cerr << "Error: not enough input .fid files. Header expects " << required_planes
                  << " files, but range provides " << n_planes << "." << std::endl;
        return false;
    }
    int effective_end_idx = end_idx;
    if (n_planes > required_planes)
    {
        std::cout << "Warning: input range has extra files (" << n_planes << "), using first "
                  << required_planes << " according to header." << std::endl;
        effective_end_idx = start_idx + required_planes - 1;
    }

    // Step 1: Read all time-domain planes from disk
    if (!read_time_domain_planes(path_name, prefix, file_ext, start_idx, effective_end_idx, digit_width))
    {
        return false;
    }

    // Keep the loaded time-domain data in the main real buffer.
    spect = spectrum_rrr.data();

    print_axis_summary();

    return true;
}

void fid_3d::print_axis_summary() const
{
    auto print_one_axis = [](const char *axis_name,
                             const std::string &axis_nucleus,
                             double axis_sw,
                             double axis_obs,
                             double axis_origin,
                             int axis_size)
    {
        const std::string nucleus_name = axis_nucleus.empty() ? "?" : axis_nucleus;
        if (axis_size <= 0 || axis_obs == 0.0)
        {
            std::cout << axis_name << ": nucleus=" << nucleus_name
                      << ", size=" << axis_size
                      << ", ppm=[n/a, n/a]" << std::endl;
            return;
        }

        const double stop_ppm = axis_origin / axis_obs;
        const double begin_ppm = stop_ppm + axis_sw / axis_obs;

        std::cout << axis_name << ": nucleus=" << nucleus_name
                  << ", size=" << axis_size
                  << ", ppm=[" << begin_ppm << ", " << stop_ppm << "]" << std::endl;
    };

    auto get_axis_metadata = [&](int physical_axis,
                                 std::string &axis_nucleus,
                                 double &axis_sw,
                                 double &axis_obs,
                                 double &axis_origin)
    {
        if (physical_axis == 0)
        {
            axis_nucleus = nucleus;
            axis_sw = spectral_width;
            axis_obs = observed_frequency;
            axis_origin = origin_frequency;
            return;
        }
        if (physical_axis == 1)
        {
            axis_nucleus = nucleus_indirect1;
            axis_sw = spectral_width_indirect1;
            axis_obs = observed_frequency_indirect1;
            axis_origin = origin_frequency_indirect1;
            return;
        }

        axis_nucleus = nucleus_indirect2;
        axis_sw = spectral_width_indirect2;
        axis_obs = observed_frequency_indirect2;
        axis_origin = origin_frequency_indirect2;
    };

    auto get_axis_size = [&](int physical_axis) -> int
    {
        if (physical_axis == 0)
        {
            return b_frq ? ndata_frq : ndata;
        }
        if (physical_axis == 1)
        {
            return b_frq_indirect1 ? ndata_frq_indirect1 : ndata_indirect1;
        }
        if (physical_axis == 2)
        {
            return b_frq_indirect2 ? ndata_frq_indirect2 : ndata_indirect2;
        }
        return 0;
    };

    auto print_axis_from_physical = [&](const char *role_name, int physical_axis)
    {
        if (physical_axis < 0 || physical_axis > 2)
        {
            return;
        }
        std::string axis_nucleus;
        double axis_sw = 0.0;
        double axis_obs = 0.0;
        double axis_origin = 0.0;
        get_axis_metadata(physical_axis, axis_nucleus, axis_sw, axis_obs, axis_origin);

        const int axis_size = get_axis_size(physical_axis);
        print_one_axis(role_name, axis_nucleus, axis_sw, axis_obs, axis_origin, axis_size);
    };

    std::cout << "Axis summary after read:" << std::endl;
    print_axis_from_physical("F2", 0);
    print_axis_from_physical("F1", 1);
    print_axis_from_physical("F3", 2);
}

bool fid_3d::set_varibles_from_nmrpipe_dictionary()
{
    /**
     * These are not used in nmrPipe dictionary ?
     */
    if (nmrpipe_dict_float.find("GRPDLY") != nmrpipe_dict_float.end())
    {
        grpdly = nmrpipe_dict_float["GRPDLY"];
        if (grpdly <= 0.0)
        {
            grpdly = -1.0; // 0.0f means not set in nmrPipe header. We use < 0 as flag.
        }
    }

    data_type = FID_DATA_TYPE::FID_DATA_TYPE_NOT_DEFINED; // not defined
    receiver_gain = 1.0;

    b_imaginary = nmrpipe_dict_float["FDF2QUADFLAG"] == 0.0;
    b_imaginary_indirect = nmrpipe_dict_float["FDF1QUADFLAG"] == 0.0;
    b_imaginary_indirect2 = nmrpipe_dict_float["FDF3QUADFLAG"] == 0.0;

    // Restore Bruker FnMODE values saved in custom user slots.
    fnmode2 = 0;
    fnmode3 = 0;
    if (nmrpipe_dict_float.find("FDUSER1") != nmrpipe_dict_float.end())
    {
        fnmode2 = static_cast<int>(std::lround(nmrpipe_dict_float["FDUSER1"]));
    }
    if (nmrpipe_dict_float.find("FDUSER2") != nmrpipe_dict_float.end())
    {
        fnmode3 = static_cast<int>(std::lround(nmrpipe_dict_float["FDUSER2"]));
    }

    // F2 Dimension (Direct in 2D plane)
    spectral_width = nmrpipe_dict_float["FDF2SW"];
    observed_frequency = nmrpipe_dict_float["FDF2OBS"];
    carrier_frequency = nmrpipe_dict_float["FDF2CAR"] * observed_frequency;
    origin_frequency = nmrpipe_dict_float["FDF2ORIG"];

    // F1 Dimension (Indirect 1 in 2D plane)
    spectral_width_indirect1 = nmrpipe_dict_float["FDF1SW"];
    observed_frequency_indirect1 = nmrpipe_dict_float["FDF1OBS"];
    carrier_frequency_indirect1 = nmrpipe_dict_float["FDF1CAR"] * observed_frequency_indirect1;
    origin_frequency_indirect1 = nmrpipe_dict_float["FDF1ORIG"];

    // F3 (or whatever 3rd dimension is in pipe header)
    // Assuming FDF3... keywords usually exist for 3D
    spectral_width_indirect2 = nmrpipe_dict_float["FDF3SW"];
    observed_frequency_indirect2 = nmrpipe_dict_float["FDF3OBS"];
    carrier_frequency_indirect2 = nmrpipe_dict_float["FDF3CAR"] * observed_frequency_indirect2;
    origin_frequency_indirect2 = nmrpipe_dict_float["FDF3ORIG"];

    int direct_ndx = nmrpipe_dict_float["FDDIMORDER1"];    // storage inner axis label in nmrPipe convention
    int indirect_ndx = nmrpipe_dict_float["FDDIMORDER2"];  // storage middle axis label in nmrPipe convention
    int indirect_ndx2 = nmrpipe_dict_float["FDDIMORDER3"]; // storage outer axis label in nmrPipe convention

    auto pipe_dim_to_physical_axis = [](int dim_id) -> int
    {
        if (dim_id == 2)
        {
            return 0; // F2
        }
        if (dim_id == 1)
        {
            return 1; // F1
        }
        if (dim_id == 3)
        {
            return 2; // F3
        }
        return -1;
    };

    const int order0 = pipe_dim_to_physical_axis(direct_ndx);
    const int order1 = pipe_dim_to_physical_axis(indirect_ndx);
    const int order2 = pipe_dim_to_physical_axis(indirect_ndx2);
    if (order0 < 0 || order1 < 0 || order2 < 0 ||
        order0 == order1 || order0 == order2 || order1 == order2)
    {
        std::cerr << "Error: invalid FDDIMORDER1/2/3 in nmrPipe header: "
                  << direct_ndx << "/" << indirect_ndx << "/" << indirect_ndx2 << std::endl;
        return false;
    }
    current_axis_order = {order0, order1, order2};

    nucleus = nmrpipe_dict_string.count("FDF2LABEL") ? nmrpipe_dict_string["FDF2LABEL"] : "";
    nucleus_indirect1 = nmrpipe_dict_string.count("FDF1LABEL") ? nmrpipe_dict_string["FDF1LABEL"] : "";
    nucleus_indirect2 = nmrpipe_dict_string.count("FDF3LABEL") ? nmrpipe_dict_string["FDF3LABEL"] : "";

    auto pipe_dim_to_label = [](int dim_id) -> const char *
    {
        if (dim_id == 2)
        {
            return "F2";
        }
        if (dim_id == 1)
        {
            return "F1";
        }
        if (dim_id == 3)
        {
            return "F3";
        }
        return "unknown";
    };
    std::cout << "Read header dimension order (FDDIMORDER1/2/3): "
              << direct_ndx << "/" << indirect_ndx << "/" << indirect_ndx2
              << " => "
              << pipe_dim_to_label(direct_ndx) << ","
              << pipe_dim_to_label(indirect_ndx) << ","
              << pipe_dim_to_label(indirect_ndx2) << std::endl;

    const int nx = static_cast<int>(nmrpipe_dict_float["FDSIZE"]);
    int ny = static_cast<int>(nmrpipe_dict_float["FDSPECNUM"]);
    int nz = static_cast<int>(nmrpipe_dict_float["FDF3SIZE"]);
    if (nz <= 0)
    {
        nz = static_cast<int>(nmrpipe_dict_float["FDF3TDSIZE"]);
    }

    // Undo nmrPipe storage-size doubling using current storage order:
    // writer doubles FDSPECNUM when storage dim0 and dim1 are both complex.
    // Some external files may also double FDF3SIZE when all three storage dims are complex.
    const bool axis_complex[3] = {b_imaginary, b_imaginary_indirect, b_imaginary_indirect2};
    const bool storage_complex0 = axis_complex[current_axis_order[0]];
    const bool storage_complex1 = axis_complex[current_axis_order[1]];
    const bool storage_complex2 = axis_complex[current_axis_order[2]];

    if (storage_complex0 && storage_complex1 && ny > 1)
    {
        ny /= 2;
    }
    if (storage_complex0 && storage_complex1 && storage_complex2 && nz > 1)
    {
        nz /= 2;
    }

    /**
     * Check for FT flag.
     */
    b_frq = false;
    if (nmrpipe_dict_float["FDF2FTFLAG"] == 1.0)
    {
        b_frq = true;
    }
    b_frq_indirect1 = (nmrpipe_dict_float["FDF1FTFLAG"] == 1.0);
    b_frq_indirect2 = (nmrpipe_dict_float["FDF3FTFLAG"] == 1.0);

    // Convert storage-order sizes (nx, ny, nz) to physical-axis sizes (F2, F1, F3).
    int size_by_storage_pos[3] = {nx, ny, nz};
    int size_by_physical_axis[3] = {0, 0, 0};
    for (int pos = 0; pos < 3; ++pos)
    {
        const int physical_axis = current_axis_order[pos];
        if (physical_axis < 0 || physical_axis > 2)
        {
            std::cerr << "Error: invalid axis order while mapping header sizes." << std::endl;
            return false;
        }
        size_by_physical_axis[physical_axis] = std::max(1, size_by_storage_pos[pos]);
    }

    const int size_f2 = size_by_physical_axis[0];
    const int size_f1 = size_by_physical_axis[1];
    const int size_f3 = size_by_physical_axis[2];

    if (b_frq)
    {
        ndata_frq = size_f2;
        ndata = static_cast<int>(nmrpipe_dict_float["FDF2TDSIZE"]);
    }
    else
    {
        ndata = size_f2;
        ndata_frq = 0;
    }

    if (b_frq_indirect1)
    {
        ndata_frq_indirect1 = size_f1;
        ndata_indirect1 = static_cast<int>(nmrpipe_dict_float["FDF1TDSIZE"]);
    }
    else
    {
        ndata_indirect1 = size_f1;
        ndata_frq_indirect1 = 0;
    }

    if (b_frq_indirect2)
    {
        ndata_frq_indirect2 = size_f3;
        ndata_indirect2 = static_cast<int>(nmrpipe_dict_float["FDF3TDSIZE"]);
    }
    else
    {
        ndata_indirect2 = size_f3;
        ndata_frq_indirect2 = 0;
    }

    ndata_power_of_2 = fid_3d_aux::next_power_of_2_int(std::max(1, ndata));
    ndata_power_of_2_indirect1 = fid_3d_aux::next_power_of_2_int(std::max(1, ndata_indirect1));
    ndata_power_of_2_indirect2 = fid_3d_aux::next_power_of_2_int(std::max(1, ndata_indirect2));

    const int n1 = b_frq ? ndata_frq : ndata;
    const int n2 = b_frq_indirect1 ? ndata_frq_indirect1 : ndata_indirect1;
    const int n3 = b_frq_indirect2 ? ndata_frq_indirect2 : ndata_indirect2;

    n_center = nmrpipe_dict_float.count("FDF2CENTER") ? static_cast<int>(nmrpipe_dict_float["FDF2CENTER"]) : (n1 / 2 + 1);
    n_center_indirect1 = nmrpipe_dict_float.count("FDF1CENTER") ? static_cast<int>(nmrpipe_dict_float["FDF1CENTER"]) : (n2 / 2 + 1);
    n_center_indirect2 = nmrpipe_dict_float.count("FDF3CENTER") ? static_cast<int>(nmrpipe_dict_float["FDF3CENTER"]) : (n3 / 2 + 1);

    stop1 = origin_frequency / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency;
    step1 = (stop1 - begin1) / n1;
    begin1 += step1; // Adjust to center of first point using nmrPipe convention

    stop2 = origin_frequency_indirect1 / observed_frequency_indirect1;
    begin2 = stop2 + spectral_width_indirect1 / observed_frequency_indirect1;
    step2 = (stop2 - begin2) / n2;
    begin2 += step2; // Adjust to center of first point using nmrPipe convention

    stop3 = origin_frequency_indirect2 / observed_frequency_indirect2;
    begin3 = stop3 + spectral_width_indirect2 / observed_frequency_indirect2;
    step3 = (stop3 - begin3) / n3;
    begin3 += step3; // Adjust to center of first point using nmrPipe convention

    return true;
}

bool fid_3d::set_up_apodization(apodization *a1, apodization *a2, apodization *a3)
{
    apodization_direct = a1;
    apodization_indirect1 = a2;
    apodization_indirect2 = a3;
    return true;
}

bool fid_3d::set_up_apodization_from_string(const std::string &apodization_string_direct,
                                            const std::string &apodization_string_indirect1,
                                            const std::string &apodization_string_indirect2)
{
    apodization_direct = new apodization(apodization_string_direct);
    apodization_indirect1 = new apodization(apodization_string_indirect1);
    apodization_indirect2 = new apodization(apodization_string_indirect2);
    return true;
}

bool fid_3d::run_zf(int n_zf_direct, int n_zf_indirect1, int n_zf_indirect2)
{
    if (n_zf_direct < 1 || n_zf_indirect1 < 1 || n_zf_indirect2 < 1)
    {
        std::cerr << "Error: zero filling factors must be >= 1." << std::endl;
        return false;
    }

    zf = n_zf_direct;
    zf_indirect1 = n_zf_indirect1;
    zf_indirect2 = n_zf_indirect2;
    return true;
}

#ifdef WEBASSEMBLY
uintptr_t fid_3d::get_nmrpipe_header_data()
{
    return reinterpret_cast<uintptr_t>(nmrpipe_header_data.data());
}

uintptr_t fid_3d::get_data_of_rrr()
{
    return reinterpret_cast<uintptr_t>(spectrum_rrr.data());
}

bool fid_3d::read_bruker_fid_data_bytes_raw(uintptr_t ptr, size_t size)
{
    if (ptr == 0 || size == 0)
    {
        std::cerr << "Error: empty Bruker FID raw payload" << std::endl;
        return false;
    }

    if (expected_bruker_raw_points == 0)
    {
        std::cerr << "Error: invalid Bruker dimensions. Call read_bruker_files_as_strings first." << std::endl;
        return false;
    }

    const uint8_t* fid_data_bytes_ptr = reinterpret_cast<const uint8_t*>(ptr);

    std::vector<float> fid_data_float_in;
    fid_data_float_in.resize(expected_bruker_raw_points, 0.0f);

    if (data_type == FID_DATA_TYPE_INT32)
    {
        const size_t expected_bytes = expected_bruker_raw_points * sizeof(int32_t);
        if (size != expected_bytes)
        {
            std::cerr << "Error: Bruker INT32 payload size mismatch (expected "
                      << expected_bytes << " bytes, got " << size << ")." << std::endl;
            return false;
        }

        for (size_t i = 0; i < expected_bruker_raw_points; ++i)
        {
            int32_t v = 0;
            std::memcpy(&v, fid_data_bytes_ptr + i * sizeof(int32_t), sizeof(int32_t));
            fid_data_float_in[i] = static_cast<float>(v);
        }
    }
    else if (data_type == FID_DATA_TYPE_FLOAT64)
    {
        const size_t expected_bytes = expected_bruker_raw_points * sizeof(double);
        if (size != expected_bytes)
        {
            std::cerr << "Error: Bruker FLOAT64 payload size mismatch (expected "
                      << expected_bytes << " bytes, got " << size << ")." << std::endl;
            return false;
        }

        for (size_t i = 0; i < expected_bruker_raw_points; ++i)
        {
            double v = 0.0;
            std::memcpy(&v, fid_data_bytes_ptr + i * sizeof(double), sizeof(double));
            fid_data_float_in[i] = static_cast<float>(v);
        }
    }
    else
    {
        std::cerr << "Error: unsupported Bruker data type" << std::endl;
        return false;
    }

    return read_bruker_fid_data(fid_data_float_in);
}

bool fid_3d::read_ft3_from_buffer_raw(uintptr_t ptr, size_t size)
{
    const size_t header_bytes = 512 * sizeof(float);
    if (size < header_bytes)
    {
        std::cerr << "Error: FT3 payload is too small to contain a valid nmrPipe header." << std::endl;
        return false;
    }

    const uint8_t* ft3_bytes_ptr = reinterpret_cast<const uint8_t*>(ptr);

    nmrpipe_header_data.assign(512, 0.0f);
    std::memcpy(nmrpipe_header_data.data(), ft3_bytes_ptr, header_bytes);

    nmrPipe::nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
    if (!set_varibles_from_nmrpipe_dictionary())
    {
        return false;
    }

    const bool is_frq_data = b_frq && b_frq_indirect1 && b_frq_indirect2;
    const bool is_fid_data = !b_frq && !b_frq_indirect1 && !b_frq_indirect2;
    const bool is_mixed_stage = !is_frq_data && !is_fid_data;
    if (!is_frq_data && !is_mixed_stage)
    {
        std::cerr << "Error: read_ft3_from_buffer_raw expects stacked FT3 data (frequency-domain or mixed-stage)." << std::endl;
        return false;
    }

    const auto sizes = current_position_lengths();
    const int n0 = sizes[0];
    const int n1 = sizes[1];
    const int n2 = sizes[2];
    if (n0 <= 0 || n1 <= 0 || n2 <= 0)
    {
        std::cerr << "Error: invalid dimensions while reading FT3 buffer." << std::endl;
        return false;
    }

    const bool axis_complex[3] = {b_imaginary, b_imaginary_indirect, b_imaginary_indirect2};
    const int c0 = axis_complex[current_axis_order[0]] ? 2 : 1;
    const int c1 = axis_complex[current_axis_order[1]] ? 2 : 1;
    const int c2 = axis_complex[current_axis_order[2]] ? 2 : 1;

    const size_t logical_total = static_cast<size_t>(n0) * static_cast<size_t>(n1) * static_cast<size_t>(n2);
    spectrum_rrr.assign(logical_total, 0.0f);
    spectrum_rri.assign(logical_total, 0.0f);
    spectrum_rir.assign(logical_total, 0.0f);
    spectrum_rii.assign(logical_total, 0.0f);
    spectrum_irr.assign(logical_total, 0.0f);
    spectrum_iri.assign(logical_total, 0.0f);
    spectrum_iir.assign(logical_total, 0.0f);
    spectrum_iii.assign(logical_total, 0.0f);

    auto get_component = [&](int imag_f2, int imag_f1, int imag_f3) -> std::vector<float> *
    {
        const int idx = imag_f2 + 2 * imag_f1 + 4 * imag_f3;
        switch (idx)
        {
        case 0:
            return &spectrum_rrr;
        case 1:
            return &spectrum_rri;
        case 2:
            return &spectrum_rir;
        case 3:
            return &spectrum_rii;
        case 4:
            return &spectrum_irr;
        case 5:
            return &spectrum_iri;
        case 6:
            return &spectrum_iir;
        case 7:
            return &spectrum_iii;
        default:
            return nullptr;
        }
    };

    const size_t trace_bytes = static_cast<size_t>(n0) * sizeof(float);
    const size_t expected_payload_bytes =
        static_cast<size_t>(n0) * static_cast<size_t>(n1) * static_cast<size_t>(n2) *
        static_cast<size_t>(c0) * static_cast<size_t>(c1) * static_cast<size_t>(c2) * sizeof(float);
    if (size != header_bytes + expected_payload_bytes)
    {
        std::cerr << "Error: FT3 payload size mismatch (expected "
                  << (header_bytes + expected_payload_bytes) << " bytes, got "
                  << size << ")." << std::endl;
        return false;
    }

    size_t payload_offset = header_bytes;
    std::vector<float> trace_data(static_cast<size_t>(n0), 0.0f);
    for (int i2 = 0; i2 < n2; ++i2)
    {
        for (int i1 = 0; i1 < n1; ++i1)
        {
            const size_t base = (static_cast<size_t>(i2) * static_cast<size_t>(n1) + static_cast<size_t>(i1)) * static_cast<size_t>(n0);
            for (int q2 = 0; q2 < c2; ++q2)
            {
                for (int q1 = 0; q1 < c1; ++q1)
                {
                    for (int q0 = 0; q0 < c0; ++q0)
                    {
                        std::memcpy(trace_data.data(), ft3_bytes_ptr + payload_offset, trace_bytes);
                        payload_offset += trace_bytes;

                        int imag_by_axis[3] = {0, 0, 0};
                        imag_by_axis[current_axis_order[0]] = q0;
                        imag_by_axis[current_axis_order[1]] = q1;
                        imag_by_axis[current_axis_order[2]] = q2;

                        std::vector<float> *component = get_component(
                            imag_by_axis[0],
                            imag_by_axis[1],
                            imag_by_axis[2]);
                        if (component == nullptr)
                        {
                            std::cerr << "Error: invalid component mapping while reading FT3 buffer." << std::endl;
                            return false;
                        }

                        std::memcpy(component->data() + base, trace_data.data(), trace_bytes);
                    }
                }
            }
        }
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;
    spect = spectrum_rrr.data();
    cleanup_unused_components();
   
    if(noise_level_for_nus == true)
    {
        /**
         * Get maximal abs value of spectrum_rrr and set noise level to 1% of it.
         */
        noise_level = calculate_noise_level_for_nus(); //return 0.0f if spectrum_rrr is empty or all zeros
    }

    if(noise_level == 0.0f)
    {
        noise_level = estimate_final_rrr_segment_noise();
    }
    std::cout << "Estimated 3D Noise Level: " << noise_level << std::endl;
    print_axis_summary();
    return true;
}

bool fid_3d::serialize_ft3_to_internal_buffer()
{
    ft3_serialized_buffer.clear();
    return serialize_ft3_bytes(ft3_serialized_buffer);
}

uintptr_t fid_3d::get_ft3_buffer_ptr() const
{
    return reinterpret_cast<uintptr_t>(ft3_serialized_buffer.data());
}

size_t fid_3d::get_ft3_buffer_size() const
{
    return ft3_serialized_buffer.size();
}

EMSCRIPTEN_BINDINGS(dp_module_fid_3d)
{
    class_<fid_3d, base<fid_base>>("fid_3d")
        .constructor()
        .function("set_up_apodization_from_string", &fid_3d::set_up_apodization_from_string)
        .function("run_zf", &fid_3d::run_zf)
        .function("extract_region", &fid_3d::extract_region)
        .function("extract_region_ppm", &fid_3d::extract_region_ppm)
        .function("read_phase_correction_from_string", &fid_3d::read_phase_correction_from_string)
        .function("set_time_domain_polynorminal_order", &fid_3d::set_time_domain_polynorminal_order)
        .function("set_frq_domain_polynorminal_order", &fid_3d::set_frq_domain_polynorminal_order)
        .function("set_final_frq_polynorminal_order", &fid_3d::set_final_frq_polynorminal_order)
        .function("set_inverse", &fid_3d::set_inverse)
        .function("set_delete_image", &fid_3d::set_delete_image)
        .function("read_bruker_files_as_strings", &fid_3d::read_bruker_files_as_strings)
        .function("read_bruker_fid_data", &fid_3d::read_bruker_fid_data)
        .function("read_bruker_fid_data_bytes", &fid_3d::read_bruker_fid_data_bytes)
        .function("read_bruker_fid_data_bytes_raw", &fid_3d::read_bruker_fid_data_bytes_raw)
        .function("read_ft3_from_buffer", &fid_3d::read_ft3_from_buffer)
        .function("read_ft3_from_buffer_raw", &fid_3d::read_ft3_from_buffer_raw)
        .function("full_process", &fid_3d::full_process)
        .function("direct_only_process", &fid_3d::direct_only_process)
        .function("indirect_only_process", &fid_3d::indirect_only_process)
        .function("write_ft3_to_buffer", &fid_3d::write_ft3_to_buffer)
        .function("serialize_ft3_to_internal_buffer", &fid_3d::serialize_ft3_to_internal_buffer)
        .function("get_ft3_buffer_ptr", &fid_3d::get_ft3_buffer_ptr)
        .function("get_ft3_buffer_size", &fid_3d::get_ft3_buffer_size)
        .function("prepare_header_for_nmrpipe", &fid_3d::prepare_header_for_nmrpipe)
        .function("get_nmrpipe_header_data", &fid_3d::get_nmrpipe_header_data)
        .function("get_data_of_rrr", &fid_3d::get_data_of_rrr)
        .function("get_ndata_direct", &fid_3d::get_ndata_direct)
        .function("get_ndata_indirect1", &fid_3d::get_ndata_indirect1)
        .function("get_ndata_indirect2", &fid_3d::get_ndata_indirect2)
        .function("postprocess_loaded_ft3", &fid_3d::postprocess_loaded_ft3);
}
#endif // WEBASSEMBLY
