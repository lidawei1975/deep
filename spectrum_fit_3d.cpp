#include "spectrum_fit_3d.h"

#ifdef WEBASSEMBLY
#include <emscripten/bind.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>
#include "lmminimizer.h"

// ============================================================================
// Constants for 3D peak fitting
// ============================================================================

static constexpr float FwhhFacctorSeperation = 5.0f; // For spectrum separation (peak model -> peak profile).
static constexpr float FwhhFacctorFitting = 2.0f;   // For individual peak fitting (separated profile -> peak parameters).
static constexpr bool kDebugDumpEmRoundPeaks = false; // Whether to dump peak info at each EM round for debugging. This can be helpful to understand how peaks evolve during EM iterations, but may produce a large number of files.    
static constexpr int kDebugDumpEmRoundInterval = 100; // Interval for dumping EM round peaks. Only dump when EM round number is divisible by this value. Set to 1 to dump every round, or set to a large value to effectively disable dumping.
static constexpr bool kUseWatershedFittingRegion = true;  // Use watershed algorithm to define fitting region

// ============================================================================
// Helper function for dumping peak debug info
// ============================================================================

static void dump_prefit_em_round_peaks(
    const std::vector<Peak3D>& peaks,
    int em_round,
    const fid_3d::partition& coor,
    double begin1, double step1,
    double begin2, double step2,
    double begin3, double step3,
    int x0, int y0, int z0)
{
    std::ostringstream fname;
    fname << "3d_prefit_em_round_" << std::setw(4) << std::setfill('0') << em_round
          << "_x" << coor.min_x << "_" << coor.max_x
          << "_y" << coor.min_y << "_" << coor.max_y
          << "_z" << coor.min_z << "_" << coor.max_z << ".txt";

    std::ofstream out(fname.str());
    if (!out.is_open()) {
        std::cerr << "Warning: cannot write EM debug peak file " << fname.str() << std::endl;
        return;
    }

    out << "# EM Round Peak Snapshot" << std::endl;
    out << "# EM Round: " << em_round << std::endl;
    out << "# Number of Peaks Fitted: " << peaks.size() << std::endl;
    out << "# Partition Bounds (global): X[" << coor.min_x << ", " << coor.max_x
        << "], Y[" << coor.min_y << ", " << coor.max_y
        << "], Z[" << coor.min_z << ", " << coor.max_z << "]" << std::endl;
    out << std::endl;
    out << "Peak\tAmplitude\tX_Center\tY_Center\tZ_Center\tX_Center_ppm\tY_Center_ppm\tZ_Center_ppm\tX_FWHH\t\tY_FWHH\t\tZ_FWHH\t\tLx\tLy\tLz" << std::endl;
    out << "====\t=========\t========\t========\t========\t======\t\t======\t\t======\t\t=======\t\t=======\t\t=======\t\t==\t==\t==" << std::endl;

    for (size_t i = 0; i < peaks.size(); ++i) {
        const auto& peak = peaks[i];

        double ppm_f3 = begin1 + peak.x_center * step1;
        double ppm_f2 = begin2 + peak.y_center * step2;
        double ppm_f1 = begin3 + peak.z_center * step3;

        out << (i + 1) << "\t"
            << std::scientific << std::setprecision(6) << peak.amplitude << "\t"
            << std::fixed << peak.x_center << "\t" << peak.y_center << "\t" << peak.z_center << "\t"
            << std::fixed << std::setprecision(6) << ppm_f3 << "\t"
            << ppm_f2 << "\t"
            << ppm_f1 << "\t"
            << std::scientific << std::setprecision(6) << peak.x_width << "\t\t"
            << peak.y_width << "\t\t"
            << peak.z_width << "\t\t"
            << std::fixed << std::setprecision(2) << peak.x_lfrac << "\t"
            << peak.y_lfrac << "\t"
            << peak.z_lfrac << std::endl;
    }

    out.close();

}

/**
 * Watershed-like scan to define fitting region for a peak, starting from the peak center
 * and expanding outwards until reaching noise threshold or local minima.
 * This is designed to capture the full peak profile for better fitting, while excluding distant voxels
 */
void spectrum_fit_3d::watershed_fitting_region(
    const std::vector<float>& peak_data,      // Separated peak data for this peak
    const std::vector<int>& x_coords,         // X coordinates of voxels
    const std::vector<int>& y_coords,         // Y coordinates of voxels
    const std::vector<int>& z_coords,         // Z coordinates of voxels
    float peak_x_center,                      // Peak center in local coords
    float peak_y_center,
    float peak_z_center,
    std::set<int>& watershed_region)          // Output: set of included voxel indices
{
    watershed_region.clear();
    
    if (peak_data.empty()) {
        return;
    }
    
    // Find the voxel closest to peak center
    int start_idx = -1;
    float min_dist_sq = std::numeric_limits<float>::max();
    for (size_t idx = 0; idx < peak_data.size(); idx++) {
        if (peak_data[idx] <= noise_level) continue;
        
        float dx = (float)x_coords[idx] - peak_x_center;
        float dy = (float)y_coords[idx] - peak_y_center;
        float dz = (float)z_coords[idx] - peak_z_center;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            start_idx = (int)idx;
        }
    }
    
    if (start_idx < 0) {
        return;  // No valid starting point
    }
    
    float peak_max = peak_data[start_idx];
    float threshold = std::min(scale2 * noise_level, 0.01f * peak_max);
    
    // Build spatial index for fast neighbor lookup
    std::map<std::tuple<int, int, int>, int> coord_to_idx;
    for (size_t idx = 0; idx < peak_data.size(); idx++) {
        auto key = std::make_tuple(x_coords[idx], y_coords[idx], z_coords[idx]);
        coord_to_idx[key] = (int)idx;
    }
    
    int x0 = (int)std::round(peak_x_center);
    int y0 = (int)std::round(peak_y_center);
    int z0 = (int)std::round(peak_z_center);

    // Anchor scan at rounded center index when available.
    // Sparse partitions may not contain that exact voxel, so fallback to nearest valid voxel.
    if (coord_to_idx.find(std::make_tuple(x0, y0, z0)) == coord_to_idx.end()) {
        x0 = x_coords[start_idx];
        y0 = y_coords[start_idx];
        z0 = z_coords[start_idx];
    }

    auto value_at = [&](int x, int y, int z, float& value) -> bool {
        auto it = coord_to_idx.find(std::make_tuple(x, y, z));
        if (it == coord_to_idx.end()) {
            return false;
        }
        value = peak_data[it->second];
        return true;
    };

    auto scan_axis = [&](int axis) -> std::pair<int, int> {
        int lo = 0;
        int hi = 0;
        const int kConsecutiveStopCount = 3;

        // Negative direction
        float prev = peak_max;
        int bad_count = 0;
        int step = -1;
        int pos = step;
        while (true) {
            int x = x0;
            int y = y0;
            int z = z0;
            if (axis == 0) x = x0 + pos;
            if (axis == 1) y = y0 + pos;
            if (axis == 2) z = z0 + pos;

            float v = 0.0f;
            if (!value_at(x, y, z, v)) break;
            if (v < threshold || v > prev) {
                bad_count++;
                if (bad_count >= kConsecutiveStopCount) {
                    break;
                }
                pos += step;
                continue;
            }
            bad_count = 0;
            lo = pos;
            prev = v;
            pos += step;
        }

        // Positive direction
        prev = peak_max;
        bad_count = 0;
        step = 1;
        pos = step;
        while (true) {
            int x = x0;
            int y = y0;
            int z = z0;
            if (axis == 0) x = x0 + pos;
            if (axis == 1) y = y0 + pos;
            if (axis == 2) z = z0 + pos;

            float v = 0.0f;
            if (!value_at(x, y, z, v)) break;
            if (v < threshold || v > prev) {
                bad_count++;
                if (bad_count >= kConsecutiveStopCount) {
                    break;
                }
                pos += step;
                continue;
            }
            bad_count = 0;
            hi = pos;
            prev = v;
            pos += step;
        }

        return std::make_pair(lo, hi);
    };

    std::pair<int, int> x_span = scan_axis(0);
    std::pair<int, int> y_span = scan_axis(1);
    std::pair<int, int> z_span = scan_axis(2);

    int x_lo = x0 + x_span.first;
    int x_hi = x0 + x_span.second;
    int y_lo = y0 + y_span.first;
    int y_hi = y0 + y_span.second;
    int z_lo = z0 + z_span.first;
    int z_hi = z0 + z_span.second;

    // Rectangular prism region from independent 1D axis bounds.
    // Do not inspect other data values when filling this box.
    for (size_t idx = 0; idx < peak_data.size(); ++idx) {
        int x = x_coords[idx];
        int y = y_coords[idx];
        int z = z_coords[idx];
        if (x >= x_lo && x <= x_hi &&
            y >= y_lo && y <= y_hi &&
            z >= z_lo && z <= z_hi) {
            watershed_region.insert((int)idx);
        }
    }
}

// Standard Disjoint Set Union for merging labels
struct DSU {
    std::vector<int> parent;
    DSU(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    void unite(int i, int j) {
        int ri = find(i);
        int rj = find(j);
        if (ri != rj) parent[rj] = ri;
    }
};

struct Segment {
    int y, z;
    int x_start, x_end;
    int label; // This will become the final 3D region ID
};

class RegionLabeler3D {
public:
    int nx, ny, nz;

    RegionLabeler3D(int x, int y, int z) : nx(x), ny(y), nz(z) {}

    void process(const float* spect, float threshold, std::vector<int>& final_labels) {
        size_t total_points = (size_t)nx * ny * nz;
        std::vector<Segment> segments;

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    if (spect[(z * ny + y) * nx + x] > threshold) {
                        int start = x;
                        while (x < nx && spect[(z * ny + y) * nx + x] > threshold) {
                            x++;
                        }
                        segments.push_back({y, z, start, x - 1, (int)segments.size()});
                    }
                }
            }
        }

        if (segments.empty()) {
            final_labels.assign(total_points, 0);
            return;
        }

        DSU dsu(segments.size());

        for (size_t i = 0; i < segments.size(); ++i) {
            for (size_t j = i + 1; j < segments.size(); ++j) {
                const auto& s1 = segments[i];
                const auto& s2 = segments[j];

                if (s2.z > s1.z + 1) break;

                bool is_y_neighbor = (s1.z == s2.z && s2.y == s1.y + 1);
                bool is_z_neighbor = (s1.y == s2.y && s2.z == s1.z + 1);

                if (is_y_neighbor || is_z_neighbor) {
                    if (std::max(s1.x_start, s2.x_start) <= std::min(s1.x_end, s2.x_end)) {
                        dsu.unite(i, j);
                    }
                }
            }
        }

        final_labels.assign(total_points, 0);
        for (size_t i = 0; i < segments.size(); ++i) {
            int root_label = dsu.find(i) + 1;
            const auto& s = segments[i];
            for (int x = s.x_start; x <= s.x_end; ++x) {
                final_labels[(s.z * ny + s.y) * nx + x] = root_label;
            }
        }
        std::cout << "Detected " << segments.size() << " segments." << std::endl;
    }

    // Collect partition voxel coordinates and statistics from an existing label array.
    // - `labels` should be a per-voxel label array produced by `process()` (0 = background,
    //    >0 = partition id). The output `out_partitions` is filled with one
    //    `fid_3d::partition` per detected partition (ordered by label value appearance).
    void collect_partitions_from_labels(const float* spect, const std::vector<int>& labels, std::vector<fid_3d::partition>& out_partitions) {
        out_partitions.clear();
        std::map<int, size_t> label_to_index;
        double max_total_intensity = 0.0;

        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                    size_t idx = (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) * static_cast<size_t>(nx) + static_cast<size_t>(x);
                    int label = labels[idx];
                    if (label > 0) {
                        auto it = label_to_index.find(label);
                        if (it == label_to_index.end()) {
                            out_partitions.push_back(fid_3d::partition());
                            size_t new_index = out_partitions.size() - 1;
                            label_to_index[label] = new_index;
                            it = label_to_index.find(label);

                            // Initialize bounds
                            out_partitions[new_index].min_x = x;
                            out_partitions[new_index].max_x = x;
                            out_partitions[new_index].min_y = y;
                            out_partitions[new_index].max_y = y;
                            out_partitions[new_index].min_z = z;
                            out_partitions[new_index].max_z = z;
                        }

                        auto& p = out_partitions[it->second];
                        p.x.push_back(x);
                        p.y.push_back(y);
                        p.z.push_back(z);
                        float v = spect[idx];
                        p.total_intensity += v;
                        p.max_intensity = std::max(p.max_intensity, v);

                        p.min_x = std::min(p.min_x, x);
                        p.max_x = std::max(p.max_x, x);
                        p.min_y = std::min(p.min_y, y);
                        p.max_y = std::max(p.max_y, y);
                        p.min_z = std::min(p.min_z, z);
                        p.max_z = std::max(p.max_z, z);

                        if (p.total_intensity > max_total_intensity) {
                            max_total_intensity = p.total_intensity;
                        }
                    }
                }
            }
        }
    }
};


spectrum_fit_3d::spectrum_fit_3d()
    : verbose_level(2),
      partition_threshold(0.0f),
      scale1(6.0f),
      scale2(3.5f),
      nor_res_cutoff(0.4f)
{
}

spectrum_fit_3d::~spectrum_fit_3d()
{
}

bool spectrum_fit_3d::load_picked_peaks_file(const std::string& fname)
{
    picked_peaks_all.clear();

    std::ifstream peaks_file(fname);
    if (!peaks_file.is_open()) return false;
    std::string line;
    while (std::getline(peaks_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("X_Center") != std::string::npos) continue;
        if (line.find("====") != std::string::npos) continue;
        std::istringstream iss(line);
        int i, j, k;
        double ppm1, ppm2, ppm3, height;
        if (iss >> i >> j >> k >> ppm1 >> ppm2 >> ppm3 >> height) {
            picked_peaks_all.emplace_back(i, j, k, (float)height);
        }
    }
    peaks_file.close();
    return true;
}

bool spectrum_fit_3d::peak_picking(std::string outfname)
{
    if (!spect) {
        std::cerr << "Error: Spectrum data not loaded." << std::endl;
        return false;
    }

    if (ndata_frq <= 0 || ndata_frq_indirect1 <= 0 || ndata_frq_indirect2 <= 0) {
        std::cerr << "Error: Invalid dimensions for peak picking." << std::endl;
        return false;
    }

    if (current_axis_order[0] < 0 || current_axis_order[0] > 2 ||
        current_axis_order[1] < 0 || current_axis_order[1] > 2 ||
        current_axis_order[2] < 0 || current_axis_order[2] > 2) {
        std::cerr << "Error: invalid storage axis order for peak picking." << std::endl;
        return false;
    }

    float threshold = scale1 * noise_level;
    if (verbose_level >= 1) {
        std::cout << "Peak picking with threshold: " << threshold
                << " (" << scale1 << " * noise level)" << std::endl;
    }
    std::vector<std::tuple<int, int, int, float>> peaks; // index i, j, k, and peak height

    // Physical-axis sizes (F2/F1/F3).
    const int ndata_by_physical_axis[3] = {ndata_frq, ndata_frq_indirect1, ndata_frq_indirect2};

    // Storage-order sizes. Inner is always the direct axis (F2), while the two indirect axes
    // swap depending on whether the header order is 2/1/3 or 2/3/1.
    const int ndata_inner = ndata_by_physical_axis[current_axis_order[0]];
    const int ndata_intermediate = ndata_by_physical_axis[current_axis_order[1]];
    const int ndata_outer = ndata_by_physical_axis[current_axis_order[2]];

    const int phys_axis_at_storage_pos[3] = {
        current_axis_order[0],
        current_axis_order[1],
        current_axis_order[2]
    };

    int storage_pos_by_phys_axis[3] = {0, 0, 0};
    for (int pos = 0; pos < 3; ++pos) {
        storage_pos_by_phys_axis[phys_axis_at_storage_pos[pos]] = pos;
    }

    const size_t stride_inner = 1;
    const size_t stride_intermediate = static_cast<size_t>(ndata_inner);
    const size_t stride_outer = static_cast<size_t>(ndata_inner) * static_cast<size_t>(ndata_intermediate);

    // Iterate through the 3D spectrum
    // Memory layout follows current storage order: [outer][intermediate][inner]
    // Start at 2 and end at -2 to accommodate 2-step neighbor checks
    for (int outer = 2; outer < ndata_outer - 2; outer++) {
        for (int intermediate = 2; intermediate < ndata_intermediate - 2; intermediate++) {
            for (int inner = 2; inner < ndata_inner - 2; inner++) {
                size_t idx = (static_cast<size_t>(outer) * static_cast<size_t>(ndata_intermediate) + static_cast<size_t>(intermediate)) * static_cast<size_t>(ndata_inner) + static_cast<size_t>(inner);
                float val = spect[idx];

                // Check if above noise threshold
                if (val <= threshold ) { // 10.0 is a safety factor to avoid picking noise peaks
                    continue;
                }

                // Check against 12 direct neighbors (1 and 2 steps in each direction)
                // These are at Manhattan distance 1 or 2 along axes
                float neighbors_direct[12] = {
                    spect[idx + stride_inner],              // inner+1
                    spect[idx - stride_inner],              // inner-1
                    spect[idx + 2 * stride_inner],          // inner+2
                    spect[idx - 2 * stride_inner],          // inner-2
                    spect[idx + stride_intermediate],       // intermediate+1
                    spect[idx - stride_intermediate],       // intermediate-1
                    spect[idx + 2 * stride_intermediate],   // intermediate+2
                    spect[idx - 2 * stride_intermediate],   // intermediate-2
                    spect[idx + stride_outer],              // outer+1
                    spect[idx - stride_outer],              // outer-1
                    spect[idx + 2 * stride_outer],          // outer+2
                    spect[idx - 2 * stride_outer]           // outer-2
                };

                // Must be larger than ALL direct neighbors
                bool is_larger_than_direct = true;
                for (int n = 0; n < 12; n++) {
                    if (val <= neighbors_direct[n]) {
                        is_larger_than_direct = false;
                        break;
                    }
                }

                if (!is_larger_than_direct) {
                    continue;
                }

                // Check sqrt(2) distance neighbors (edge diagonal neighbors)
                // 12 total: 3 planes with 4 edge neighbors each
                float neighbors_sqrt2[12] = {
                    spect[idx + stride_inner + stride_intermediate],                       // inner+1, intermediate+1
                    spect[idx + stride_inner - stride_intermediate],                       // inner+1, intermediate-1
                    spect[idx - stride_inner + stride_intermediate],                       // inner-1, intermediate+1
                    spect[idx - stride_inner - stride_intermediate],                       // inner-1, intermediate-1
                    spect[idx + stride_inner + stride_outer],                              // inner+1, outer+1
                    spect[idx + stride_inner - stride_outer],                              // inner+1, outer-1
                    spect[idx - stride_inner + stride_outer],                              // inner-1, outer+1
                    spect[idx - stride_inner - stride_outer],                              // inner-1, outer-1
                    spect[idx + stride_intermediate + stride_outer],                        // intermediate+1, outer+1
                    spect[idx + stride_intermediate - stride_outer],                        // intermediate+1, outer-1
                    spect[idx - stride_intermediate + stride_outer],                        // intermediate-1, outer+1
                    spect[idx - stride_intermediate - stride_outer]                         // intermediate-1, outer-1
                };

                // Count how many sqrt(2) neighbors are smaller than val
                int count_smaller_sqrt2 = 0;
                for (int n = 0; n < 12; n++) {
                    if (val > neighbors_sqrt2[n]) {
                        count_smaller_sqrt2++;
                    }
                }

                // Need at least 12 out of 12 sqrt(2) neighbors
                if (count_smaller_sqrt2 < 12) {
                    continue;
                }

                // Check sqrt(3) distance neighbors (vertex/corner diagonal neighbors)
                // 8 total: all 8 corners of the unit cube (±1, ±1, ±1)
                float neighbors_sqrt3[8] = {
                    spect[idx + stride_inner + stride_intermediate + stride_outer],      // +1,+1,+1
                    spect[idx + stride_inner + stride_intermediate - stride_outer],      // +1,+1,-1
                    spect[idx + stride_inner - stride_intermediate + stride_outer],      // +1,-1,+1
                    spect[idx + stride_inner - stride_intermediate - stride_outer],      // +1,-1,-1
                    spect[idx - stride_inner + stride_intermediate + stride_outer],      // -1,+1,+1
                    spect[idx - stride_inner + stride_intermediate - stride_outer],      // -1,+1,-1
                    spect[idx - stride_inner - stride_intermediate + stride_outer],      // -1,-1,+1
                    spect[idx - stride_inner - stride_intermediate - stride_outer]       // -1,-1,-1
                };

                // Count how many sqrt(3) neighbors are smaller than val
                int count_smaller_sqrt3 = 0;
                for (int n = 0; n < 8; n++) {
                    if (val > neighbors_sqrt3[n]) {
                        count_smaller_sqrt3++;
                    }
                }

                // Need at least 6 out of 8 sqrt(3) neighbors to be smaller
                if (count_smaller_sqrt3 < 6) {
                    continue;
                }

                // This is a peak - store it in storage-order x/y/z.
                peaks.push_back(std::make_tuple(inner, intermediate, outer, val));
            }
        }
    }

    // Write peaks to file
    // Cache picked peaks in the class member so other routines can reuse them without re-reading the file
    picked_peaks_all = peaks;

    if(outfname == "" || outfname == "none") {
        std::cout << "No output file specified for peak picking results. Skipping file writing." << std::endl;
        return true;
    }

    std::ofstream outfile(outfname);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open output file " << outfname << std::endl;
        return false;
    }

    // Write header
    outfile << "# 3D Peak Picking Results" << std::endl;
    outfile << "# Threshold: " << threshold << " (" << scale1 << " * noise_level)" << std::endl;
    outfile << "# Noise Level: " << noise_level << std::endl;
    outfile << "# Dimensions: F2=" << ndata_frq << ", F1=" << ndata_frq_indirect1 << ", F3=" << ndata_frq_indirect2 << std::endl;
    outfile << "# Total Peaks Found: " << peaks.size() << std::endl;
    outfile << std::endl;
    outfile << "X_Center\tY_Center\tZ_Center\tX_Center_ppm\tY_Center_ppm\tZ_Center_ppm\tHeight" << std::endl;
    outfile << "========\t========\t========\t===========\t===========\t===========\t======" << std::endl;

    // Write each peak
    for (const auto &peak : peaks) {
        int x = std::get<0>(peak);
        int y = std::get<1>(peak);
        int z = std::get<2>(peak);
        float height = std::get<3>(peak);

        // Calculate ppm using the current storage-axis order.
        const double begin_by_phys_axis[3] = {begin1, begin2, begin3};
        const double step_by_phys_axis[3] = {step1, step2, step3};
        const double ppm_begin_x = begin_by_phys_axis[current_axis_order[0]];
        const double ppm_step_x = step_by_phys_axis[current_axis_order[0]];
        const double ppm_begin_y = begin_by_phys_axis[current_axis_order[1]];
        const double ppm_step_y = step_by_phys_axis[current_axis_order[1]];
        const double ppm_begin_z = begin_by_phys_axis[current_axis_order[2]];
        const double ppm_step_z = step_by_phys_axis[current_axis_order[2]];

        double ppm_x = ppm_begin_x + x * ppm_step_x;
        double ppm_y = ppm_begin_y + y * ppm_step_y;
        double ppm_z = ppm_begin_z + z * ppm_step_z;

        outfile << x << "\t" << y << "\t" << z << "\t"
                << std::fixed << std::setprecision(6)
            << ppm_x << "\t" << ppm_y << "\t" << ppm_z << "\t"
                << height << std::endl;
    }

    outfile.close();
    if (verbose_level >= 1) {
        std::cout << "Peak picking complete. Found " << peaks.size() << " peaks." << std::endl;
        if (!outfname.empty()) {
            std::cout << "Results written to " << outfname << std::endl;
        }
    }

    return true;
}

bool spectrum_fit_3d::partition_signal_regions(std::string outfname)
{
    if (!spect) {
        std::cerr << "Error: Spectrum data not loaded." << std::endl;
        return false;
    }

    if (ndata_frq <= 0 || ndata_frq_indirect1 <= 0 || ndata_frq_indirect2 <= 0) {
        std::cerr << "Error: Invalid dimensions for region partitioning." << std::endl;
        return false;
    }

    partition_threshold = scale2 * noise_level;
    size_t total_points = (size_t)ndata_frq * ndata_frq_indirect1 * ndata_frq_indirect2;

    // Count signal voxels
    int signal_voxel_count = 0;
    for (size_t idx = 0; idx < total_points; idx++) {
        if (spect[idx] > partition_threshold) signal_voxel_count++;
    }

    const int ndata_by_physical_axis[3] = {ndata_frq, ndata_frq_indirect1, ndata_frq_indirect2};
    const int ndata_inner = ndata_by_physical_axis[current_axis_order[0]];
    const int ndata_intermediate = ndata_by_physical_axis[current_axis_order[1]];
    const int ndata_outer = ndata_by_physical_axis[current_axis_order[2]];

    if (verbose_level >= 1) {
        std::cout << "Signal threshold: " << partition_threshold
                << " (" << scale2 << " * noise_level)" << std::endl;
        std::cout << "Signal voxel count: " << signal_voxel_count << " out of " << total_points << std::endl;
        std::cout << "Storage order sizes: inner=" << ndata_inner << ", intermediate=" << ndata_intermediate << ", outer=" << ndata_outer << std::endl;
    }

    // Create RegionLabeler3D and process
    RegionLabeler3D labeler(ndata_inner, ndata_intermediate, ndata_outer);
    labeler.process(spect, partition_threshold, partition_labels);
    const std::vector<int>& labels = partition_labels;

    partitionArray.clear();
    std::map<int, size_t> label_to_index;
    
    int max_partition_idx = -1;
    double max_total_intensity = 0.0;
    int max_label = 0;


    int point_partition_idx = -1;
    bool logged_partition_point = false;
    for (int z = 0; z < ndata_outer; ++z) {
        for (int y = 0; y < ndata_intermediate; ++y) {
            for (int x = 0; x < ndata_inner; ++x) {
                size_t idx = (static_cast<size_t>(z) * static_cast<size_t>(ndata_intermediate) + static_cast<size_t>(y)) * static_cast<size_t>(ndata_inner) + static_cast<size_t>(x);
                int label = labels[idx];
                if (label > 0) {
                    if (label > max_label) {
                        max_label = label;
                    }
                    auto found = label_to_index.find(label);
                    if (found == label_to_index.end()) {
                        partitionArray.push_back(fid_3d::partition());
                        size_t new_index = partitionArray.size() - 1;
                        label_to_index[label] = new_index;
                        found = label_to_index.find(label);

                        // Initialize min/max coords on first occurrence
                        partitionArray[new_index].min_x = x;
                        partitionArray[new_index].max_x = x;
                        partitionArray[new_index].min_y = y;
                        partitionArray[new_index].max_y = y;
                        partitionArray[new_index].min_z = z;
                        partitionArray[new_index].max_z = z;
                    }

                    auto& coor = partitionArray[found->second];
                    coor.x.push_back(x);
                    coor.y.push_back(y);
                    coor.z.push_back(z);

                    float val = spect[idx];
                    coor.total_intensity += val;
                    coor.max_intensity = std::max(coor.max_intensity, val);

                    // Update min/max coordinates
                    coor.min_x = std::min(coor.min_x, x);
                    coor.max_x = std::max(coor.max_x, x);
                    coor.min_y = std::min(coor.min_y, y);
                    coor.max_y = std::max(coor.max_y, y);
                    coor.min_z = std::min(coor.min_z, z);
                    coor.max_z = std::max(coor.max_z, z);

                    // Track max intensity partition
                    if (coor.total_intensity > max_total_intensity) {
                        max_total_intensity = coor.total_intensity;
                        max_partition_idx = (int)found->second;
                    }

                    if (!logged_partition_point && x == 187 && y == 49 && z == 27) {
                        point_partition_idx = (int)found->second;
                        std::cout << "Debug: point [187,49,27] belongs to raw partition "
                                  << found->second << std::endl;
                        logged_partition_point = true;
                    }
                }
            }
        }
    }

    partition_label_to_index.assign(max_label + 1, -1);
    for (const auto& pair : label_to_index) {
        if (pair.first >= 0 && pair.first < (int)partition_label_to_index.size()) {
            partition_label_to_index[pair.first] = (int)pair.second;
        }
    }

    // Filter partitions by minimum extent in all dimensions
    std::vector<int> valid_indices;
    int max_extent = 3;
    int max_valid_idx = -1;
    double max_valid_intensity = 0.0;
    
    for (size_t p_idx = 0; p_idx < partitionArray.size(); p_idx++) {
        const auto& coor = partitionArray[p_idx];
        if (coor.x.empty()) continue;
        
        int extent_x = coor.max_x - coor.min_x;
        int extent_y = coor.max_y - coor.min_y;
        int extent_z = coor.max_z - coor.min_z;
        
        if (extent_x >= max_extent && extent_y >= max_extent && extent_z >= max_extent) {
            valid_indices.push_back((int)p_idx);
            if (coor.total_intensity > max_valid_intensity) {
                max_valid_intensity = coor.total_intensity;
                max_valid_idx = (int)p_idx;
            }
        }
    }

    if (point_partition_idx >= 0) {
        if (point_partition_idx < (int)partitionArray.size()) {
            const auto& dbg_coor = partitionArray[point_partition_idx];
            std::cout << "Debug: point [187,49,27] partition bounds X[" << dbg_coor.min_x << ", "
                      << dbg_coor.max_x << "] Y[" << dbg_coor.min_y << ", " << dbg_coor.max_y
                      << "] Z[" << dbg_coor.min_z << ", " << dbg_coor.max_z << "]" << std::endl;
        }
        bool in_valid = std::find(valid_indices.begin(), valid_indices.end(), point_partition_idx) != valid_indices.end();
        if (in_valid) {
            std::cout << "Debug: point [187,49,27] partition index in 3d_signal_regions.txt is "
                      << point_partition_idx << std::endl;
        } else {
            std::cout << "Debug: point [187,49,27] partition " << point_partition_idx
                      << " is filtered out and not listed in 3d_signal_regions.txt" << std::endl;
        }
    }

    if (verbose_level >= 1) {
        std::cout << "Partitions after filtering (extent >= 5 in all dimensions): " << valid_indices.size() << std::endl;
    }

    // Write maximal intensity partition to debug file (if any valid partitions exist)
#ifdef SPECTRUM_FIT_3D_DEBUG
    std::string debug_fname = "max_partition_debug.txt";
    std::ofstream debug_file(debug_fname);
    if (!debug_file.is_open()) {
        std::cerr << "Warning: Cannot open debug file " << debug_fname << std::endl;
    } else {
        if (max_valid_idx < 0) {
            debug_file << "# No valid partitions found after filtering" << std::endl;
            debug_file << "# Minimum extent required: 5 voxels in all dimensions" << std::endl;
        } else {
            const auto& max_coor = partitionArray[max_valid_idx];
            debug_file << "# Max Intensity Partition Debug Output (After Filtering)" << std::endl;
            debug_file << "# Index: " << max_valid_idx << std::endl;
            debug_file << "# Total Intensity: " << std::fixed << std::setprecision(6) << max_coor.total_intensity << std::endl;
            debug_file << "# Max Intensity (single voxel): " << max_coor.max_intensity << std::endl;
            debug_file << "# Voxel Count: " << max_coor.x.size() << std::endl;
            debug_file << "# Min Coords (x,y,z): (" << max_coor.min_x << ", " 
                      << max_coor.min_y << ", " << max_coor.min_z << ")" << std::endl;
            debug_file << "# Max Coords (x,y,z): (" << max_coor.max_x << ", " 
                      << max_coor.max_y << ", " << max_coor.max_z << ")" << std::endl;
            int extent_x = max_coor.max_x - max_coor.min_x;
            int extent_y = max_coor.max_y - max_coor.min_y;
            int extent_z = max_coor.max_z - max_coor.min_z;
            debug_file << "# Extents (dx,dy,dz): (" << extent_x << ", " << extent_y << ", " << extent_z << ")" << std::endl;
            debug_file << "# Format: x y z intensity" << std::endl;
            debug_file << std::endl;

            // Write all voxels in max partition
            for (size_t v_idx = 0; v_idx < max_coor.x.size(); v_idx++) {
                int i = max_coor.x[v_idx];
                int j = max_coor.y[v_idx];
                int k = max_coor.z[v_idx];
                size_t idx = (static_cast<size_t>(k) * static_cast<size_t>(ndata_intermediate) + static_cast<size_t>(j)) * static_cast<size_t>(ndata_inner) + static_cast<size_t>(i);
                debug_file << i << "\t" << j << "\t" << k << "\t"
                          << std::fixed << std::setprecision(6) << spect[idx] << std::endl;
            }
        }
        debug_file.close();
        std::cout << "Max partition debug written to " << debug_fname << std::endl;
    }
#endif

    // If output filename is provided, write summary of valid regions
    if (outfname!="" && outfname!="none" && !outfname.empty()) {
        std::vector<std::tuple<int, int, int>> picked_peaks;
        picked_peaks.reserve(picked_peaks_all.size());
        for (const auto &t : picked_peaks_all) {
            picked_peaks.emplace_back(std::get<0>(t), std::get<1>(t), std::get<2>(t));
        }

        // --- Build a map from (i,j,k) to partition index for fast lookup ---
        std::map<std::tuple<int, int, int>, int> voxel_to_partition;
        for (int idx : valid_indices) {
            const auto& coor = partitionArray[idx];
            for (size_t v = 0; v < coor.x.size(); ++v) {
                voxel_to_partition[std::make_tuple(coor.x[v], coor.y[v], coor.z[v])] = idx;
            }
        }

        // --- Count number of picked peaks in each partition ---
        std::map<int, int> partition_peak_count;
        for (const auto& peak : picked_peaks) {
            auto it = voxel_to_partition.find(peak);
            if (it != voxel_to_partition.end()) {
                partition_peak_count[it->second]++;
            }
        }

        std::ofstream outfile(outfname);
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open output file " << outfname << std::endl;
            return false;
        }

        outfile << "# 3D Signal Region Partition Results (After Filtering)" << std::endl;
        outfile << "# Signal Threshold: " << partition_threshold << " (" << scale2 << " * noise_level)" << std::endl;
        outfile << "# Noise Level: " << noise_level << std::endl;
        outfile << "# Dimensions: F3=" << ndata_frq << ", F2=" << ndata_frq_indirect1 << ", F1=" << ndata_frq_indirect2 << std::endl;
        outfile << "# Total Signal Voxels: " << signal_voxel_count << std::endl;
        outfile << "# Total Partitions Before Filter: " << partitionArray.size() << std::endl;
        outfile << "# Valid Partitions After Filter (extent >= 5 in all dims): " << valid_indices.size() << std::endl;
        outfile << std::endl;
        outfile << "Index\tTotal_Intensity\tMax_Intensity\tVoxel_Count\tMin_X\tMax_X\tMin_Y\tMax_Y\tMin_Z\tMax_Z\tExtent_X\tExtent_Y\tExtent_Z\tPeak_Count" << std::endl;
        outfile << "=====\t===============\t=============\t===========\t=====\t=====\t=====\t=====\t=====\t=====\t========\t========\t========\t==========" << std::endl;

        for (int idx : valid_indices) {
            const auto& coor = partitionArray[idx];
            int extent_x = coor.max_x - coor.min_x;
            int extent_y = coor.max_y - coor.min_y;
            int extent_z = coor.max_z - coor.min_z;
            int peak_count = partition_peak_count.count(idx) ? partition_peak_count[idx] : 0;
            outfile << idx << "\t"
                   << std::scientific << std::setprecision(6) << coor.total_intensity << "\t"
                   << coor.max_intensity << "\t"
                   << std::fixed << coor.x.size() << "\t"
                   << coor.min_x << "\t" << coor.max_x << "\t"
                   << coor.min_y << "\t" << coor.max_y << "\t"
                   << coor.min_z << "\t" << coor.max_z << "\t"
                   << extent_x << "\t" << extent_y << "\t" << extent_z << "\t"
                   << peak_count << std::endl;
            // Reset to fixed for other columns if needed
            // outfile << std::fixed; // Removed erroneous line
        }

        outfile.close();
        std::cout << "Region partition results written to " << outfname << std::endl;
    }

    // === SORT PARTITIONS BY MAX INTENSITY (HIGH TO LOW) ===
    // Build mapping of old index to sort rank
    std::vector<std::pair<int, float>> idx_max_intensity;
    for (size_t p_idx = 0; p_idx < partitionArray.size(); p_idx++) {
        idx_max_intensity.push_back(std::make_pair((int)p_idx, partitionArray[p_idx].max_intensity));
    }
    
    // Sort by max_intensity descending
    std::sort(idx_max_intensity.begin(), idx_max_intensity.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });
    
    // Create mapping from old index to new index
    std::map<int, int> old_to_new_idx;
    partition_old_to_new.assign(idx_max_intensity.size(), -1);
    partition_new_to_old.assign(idx_max_intensity.size(), -1);
    for (size_t new_idx = 0; new_idx < idx_max_intensity.size(); new_idx++) {
        int old_idx = idx_max_intensity[new_idx].first;
        old_to_new_idx[old_idx] = (int)new_idx;
        partition_old_to_new[old_idx] = (int)new_idx;
        partition_new_to_old[new_idx] = old_idx;
    }
    
    // Create sorted partition array
    std::vector<fid_3d::partition> sorted_partitionArray;
    for (const auto& pair : idx_max_intensity) {
        sorted_partitionArray.push_back(partitionArray[pair.first]);
    }
    partitionArray = sorted_partitionArray;

    std::vector<int> label_to_index_before_sort = partition_label_to_index;

    // Remap partition_labels to use new indices
    for (size_t idx = 0; idx < partition_labels.size(); idx++) {
        int old_label = partition_labels[idx];
        if (old_label > 0) {
            int old_idx = (old_label < (int)label_to_index_before_sort.size())
                              ? label_to_index_before_sort[old_label]
                              : -1;
            if (old_idx >= 0) {
                int new_idx = partition_old_to_new[old_idx];
                if (new_idx >= 0) {
                    partition_labels[idx] = new_idx + 1;  // Convert back to 1-indexed
                }
            }
        }
    }

    // Remap label->index mapping to sorted order
    if (!partition_label_to_index.empty()) {
        for (size_t label = 1; label < partition_label_to_index.size(); label++) {
            int old_idx = label_to_index_before_sort[label];
            if (old_idx >= 0 && old_idx < (int)partition_old_to_new.size()) {
                partition_label_to_index[label] = partition_old_to_new[old_idx];
            }
        }
    }

    if (verbose_level >= 1) {
        std::cout << "Partitions sorted by max intensity (descending)" << std::endl;
    }

    return true;
}

bool spectrum_fit_3d::dump_partition_voxels(int partition_idx, const std::string& outfname) const
{
    if (!spect) {
        std::cerr << "Error: Spectrum data not loaded." << std::endl;
        return false;
    }

    if (!partition_old_to_new.empty()) {
        if (partition_idx < 0 || partition_idx >= (int)partition_old_to_new.size()) {
            std::cerr << "Error: signal-region index " << partition_idx
                      << " out of range [0, " << (partition_old_to_new.empty() ? 0 : (int)partition_old_to_new.size() - 1)
                      << "]." << std::endl;
            return false;
        }
        partition_idx = partition_old_to_new[partition_idx];
    }

    if (partition_idx < 0 || partition_idx >= (int)partitionArray.size()) {
        std::cerr << "Error: partition index " << partition_idx
                  << " out of range [0, " << (partitionArray.empty() ? 0 : (int)partitionArray.size() - 1)
                  << "]." << std::endl;
        return false;
    }

    std::ofstream out(outfname);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open output file: " << outfname << std::endl;
        return false;
    }

    const auto& coor = partitionArray[partition_idx];
    out << "# Partition " << partition_idx << " voxel list (storage-order indices)" << std::endl;
    out << "# X\tY\tZ\tIntensity" << std::endl;

    const int ndata_by_physical_axis[3] = {ndata_frq, ndata_frq_indirect1, ndata_frq_indirect2};
    const int ndata_inner = ndata_by_physical_axis[current_axis_order[0]];
    const int ndata_intermediate = ndata_by_physical_axis[current_axis_order[1]];

    for (size_t i = 0; i < coor.x.size(); i++) {
        int x = coor.x[i];
        int y = coor.y[i];
        int z = coor.z[i];
        size_t idx = (static_cast<size_t>(z) * static_cast<size_t>(ndata_intermediate)
                      + static_cast<size_t>(y)) * static_cast<size_t>(ndata_inner)
                      + static_cast<size_t>(x);
        out << x << "\t" << y << "\t" << z << "\t" << std::setprecision(8) << spect[idx] << std::endl;
    }

    if (verbose_level >= 1) {
        std::cout << "Wrote " << coor.x.size() << " voxels for partition "
                << partition_idx << " to " << outfname << std::endl;
    }
    return true;
}

/**
 * Calculate 3D pseudo-Voigt peak model. 
 * Input x,y,z cooridnate and peak parameters, output model value at that coordinate for that peak.
 * The model is a weighted combination of Gaussian and Lorentzian:
 */

bool spectrum_fit_3d::iterative_fit_one_partition(int partition_idx)
{
    if (!spect) {
        std::cerr << "Error: Spectrum data not loaded." << std::endl;
        return false;
    }

    if (ndata_frq <= 0 || ndata_frq_indirect1 <= 0 || ndata_frq_indirect2 <= 0) {
        std::cerr << "Error: Invalid dimensions for iterative fitting." << std::endl;
        return false;
    }

    size_t total_points = (size_t)ndata_frq * ndata_frq_indirect1 * ndata_frq_indirect2;
    if (partition_labels.size() != total_points) {
        std::cerr << "Error: Partition labels not available. Run partition_signal_regions first." << std::endl;
        return false;
    }
    
    if (partitionArray.empty()) {
        std::cerr << "Error: Partition coordinates not available. Run partition_signal_regions first." << std::endl;
        return false;
    }
    
    int target_partition_idx = partition_idx;
    
    // Validate partition exists and has valid data
    if (target_partition_idx < 0 || target_partition_idx >= (int)partitionArray.size()) {
        std::cerr << "Error: Partition index " << target_partition_idx << " out of range [0, " 
                  << partitionArray.size()-1 << "]." << std::endl;
        return false;
    }
    
    const auto& coor = partitionArray[target_partition_idx];
    if (coor.x.empty()) {
        std::cerr << "Error: Partition " << target_partition_idx << " is empty." << std::endl;
        return false;
    }
    
    // Check extent requirements (must have >= 5 in all dimensions)
    int span_x = coor.max_x - coor.min_x;
    int span_y = coor.max_y - coor.min_y;
    int span_z = coor.max_z - coor.min_z;

    if (span_x < 3 || span_y < 3 || span_z < 3) {
        std::cerr << "Error: Partition does not meet extent requirements (min 3 in all dims)." << std::endl;
        std::cerr << "  Spans: X=" << span_x << ", Y=" << span_y << ", Z=" << span_z << std::endl;
        return false;
    }
    
    if (verbose_level >= 1) {
        std::cout << "\nStarting iterative fitting on partition " << target_partition_idx << std::endl;
        std::cout << "Coordinates: X[" << coor.min_x << ", " << coor.max_x
                << "], Y[" << coor.min_y << ", " << coor.max_y
                << "], Z[" << coor.min_z << ", " << coor.max_z << "]" << std::endl;
        std::cout << "Span: X=" << span_x << ", Y=" << span_y << ", Z=" << span_z << std::endl;
        std::cout << "Total Intensity: " << std::fixed << std::setprecision(6) << coor.total_intensity << std::endl;
        std::cout << "Max Intensity (single voxel): " << coor.max_intensity << std::endl;
    }
    
    // Extract only the actual partition voxels (sparse)
    int x0 = coor.min_x;
    int y0 = coor.min_y;
    int z0 = coor.min_z;
    
    std::vector<float> partition_data;
    std::vector<int> x_coords, y_coords, z_coords;

    // x/y/z are already storage-order coordinates. Only ppm conversion needs axis-order mapping.
    const double axis_begin[3] = {begin1, begin2, begin3};
    const double axis_step[3] = {step1, step2, step3};
    const double storage_begin[3] = {
        axis_begin[current_axis_order[0]],
        axis_begin[current_axis_order[1]],
        axis_begin[current_axis_order[2]]
    };
    const double storage_step[3] = {
        axis_step[current_axis_order[0]],
        axis_step[current_axis_order[1]],
        axis_step[current_axis_order[2]]
    };

    const int size_by_physical_axis[3] = {ndata_frq, ndata_frq_indirect1, ndata_frq_indirect2};
    const int nx = size_by_physical_axis[current_axis_order[0]];
    const int ny = size_by_physical_axis[current_axis_order[1]];
    const int nz = size_by_physical_axis[current_axis_order[2]];

    // Extract only the voxels explicitly in the partition
    for (size_t v_idx = 0; v_idx < coor.x.size(); v_idx++) {
        int i = coor.x[v_idx];
        int j = coor.y[v_idx];
        int k = coor.z[v_idx];
        size_t idx_global = (static_cast<size_t>(k) * static_cast<size_t>(ny) + static_cast<size_t>(j)) * static_cast<size_t>(nx) + static_cast<size_t>(i);

        partition_data.push_back(spect[idx_global]);
        x_coords.push_back(i - x0);  // local coordinate
        y_coords.push_back(j - y0);  // local coordinate
        z_coords.push_back(k - z0);  // local coordinate
    }

    std::vector<Peak3D> pre_fit_peaks = prefit_partition_peaks(partition_data, x_coords, y_coords, z_coords, coor,
                                                                storage_begin[0], storage_step[0], storage_begin[1], storage_step[1], storage_begin[2], storage_step[2],
                                                                x0, y0, z0);

    // Convert local coordinates to absolute and set partition index before global storage
    for (auto& peak : pre_fit_peaks) {
        peak.partition_idx = target_partition_idx;
        peak.x_center += (float)x0;
        peak.y_center += (float)y0;
        peak.z_center += (float)z0;
    }

    /**
     * Append pre-fit peaks to the global list. 
     */                                                            
    fitted_peaks_all.insert(fitted_peaks_all.end(), pre_fit_peaks.begin(), pre_fit_peaks.end());                                                   

    // If no peaks were found in the partition, skip fitting.
    if (pre_fit_peaks.empty()) {
        if(verbose_level >= 1) {
            std::cerr << "No peaks detected in partition " << target_partition_idx << ". Skipping fitting." << std::endl;
        }
        return false;
    }
    
    return true;
}

bool spectrum_fit_3d::iterative_fit_all_partitions(int top_n)
{
    if (!spect) {
        std::cerr << "Error: Spectrum data not loaded." << std::endl;
        return false;
    }

    if (top_n <= 0) {
        std::cerr << "Error: top_n must be > 0." << std::endl;
        return false;
    }

    if (partitionArray.empty()) {
        std::cerr << "Error: No partitions available. Run partition_signal_regions first." << std::endl;
        return false;
    }

    int num_valid_partitions = 0;
    int num_successful_partitions = 0;
    int num_failed_partitions = 0;

    if (verbose_level >= 1) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Fitting Top-N Partitions (Pre-Sorted by Max Intensity)" << std::endl;
        std::cout << "Total partitions: " << partitionArray.size() << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    std::vector<int> eligible_partitions;
    eligible_partitions.reserve(partitionArray.size());

    const int ndata_by_physical_axis[3] = {ndata_frq, ndata_frq_indirect1, ndata_frq_indirect2};
    const int ndata_inner = ndata_by_physical_axis[current_axis_order[0]];
    const int ndata_intermediate = ndata_by_physical_axis[current_axis_order[1]];
    const int ndata_outer = ndata_by_physical_axis[current_axis_order[2]];

    std::vector<bool> partition_has_peak(partitionArray.size(), false);
    for (const auto& peak : picked_peaks_all) {
        int i = std::get<0>(peak);
        int j = std::get<1>(peak);
        int k = std::get<2>(peak);
        if (i < 0 || i >= ndata_inner || j < 0 || j >= ndata_intermediate || k < 0 || k >= ndata_outer) {
            continue;
        }
        size_t idx = (static_cast<size_t>(k) * static_cast<size_t>(ndata_intermediate) + static_cast<size_t>(j))
                     * static_cast<size_t>(ndata_inner) + static_cast<size_t>(i);
        if (idx >= partition_labels.size()) continue;
        int label = partition_labels[idx];
        if (label > 0) {
            int p_idx = label - 1;
            if (p_idx >= 0 && p_idx < (int)partition_has_peak.size()) {
                partition_has_peak[p_idx] = true;
            }
        }
    }

    // Count partitions that pass the partition test and contain at least one picked peak.
    for (size_t p_idx = 0; p_idx < partitionArray.size(); p_idx++) {
        const auto& coor = partitionArray[p_idx];
        if (coor.x.empty()) continue;

        int extent_x = coor.max_x - coor.min_x;
        int extent_y = coor.max_y - coor.min_y;
        int extent_z = coor.max_z - coor.min_z;
        if (extent_x < 3 || extent_y < 3 || extent_z < 3) continue;

        if (!partition_has_peak[p_idx]) continue;

        eligible_partitions.push_back((int)p_idx);
    }

    num_valid_partitions = (int)eligible_partitions.size();

    if (num_valid_partitions == 0) {
        std::cerr << "Error: No partitions passing the filter with at least one picked peak." << std::endl;
        return false;
    }

    int num_skipped_partitions = num_valid_partitions - std::min(top_n, num_valid_partitions);

    if (verbose_level >= 1) {
        std::cout << "Valid partitions (extent>=3 and >=1 peak): " << num_valid_partitions << std::endl;
        std::cout << "Requested top_n: " << top_n << std::endl;
        std::cout << "Partitions to fit: " << std::min(top_n, num_valid_partitions) << std::endl;
        std::cout << "Starting iterative fitting on top-ranked partitions...\n" << std::endl;
    }

    // Fit top-N partitions (already sorted by max intensity)
    int num_fitted = 0;
    fitted_peaks_all.clear();
    fitted_peaks_all.reserve(picked_peaks_all.size()); // Pre-allocate based on picked peaks, actual fitted peaks may be fewer.
    for (size_t idx = 0; idx < eligible_partitions.size() && num_fitted < top_n; idx++) {
        int p_idx = eligible_partitions[idx];
        const auto& coor = partitionArray[p_idx];

        num_fitted++;
        if (verbose_level >= 1) {
            std::cout << "\n--- Processing Partition " << p_idx
                    << " (rank " << num_fitted << ", max=" << coor.max_intensity << ") ---" << std::endl;
        }

        bool success = iterative_fit_one_partition(p_idx);

        if (success) {
            num_successful_partitions++;
            if (verbose_level >= 1) std::cout << "✓ Partition " << p_idx << " completed successfully." << std::endl;
        } else {
            num_failed_partitions++;
            if (verbose_level >= 1) std::cerr << "✗ Partition " << p_idx << " failed or did not meet requirements." << std::endl;
        }
    }

    if (verbose_level >= 1) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Summary:" << std::endl;
        std::cout << "  Total partitions: " << partitionArray.size() << std::endl;
        std::cout << "  Valid partitions: " << num_valid_partitions << std::endl;
        std::cout << "  Requested top_n: " << top_n << std::endl;
        std::cout << "  Attempted fits: " << std::min(top_n, num_valid_partitions) << std::endl;
        std::cout << "  Successful: " << num_successful_partitions << std::endl;
        std::cout << "  Failed: " << num_failed_partitions << std::endl;
        std::cout << "  Skipped (outside top_n): " << num_skipped_partitions << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    return num_successful_partitions > 0;
}

std::string spectrum_fit_3d::get_fitted_peaks_string() const
{
    std::stringstream ss;
    ss << "# Pre-Fitted 3D Peaks (No Regularization)\n";
    ss << "Partition\tPeak\tAmplitude\tX_Center\tY_Center\tZ_Center\tX_Center_ppm\tY_Center_ppm\tZ_Center_ppm\tX_FWHH\t\tY_FWHH\t\tZ_FWHH\t\tLx\tLy\tLz\n";
    ss << "=========\t====\t=========\t========\t========\t========\t======\t\t======\t\t======\t\t=======\t\t=======\t\t=======\t\t==\t==\t==\n";

    const double axis_begin[3] = {begin1, begin2, begin3};
    const double axis_step[3] = {step1, step2, step3};
    const double storage_begin[3] = {
        axis_begin[current_axis_order[0]],
        axis_begin[current_axis_order[1]],
        axis_begin[current_axis_order[2]]
    };
    const double storage_step[3] = {
        axis_step[current_axis_order[0]],
        axis_step[current_axis_order[1]],
        axis_step[current_axis_order[2]]
    };

    int last_partition_idx = -1;
    int peak_in_partition_idx = 0;

    for (const auto& peak : fitted_peaks_all) {
        if (peak.partition_idx != last_partition_idx) {
            last_partition_idx = peak.partition_idx;
            peak_in_partition_idx = 1;
        } else {
            peak_in_partition_idx++;
        }

        double ppm_x = storage_begin[0] + peak.x_center * storage_step[0];
        double ppm_y = storage_begin[1] + peak.y_center * storage_step[1];
        double ppm_z = storage_begin[2] + peak.z_center * storage_step[2];

        char buf[512];
        snprintf(buf, sizeof(buf), "%d\t%d\t%.6e\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\t\t%.6e\t\t%.6e\t\t%.2f\t%.2f\t%.2f\n",
                 peak.partition_idx, peak_in_partition_idx, peak.amplitude,
                 peak.x_center, peak.y_center, peak.z_center,
                 ppm_x, ppm_y, ppm_z,
                 peak.x_width, peak.y_width, peak.z_width,
                 peak.x_lfrac, peak.y_lfrac, peak.z_lfrac);
        ss << buf;
    }

    return ss.str();
}





/**
 * Pre-fit all peaks in partition together (no regularization)
 */
std::vector<Peak3D> spectrum_fit_3d::prefit_partition_peaks(
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    const fid_3d::partition& coor,
    double begin1, double step1,
    double begin2, double step2,
    double begin3, double step3,
    int x0, int y0, int z0)
{
    /**
     * This is a debug flag to limit the number of peaks we consider for pre-fitting, to speed up testing.
     * This will be changed to std::limit<int>::max() for the final version, to consider all picked peaks in the partition.
     */
    const int max_peaks = 50000;
    /**
     * Final pre-fit peaks will be stored in pre_fit_peaks. We will initialize them from the picked peaks that are within the partition coordinates,
     *  and then run EM rounds to refine their parameters. 
     * The picked peaks are filtered to only include those that fall within the partition's coordinate bounds, 
     * and then sorted by amplitude to keep only the top ones for pre-fitting.
     */
    std::vector<Peak3D> pre_fit_peaks;
    std::vector<std::tuple<int, int, int, float>> picked_peaks_in_partition;
    picked_peaks_in_partition.reserve(picked_peaks_all.size());
    for (const auto &pp : picked_peaks_all) {
        int i = std::get<0>(pp);
        int j = std::get<1>(pp);
        int k = std::get<2>(pp);
        if (i >= coor.min_x && i <= coor.max_x &&
            j >= coor.min_y && j <= coor.max_y &&
            k >= coor.min_z && k <= coor.max_z) {
            picked_peaks_in_partition.push_back(pp);
        }
    }
    
    // Sort by amplitude (descending) and keep only top 60
    std::sort(picked_peaks_in_partition.begin(), picked_peaks_in_partition.end(),
              [](const std::tuple<int, int, int, float>& a, const std::tuple<int, int, int, float>& b) {
                  return std::get<3>(a) > std::get<3>(b);  // Sort by height descending
              });
    if (picked_peaks_in_partition.size() > max_peaks) {
        picked_peaks_in_partition.resize(max_peaks);
    }
    
    if (picked_peaks_in_partition.empty()) {
        return pre_fit_peaks;
    }
    
    // Step 1: Initialize peaks from picked peaks with default pseudo-Voigt parameters
    for (const auto& peak : picked_peaks_in_partition) {
        Peak3D p;
        p.amplitude = std::get<3>(peak);
        p.x_center = (float)(std::get<0>(peak) - coor.min_x); // Convert to local partition coordinates
        p.y_center = (float)(std::get<1>(peak) - coor.min_y); // Convert to local partition coordinates
        p.z_center = (float)(std::get<2>(peak) - coor.min_z); // Convert to local partition coordinates
        p.x_width = 0.8f;  // Default FWHH. Smaller value is preferred by the nonlinear optimizer.
        p.y_width = 0.8f;
        p.z_width = 0.8f;
        p.x_lfrac = 0.05f;  // 50% Lorentzian fraction as default. This can be refined in EM rounds.
        p.y_lfrac = 0.05f;
        p.z_lfrac = 0.05f;
        pre_fit_peaks.push_back(p);
        if (verbose_level >= 2) {
            std::cout << "    Input peak at (" << p.x_center << ", " << p.y_center << ", " << p.z_center 
                    << ") with amp=" << p.amplitude << std::endl;
        }
    }
    
    if (verbose_level >= 1) {
        std::cout << "  Initialized " << pre_fit_peaks.size() << " peaks from picked peaks" << std::endl;
    }
    
    /**
     * We will define a fitting cube that covers the entire partition, and pre-compute the contribution of each peak to this cube for efficient EM updates. 
     * The cube dimensions are determined by the partition's coordinate bounds, and we build a sparse index to quickly access voxel contributions during EM iterations.
     */
    int nx = coor.max_x - coor.min_x + 1;
    int ny = coor.max_y - coor.min_y + 1;
    int nz = coor.max_z - coor.min_z + 1;
    SparseCubeIndex3D prefit_cube = build_sparse_cube_index(x_coords, y_coords, z_coords);
    float prefit_ln2 = std::log(2.0f);
    float prefit_sqrt2ln2 = std::sqrt(2.0f * prefit_ln2);
    
    // EM algorithm: Run rounds of refinement and peak addition
    bool peak_added = true;
    int total_em_rounds = 0;

    /**
     * We keep a list of removed peaks and their initial locations 
     * to prevent new peaks from being added too close to recently removed peaks, 
     * which can cause endless cycles of adding and removing the same peaks.
     */
    size_t total_removed_peaks = 0;
    std::vector<bool> peak_removed(pre_fit_peaks.size(), false);
    std::vector<std::tuple<int, int, int>> peak_initial_locations;
    peak_initial_locations.reserve(pre_fit_peaks.size());
    for (const auto& peak : pre_fit_peaks) {
        peak_initial_locations.emplace_back((int)std::round(peak.x_center),
                                            (int)std::round(peak.y_center),
                                            (int)std::round(peak.z_center));
    }
    std::vector<std::tuple<int, int, int>> removed_peak_initial_locations;

    /**
     * Helper functions to count total active peaks.
     */
    auto active_peak_count = [&]() -> size_t {
        size_t c = 0;
        for (bool removed : peak_removed) {
            if (!removed) {
                c++;
            }
        }
        return c;
    };

    /**
     * Helper function to build final list of active peaks for EM rounds.
     * Used for debug output and final result only.
     */
    auto build_active_peaks = [&]() -> std::vector<Peak3D> {
        std::vector<Peak3D> active;
        active.reserve(pre_fit_peaks.size());
        for (size_t i = 0; i < pre_fit_peaks.size(); ++i) {
            if (!peak_removed[i]) {
                active.push_back(pre_fit_peaks[i]);
            }
        }
        return active;
    };

    /**
     * Helper function to compact the peaks list by removing peaks marked as removed, 
     * and updating the initial locations list accordingly. 
     * called at the end of each EM round after peaks have been marked for removal, 
     * to clean up the lists and prevent them from growing indefinitely.
     */
    auto compact_removed_peaks = [&]() -> size_t {
        std::vector<Peak3D> filtered_peaks;
        std::vector<std::tuple<int, int, int>> filtered_initial_locations;
        filtered_peaks.reserve(pre_fit_peaks.size());
        filtered_initial_locations.reserve(peak_initial_locations.size());
        size_t removed = 0;
        for (size_t i = 0; i < pre_fit_peaks.size(); ++i) {
            if (i < peak_removed.size() && !peak_removed[i]) {
                filtered_peaks.push_back(pre_fit_peaks[i]);
                if (i < peak_initial_locations.size()) {
                    filtered_initial_locations.push_back(peak_initial_locations[i]);
                }
            } else {
                removed++;
            }
        }
        pre_fit_peaks.swap(filtered_peaks);
        peak_initial_locations.swap(filtered_initial_locations);
        peak_removed.assign(pre_fit_peaks.size(), false);
        return removed;
    };

    /**
     * Helper function to check if a given location is within a certain distance (sqrt(3) ~ 1.73) 
     * of any recently removed peak's initial location.
     */
    auto is_near_removed_initial = [&](int x, int y, int z) -> bool {
        for (const auto& loc : removed_peak_initial_locations) {
            int rx = std::get<0>(loc);
            int ry = std::get<1>(loc);
            int rz = std::get<2>(loc);
            int dx = x - rx;
            int dy = y - ry;
            int dz = z - rz;
            if (dx * dx + dy * dy + dz * dz <= 3) {
                return true;
            }
        }
        return false;
    };
    
    /**
     * The main EM loop continues as long as we are adding peaks and 
     * have not exceeded the maximum number of peaks.
     */
    while (peak_added && active_peak_count() <= (size_t)max_peaks) {
        peak_added = false;

         // Step 2: if there is only one peak, we fit it directly to the entire partition data without separation, which is equivalent to a simple non-linear fit.
        if (active_peak_count() == 1)
        {
            size_t peak_idx = 0;
            for (size_t i = 0; i < pre_fit_peaks.size(); ++i)
            {
                if (!peak_removed[i])
                {
                    peak_idx = i;
                    break;
                }
            }
            int nx = coor.max_x - coor.min_x + 1;
            int ny = coor.max_y - coor.min_y + 1;
            int nz = coor.max_z - coor.min_z + 1;
            fit_one_peak(partition_data, x_coords, y_coords, z_coords, nx, ny, nz, pre_fit_peaks[peak_idx]);

            if (verbose_level >= 2)
            {
                std::cout << "    Peak " << (peak_idx + 1) << " refined at idx(" << pre_fit_peaks[peak_idx].x_center + x0 << ", "
                          << pre_fit_peaks[peak_idx].y_center + y0 << ", " << pre_fit_peaks[peak_idx].z_center + z0 << ") amp="
                          << std::scientific << pre_fit_peaks[peak_idx].amplitude << std::endl;
            }

            /**
             * Still need to check if the fitted peak is valid (not too weak, or moved out of bounds).
             */
                if (pre_fit_peaks[peak_idx].amplitude < scale1 * noise_level || // Amplitude too weak
                    pre_fit_peaks[peak_idx].x_center < 1 || pre_fit_peaks[peak_idx].x_center >= nx-1 ||
                    pre_fit_peaks[peak_idx].y_center < 1 || pre_fit_peaks[peak_idx].y_center >= ny-1 ||
                    pre_fit_peaks[peak_idx].z_center < 1 || pre_fit_peaks[peak_idx].z_center >= nz-1) {
                    if (verbose_level >= 2) {
                        std::cout << "    Single peak removed after fitting due to weak amplitude or out-of-bounds: amp=" << std::scientific << pre_fit_peaks[peak_idx].amplitude
                                << ", center=(" << pre_fit_peaks[peak_idx].x_center + x0 << ", "
                                << pre_fit_peaks[peak_idx].y_center + y0 << ", "
                                << pre_fit_peaks[peak_idx].z_center + z0 << ")" << std::endl;
                    }
                    peak_removed[peak_idx] = true;
                    removed_peak_initial_locations.push_back(peak_initial_locations[peak_idx]);
                    total_removed_peaks++;
                } 

            continue; // No need to do contribution separation or watershed for single peak case. No need to run mutiple EM rounds either, since there is no competition between peaks.
        }

        /**
         * For current peak list (without removing or adding peaks yet), 
         * we run a fixed number of EM rounds (20) to refine their parameters.
         */
        const int num_em_rounds = 20;
        for (int round = 0; round < num_em_rounds; round++) {
            total_em_rounds++;
            if (verbose_level >= 2) {
                std::cout << " EM Round " << total_em_rounds << " (Refinement round " << (round + 1) << "/" << num_em_rounds << ")" << std::endl;
                std::cout << " Active peaks: " << active_peak_count() << std::endl;
            }

            // Step 2: Calculate contribution of each peak and separate spectrum
            std::vector<std::vector<float>> peak_contributions(pre_fit_peaks.size());
            std::vector<float> total_model(partition_data.size(), 0.0f);
            
            /**
             * Calculate each peak's contribution to the spectrum
             * Save them into peak_contributions array for each peak
             * And the total to total_model
             */
            for (size_t p = 0; p < pre_fit_peaks.size(); p++) {
                if (peak_removed[p]) {
                    continue;
                }
                peak_contributions[p].assign(partition_data.size(), 0.0f);
                accumulate_peak_model_sparse(peak_contributions[p], pre_fit_peaks[p], prefit_cube, prefit_sqrt2ln2, FwhhFacctorSeperation);
                for (size_t idx = 0; idx < partition_data.size(); ++idx) {
                    total_model[idx] += peak_contributions[p][idx];
                }
            }
            
            // Step 3: Fit each peak individually using LM with pseudo-Voigt profile
            for (size_t p = 0; p < pre_fit_peaks.size(); p++) {
                if (peak_removed[p]) {
                    continue;
                }

                /**
                 * Prepare data for this peak: use weighted portion of spectrum based on peak contribution
                 * peak_data[idx] = (peak_contribution[idx] / total_model[idx]) * partition_data[idx]
                 * This is called (EM appraoch, or Gaussian mixture model).
                 */
                std::vector<float> peak_data(partition_data.size());
                for (size_t idx = 0; idx < partition_data.size(); idx++) {
                    if (total_model[idx] > std::numeric_limits<float>::epsilon()) {  // Avoid division by near-zero
                        peak_data[idx] = (peak_contributions[p][idx] / total_model[idx]) * partition_data[idx];
                    } else {
                        peak_data[idx] = 0.0f;
                    }
                }

                // Compute both watershed and FWHH-based regions, then use their intersection
                std::vector<float> fit_peak_data;
                std::vector<int> fit_x_coords;
                std::vector<int> fit_y_coords;
                std::vector<int> fit_z_coords;
                
                const Peak3D& cur_peak = pre_fit_peaks[p];

                /**
                 * === Compute FWHH-based region with minimum 5 points per direction ===
                 * FWHH region is defined as the set of points within a certain distance from the peak center, where the distance is determined by the peak's widths.
                 * fwhh_region is a set of indices of points in the fitting cube.
                 */
                std::set<int> fwhh_region;
                float sx = std::max(cur_peak.x_width, 1e-6f) * FwhhFacctorFitting;
                float sy = std::max(cur_peak.y_width, 1e-6f) * FwhhFacctorFitting;
                float sz = std::max(cur_peak.z_width, 1e-6f) * FwhhFacctorFitting;
                
                // Ensure at least 5 points along each direction
                sx = std::max(sx, 2.5f);  // 5 points means ±2.5 from center
                sy = std::max(sy, 2.5f);
                sz = std::max(sz, 2.5f);
                
                for (size_t idx = 0; idx < peak_data.size(); ++idx) {
                    if (std::fabs((float)x_coords[idx] - cur_peak.x_center) <= sx &&
                        std::fabs((float)y_coords[idx] - cur_peak.y_center) <= sy &&
                        std::fabs((float)z_coords[idx] - cur_peak.z_center) <= sz) {
                        fwhh_region.insert((int)idx);
                    }
                }

                /**
                 * === Compute watershed region inside the FWHH window ===
                 * Watershed region is defined as the set of points that are connected to the peak center 
                 * and have intensity values that do not drop below a certain threshold (e.g., noise level) 
                 * when moving away from the center.
                 * watershed_region is also a set of indices of points in the fitting cube.
                 */
                std::set<int> watershed_region;
                if (kUseWatershedFittingRegion && !fwhh_region.empty()) {
                    std::vector<float> fwhh_peak_data;
                    std::vector<int> fwhh_x_coords;
                    std::vector<int> fwhh_y_coords;
                    std::vector<int> fwhh_z_coords;
                    std::vector<int> local_to_global;

                    fwhh_peak_data.reserve(fwhh_region.size());
                    fwhh_x_coords.reserve(fwhh_region.size());
                    fwhh_y_coords.reserve(fwhh_region.size());
                    fwhh_z_coords.reserve(fwhh_region.size());
                    local_to_global.reserve(fwhh_region.size());

                    for (int idx : fwhh_region) {
                        fwhh_peak_data.push_back(peak_data[idx]);
                        fwhh_x_coords.push_back(x_coords[idx]);
                        fwhh_y_coords.push_back(y_coords[idx]);
                        fwhh_z_coords.push_back(z_coords[idx]);
                        local_to_global.push_back(idx);
                    }

                    std::set<int> watershed_local_region;
                    watershed_fitting_region(fwhh_peak_data, fwhh_x_coords, fwhh_y_coords, fwhh_z_coords,
                                             cur_peak.x_center, cur_peak.y_center, cur_peak.z_center,
                                             watershed_local_region);

                    for (int local_idx : watershed_local_region) {
                        if (local_idx >= 0 && local_idx < (int)local_to_global.size()) {
                            watershed_region.insert(local_to_global[local_idx]);
                        }
                    }
                }
                
                // === Log peak center and region bounds ===
                if (kDebugDumpEmRoundPeaks) {
                    // Peak center in ppm
                    double x_center_ppm = begin1 + (int)(cur_peak.x_center + x0) * step1;
                    double y_center_ppm = begin2 + (int)(cur_peak.y_center + y0) * step2;
                    double z_center_ppm = begin3 + (int)(cur_peak.z_center + z0) * step3;
                    
                    std::cout << "      Peak " << (p+1) << " center (indices): "
                              << "X=" << std::fixed << std::setprecision(1) << cur_peak.x_center << " "
                              << "Y=" << cur_peak.y_center << " "
                              << "Z=" << cur_peak.z_center << std::endl;
                    std::cout << "      Peak " << (p+1) << " center (ppm): "
                              << "X=" << std::setprecision(3) << x_center_ppm << " "
                              << "Y=" << y_center_ppm << " "
                              << "Z=" << z_center_ppm << std::endl;
                    
                    // FWHH region bounds
                    if (!fwhh_region.empty()) {
                        int fwhh_x_min = INT_MAX, fwhh_x_max = INT_MIN;
                        int fwhh_y_min = INT_MAX, fwhh_y_max = INT_MIN;
                        int fwhh_z_min = INT_MAX, fwhh_z_max = INT_MIN;
                        for (int idx : fwhh_region) {
                            fwhh_x_min = std::min(fwhh_x_min, x_coords[idx]);
                            fwhh_x_max = std::max(fwhh_x_max, x_coords[idx]);
                            fwhh_y_min = std::min(fwhh_y_min, y_coords[idx]);
                            fwhh_y_max = std::max(fwhh_y_max, y_coords[idx]);
                            fwhh_z_min = std::min(fwhh_z_min, z_coords[idx]);
                            fwhh_z_max = std::max(fwhh_z_max, z_coords[idx]);
                        }
                        int gfwhh_x_min = fwhh_x_min + x0, gfwhh_x_max = fwhh_x_max + x0;
                        int gfwhh_y_min = fwhh_y_min + y0, gfwhh_y_max = fwhh_y_max + y0;
                        int gfwhh_z_min = fwhh_z_min + z0, gfwhh_z_max = fwhh_z_max + z0;
                        
                        double fwhh_x_ppm_min = begin1 + gfwhh_x_min * step1;
                        double fwhh_x_ppm_max = begin1 + gfwhh_x_max * step1;
                        double fwhh_y_ppm_min = begin2 + gfwhh_y_min * step2;
                        double fwhh_y_ppm_max = begin2 + gfwhh_y_max * step2;
                        double fwhh_z_ppm_min = begin3 + gfwhh_z_min * step3;
                        double fwhh_z_ppm_max = begin3 + gfwhh_z_max * step3;
                        
                        if (fwhh_x_ppm_min > fwhh_x_ppm_max) std::swap(fwhh_x_ppm_min, fwhh_x_ppm_max);
                        if (fwhh_y_ppm_min > fwhh_y_ppm_max) std::swap(fwhh_y_ppm_min, fwhh_y_ppm_max);
                        if (fwhh_z_ppm_min > fwhh_z_ppm_max) std::swap(fwhh_z_ppm_min, fwhh_z_ppm_max);
                        
                        std::cout << "      Peak " << (p+1) << " FWHH range (indices): "
                                  << "X[" << fwhh_x_min << ", " << fwhh_x_max << "] "
                                  << "Y[" << fwhh_y_min << ", " << fwhh_y_max << "] "
                                  << "Z[" << fwhh_z_min << ", " << fwhh_z_max << "]" << std::endl;
                        std::cout << "      Peak " << (p+1) << " FWHH range (ppm): "
                                  << "X[" << std::setprecision(3) << fwhh_x_ppm_min << ", " << fwhh_x_ppm_max << "] "
                                  << "Y[" << fwhh_y_ppm_min << ", " << fwhh_y_ppm_max << "] "
                                  << "Z[" << fwhh_z_ppm_min << ", " << fwhh_z_ppm_max << "]" << std::endl;
                    }
                    
                    // Watershed region bounds
                    if (!watershed_region.empty()) {
                        int ws_x_min = INT_MAX, ws_x_max = INT_MIN;
                        int ws_y_min = INT_MAX, ws_y_max = INT_MIN;
                        int ws_z_min = INT_MAX, ws_z_max = INT_MIN;
                        for (int idx : watershed_region) {
                            ws_x_min = std::min(ws_x_min, x_coords[idx]);
                            ws_x_max = std::max(ws_x_max, x_coords[idx]);
                            ws_y_min = std::min(ws_y_min, y_coords[idx]);
                            ws_y_max = std::max(ws_y_max, y_coords[idx]);
                            ws_z_min = std::min(ws_z_min, z_coords[idx]);
                            ws_z_max = std::max(ws_z_max, z_coords[idx]);
                        }
                        int gws_x_min = ws_x_min + x0, gws_x_max = ws_x_max + x0;
                        int gws_y_min = ws_y_min + y0, gws_y_max = ws_y_max + y0;
                        int gws_z_min = ws_z_min + z0, gws_z_max = ws_z_max + z0;
                        
                        double ws_x_ppm_min = begin1 + gws_x_min * step1;
                        double ws_x_ppm_max = begin1 + gws_x_max * step1;
                        double ws_y_ppm_min = begin2 + gws_y_min * step2;
                        double ws_y_ppm_max = begin2 + gws_y_max * step2;
                        double ws_z_ppm_min = begin3 + gws_z_min * step3;
                        double ws_z_ppm_max = begin3 + gws_z_max * step3;
                        
                        if (ws_x_ppm_min > ws_x_ppm_max) std::swap(ws_x_ppm_min, ws_x_ppm_max);
                        if (ws_y_ppm_min > ws_y_ppm_max) std::swap(ws_y_ppm_min, ws_y_ppm_max);
                        if (ws_z_ppm_min > ws_z_ppm_max) std::swap(ws_z_ppm_min, ws_z_ppm_max);
                        
                        std::cout << "      Peak " << (p+1) << " Watershed range (indices): "
                                  << "X[" << ws_x_min << ", " << ws_x_max << "] "
                                  << "Y[" << ws_y_min << ", " << ws_y_max << "] "
                                  << "Z[" << ws_z_min << ", " << ws_z_max << "]" << std::endl;
                        std::cout << "      Peak " << (p+1) << " Watershed range (ppm): "
                                  << "X[" << std::setprecision(3) << ws_x_ppm_min << ", " << ws_x_ppm_max << "] "
                                  << "Y[" << ws_y_ppm_min << ", " << ws_y_ppm_max << "] "
                                  << "Z[" << ws_z_ppm_min << ", " << ws_z_ppm_max << "]" << std::endl;
                    }
                }
                
                // === Compute intersection of watershed and FWHH regions ===
                std::set<int> final_region;
                if (kUseWatershedFittingRegion && !watershed_region.empty()) {
                    // Intersection: indices in both watershed and FWHH
                    for (int idx : watershed_region) {
                        if (fwhh_region.count(idx) > 0) {
                            final_region.insert(idx);
                        }
                    }
                } else {
                    // If watershed disabled, use FWHH region alone
                    final_region = fwhh_region;
                }

                // Build fitting data from final region
                if (!final_region.empty()) {
                    fit_peak_data.reserve(final_region.size());
                    fit_x_coords.reserve(final_region.size());
                    fit_y_coords.reserve(final_region.size());
                    fit_z_coords.reserve(final_region.size());
                    
                    for (int idx : final_region) {
                        fit_peak_data.push_back(peak_data[idx]);
                        fit_x_coords.push_back(x_coords[idx]);
                        fit_y_coords.push_back(y_coords[idx]);
                        fit_z_coords.push_back(z_coords[idx]);
                    }
                    
                    if (kDebugDumpEmRoundPeaks && kUseWatershedFittingRegion) {
                        std::cout << "      Peak " << (p+1) << " region: watershed=" << watershed_region.size() 
                                  << " voxels, FWHH=" << fwhh_region.size() 
                                  << " voxels, intersection=" << final_region.size() << " voxels" << std::endl;
                    } else if (kDebugDumpEmRoundPeaks) {
                        std::cout << "      Peak " << (p+1) << " FWHH region: " << final_region.size() 
                                  << " voxels" << std::endl;
                    }
                }
                
                // Fallback to full separated profile if fitting region is too small or empty.
                if (fit_peak_data.size() < 11) {
                    if (kDebugDumpEmRoundPeaks) {
                        std::cout << "      Peak " << (p+1) << " fitting region too small (" 
                                  << fit_peak_data.size() << "), using full separated profile" << std::endl;
                    }
                    fit_peak_data = peak_data;
                    fit_x_coords = x_coords;
                    fit_y_coords = y_coords;
                    fit_z_coords = z_coords;
                }

                auto x_minmax = std::minmax_element(fit_x_coords.begin(), fit_x_coords.end());
                auto y_minmax = std::minmax_element(fit_y_coords.begin(), fit_y_coords.end());
                auto z_minmax = std::minmax_element(fit_z_coords.begin(), fit_z_coords.end());
                if (kDebugDumpEmRoundPeaks && !fit_x_coords.empty()) {
                    int gx_min = *x_minmax.first + x0;
                    int gx_max = *x_minmax.second + x0;
                    int gy_min = *y_minmax.first + y0;
                    int gy_max = *y_minmax.second + y0;
                    int gz_min = *z_minmax.first + z0;
                    int gz_max = *z_minmax.second + z0;

                    double x_ppm_from = begin1 + gx_min * step1;
                    double x_ppm_to = begin1 + gx_max * step1;
                    double y_ppm_from = begin2 + gy_min * step2;
                    double y_ppm_to = begin2 + gy_max * step2;
                    double z_ppm_from = begin3 + gz_min * step3;
                    double z_ppm_to = begin3 + gz_max * step3;

                    if (x_ppm_from > x_ppm_to) std::swap(x_ppm_from, x_ppm_to);
                    if (y_ppm_from > y_ppm_to) std::swap(y_ppm_from, y_ppm_to);
                    if (z_ppm_from > z_ppm_to) std::swap(z_ppm_from, z_ppm_to);

                    std::cout << "      Peak " << (p+1) << " fitting bounds ppm: "
                              << "X[" << std::fixed << std::setprecision(3) << x_ppm_from << ", " << x_ppm_to << "] "
                              << "Y[" << y_ppm_from << ", " << y_ppm_to << "] "
                              << "Z[" << z_ppm_from << ", " << z_ppm_to << "]"
                              << std::endl;
                    std::cout<<" peak "<<(p+1)<<" x_minmax.second = "<<*x_minmax.second<<" x_minmax.first = "<<*x_minmax.first
                             <<" y_minmax.second = "<<*y_minmax.second<<" y_minmax.first = "<<*y_minmax.first
                             <<" z_minmax.second = "<<*z_minmax.second<<" z_minmax.first = "<<*z_minmax.first
                             << std::endl;
                }

                /**
                 * If fitting region is too small, the optimization can easily go wrong and diverge,
                 * We will requir at least 3 points along any direction to perform fitting, 
                 * otherwise we will skip fitting and remove the peak directly, since it's unlikely to be a real peak.
                 */
                if(*x_minmax.second - *x_minmax.first < 2 || *y_minmax.second - *y_minmax.first < 2 || *z_minmax.second - *z_minmax.first < 2) {

                    if (verbose_level >= 1) {
                        std::cout << "      Peak " << (p+1) << " fitting region has less than 3 points in at least one direction, "
                                  << "skipping fitting and marking for removal" << std::endl;
                    }

                    if (p < peak_initial_locations.size()) {
                        removed_peak_initial_locations.push_back(peak_initial_locations[p]);
                    }
                    peak_removed[p] = true;
                    continue; // Skip fitting. We wil go directly to the next peak, and this peak will be removed in the compact step after this EM round.
                }

                fit_one_peak(fit_peak_data,fit_x_coords,fit_y_coords,fit_z_coords, nx, ny, nz,pre_fit_peaks[p]);

                /**
                 * Remove peaks that have very low amplitude (below 6x noise level) 
                 */
                if (pre_fit_peaks[p].amplitude < scale1 * noise_level) {
                    if (p < peak_initial_locations.size()) {
                        removed_peak_initial_locations.push_back(peak_initial_locations[p]);
                    }
                    peak_removed[p] = true;
                    if (kDebugDumpEmRoundPeaks) {
                        std::cout << "      Peak " << (p + 1) << " marked removed: amplitude "
                                  << std::scientific << pre_fit_peaks[p].amplitude
                                  << " < " << scale1 * noise_level << std::endl;
                    }
                }

                /**
                 * Remove peaks that moved outside of the partition box after fitting,
                 * which can happen when fitting region is small or noisy and optimization goes wrong.
                 */
                else if (pre_fit_peaks[p].x_center < 0.0f || pre_fit_peaks[p].x_center > (float)(nx - 1) ||
                    pre_fit_peaks[p].y_center < 0.0f || pre_fit_peaks[p].y_center > (float)(ny - 1) ||
                    pre_fit_peaks[p].z_center < 0.0f || pre_fit_peaks[p].z_center > (float)(nz - 1)) {
                    if (p < peak_initial_locations.size()) {
                        removed_peak_initial_locations.push_back(peak_initial_locations[p]);
                    }
                    peak_removed[p] = true;
                    if (kDebugDumpEmRoundPeaks) {
                        std::cout << "      Peak " << (p + 1)
                                  << " marked removed: fitted center moved out of partition box" << std::endl;
                    }
                }

                /**
                 * Of moved out its own fitting region (intersection of FWHH region and watershed region), which can also indicate fitting failure due to small/noisy region.
                 */
                else if(pre_fit_peaks[p].x_center < *x_minmax.first || pre_fit_peaks[p].x_center > *x_minmax.second ||
                        pre_fit_peaks[p].y_center < *y_minmax.first || pre_fit_peaks[p].y_center > *y_minmax.second ||
                        pre_fit_peaks[p].z_center < *z_minmax.first || pre_fit_peaks[p].z_center > *z_minmax.second) {
                    if (p < peak_initial_locations.size()) {
                        removed_peak_initial_locations.push_back(peak_initial_locations[p]);
                    }
                    peak_removed[p] = true;
                    if (kDebugDumpEmRoundPeaks) {
                        std::cout << "      Peak " << (p + 1)
                                  << " marked removed: fitted center moved out of its fitting region" << std::endl;
                    }
                }
                
                // Calculate global coordinates and PPM values
                double ppm_f3 = begin1 + (pre_fit_peaks[p].x_center + x0) * step1;
                double ppm_f2 = begin2 + (pre_fit_peaks[p].y_center + y0) * step2;
                double ppm_f1 = begin3 + (pre_fit_peaks[p].z_center + z0) * step3;
                
                if (verbose_level >= 2) {
                    std::cout << "    Peak " << (p+1) << " refined at idx(" << pre_fit_peaks[p].x_center + x0 << ", " 
                            << pre_fit_peaks[p].y_center + y0 << ", " << pre_fit_peaks[p].z_center + z0 << ") ppm(" << std::fixed << std::setprecision(3)
                            << ppm_f3 << ", " << ppm_f2 << ", " << ppm_f1 << ") amp=" 
                            << std::scientific << pre_fit_peaks[p].amplitude << std::endl;
                }
            } //End of loop for fitting each peak

            if (kDebugDumpEmRoundPeaks &&
                kDebugDumpEmRoundInterval > 0 &&
                (total_em_rounds % kDebugDumpEmRoundInterval) == 0) {
                std::vector<Peak3D> active_peaks_for_dump = build_active_peaks();
                dump_prefit_em_round_peaks(active_peaks_for_dump, total_em_rounds, coor,
                                           begin1, step1, begin2, step2, begin3, step3,
                                           x0, y0, z0);
            }

            size_t removed_this_round = compact_removed_peaks();
            if (removed_this_round > 0) {
                total_removed_peaks += removed_this_round;
                if (verbose_level >= 2) {
                    std::cout << "  Removed " << removed_this_round
                            << " peaks after EM Round " << total_em_rounds
                            << "; active peaks=" << pre_fit_peaks.size() << std::endl;
                }
            }
            
        }  // End of EM rounds loop
        
        // --- NEW PEAK PICKING LOGIC ---
        // Calculate total model to find residual
        std::vector<float> current_model;
        std::vector<Peak3D> active_peaks_for_model = build_active_peaks();
        calculate_total_3d_model_sparse_with_support(current_model, active_peaks_for_model, x_coords, y_coords, z_coords, FwhhFacctorSeperation);
        
        std::vector<float> residual_grid(nx * ny * nz, 0.0f);
        std::vector<float> observed_grid(nx * ny * nz, 0.0f);
        
        for (size_t idx = 0; idx < partition_data.size(); idx++) {
            int x = x_coords[idx];
            int y = y_coords[idx];
            int z = z_coords[idx];
            int grid_idx = x * ny * nz + y * nz + z;
            residual_grid[grid_idx] = partition_data[idx] - current_model[idx];
            observed_grid[grid_idx] = partition_data[idx];
        }
        
        // === TRUE WATERSHED ALGORITHM ===
        // Find local maxima as watershed markers
        std::vector<std::tuple<int, int, int, float>> peak_candidates;
        
        for (int x = 1; x < nx - 1; x++) {
            for (int y = 1; y < ny - 1; y++) {
                for (int z = 1; z < nz - 1; z++) {
                    int grid_idx = x * ny * nz + y * nz + z;
                    float val = residual_grid[grid_idx];
                    
                    if (val <= scale1 * noise_level) continue;
                    if (observed_grid[grid_idx] <= scale1 * noise_level) continue;
                    if (is_near_removed_initial(x, y, z)) continue;
                    
                    // Check 6 direct neighbors for local maximum
                    if (val > residual_grid[(x+1) * ny * nz + y * nz + z] &&
                        val > residual_grid[(x-1) * ny * nz + y * nz + z] &&
                        val > residual_grid[x * ny * nz + (y+1) * nz + z] &&
                        val > residual_grid[x * ny * nz + (y-1) * nz + z] &&
                        val > residual_grid[x * ny * nz + y * nz + (z+1)] &&
                        val > residual_grid[x * ny * nz + y * nz + (z-1)]) {
                        peak_candidates.push_back(std::make_tuple(x, y, z, val));
                    }
                }
            }
        }
        
        // Perform watershed segmentation using priority queue (steepest descent)
        // Each voxel flows to the nearest local maximum
        std::vector<int> watershed_labels(nx * ny * nz, -1);  // -1 = unassigned, >= 0 = peak index
        std::priority_queue<std::tuple<float, int, int, int>> pq;  // (residual_val_desc, x, y, z)
        
        // Initialize with peak candidates
        for (size_t p_idx = 0; p_idx < peak_candidates.size(); p_idx++) {
            int x = std::get<0>(peak_candidates[p_idx]);
            int y = std::get<1>(peak_candidates[p_idx]);
            int z = std::get<2>(peak_candidates[p_idx]);
            int gidx = x * ny * nz + y * nz + z;
            watershed_labels[gidx] = (int)p_idx;
            pq.push(std::make_tuple(residual_grid[gidx], x, y, z));
        }
        
        // Watershed flooding: grow each peak's region via steepest descent
        int dx_arr[] = {1, -1, 0, 0, 0, 0};
        int dy_arr[] = {0, 0, 1, -1, 0, 0};
        int dz_arr[] = {0, 0, 0, 0, 1, -1};
        
        while (!pq.empty()) {
            float res_val = std::get<0>(pq.top());
            int x = std::get<1>(pq.top());
            int y = std::get<2>(pq.top());
            int z = std::get<3>(pq.top());
            pq.pop();
            int gidx = x * ny * nz + y * nz + z;
            int label = watershed_labels[gidx];
            
            // Check 6 neighbors
            for (int neighbor = 0; neighbor < 6; neighbor++) {
                int nx_idx = x + dx_arr[neighbor];
                int ny_idx = y + dy_arr[neighbor];
                int nz_idx = z + dz_arr[neighbor];
                
                if (nx_idx >= 0 && nx_idx < nx && ny_idx >= 0 && ny_idx < ny && nz_idx >= 0 && nz_idx < nz) {
                    int ngidx = nx_idx * ny * nz + ny_idx * nz + nz_idx;
                    
                    // If unassigned and above noise threshold, assign to same peak
                    if (watershed_labels[ngidx] == -1 && residual_grid[ngidx] > scale2 * noise_level) {
                        watershed_labels[ngidx] = label;
                        pq.push(std::make_tuple(residual_grid[ngidx], nx_idx, ny_idx, nz_idx));
                    }
                }
            }
        }
        
        // Compute statistics for each watershed region
        std::vector<float> region_max_observed(peak_candidates.size(), 0.0f);
        std::vector<int> region_voxel_count(peak_candidates.size(), 0);
        
        for (int grid_idx = 0; grid_idx < nx * ny * nz; grid_idx++) {
            int label = watershed_labels[grid_idx];
            if (label >= 0 && label < (int)peak_candidates.size()) {
                region_max_observed[label] = std::max(region_max_observed[label], observed_grid[grid_idx]);
                region_voxel_count[label]++;
            }
        }
        
        /**
         * Now we have watershed regions for each peak candidate, and we can apply criteria to select new peaks to add:
         * - Calculate a normalized score for each peak candidate, e.g., norm_val = peak_residual / max_observed_in_region, to find peaks that are strong relative to local noise.
         * - Require a minimum region size (e.g., at least 3 voxels) to avoid spurious peaks from single-voxel noise spikes.
         */
        std::vector<float> fwhh_values;
        fwhh_values.reserve(pre_fit_peaks.size() * 3);
        for (size_t i = 0; i < pre_fit_peaks.size(); ++i) {
            if (i < peak_removed.size() && peak_removed[i]) continue;
            fwhh_values.push_back(pre_fit_peaks[i].x_width);
            fwhh_values.push_back(pre_fit_peaks[i].y_width);
            fwhh_values.push_back(pre_fit_peaks[i].z_width);
        }

        float median_fwhh = 0.8f;
        if (!fwhh_values.empty()) {
            size_t mid = fwhh_values.size() / 2;
            std::nth_element(fwhh_values.begin(), fwhh_values.begin() + mid, fwhh_values.end());
            median_fwhh = fwhh_values[mid];
        }
        float min_new_peak_dist = 3.0f * median_fwhh;
        float min_new_peak_dist_sq = min_new_peak_dist * min_new_peak_dist;

        struct NewPeakCandidate {
            float norm_val;
            float peak_residual;
            int x;
            int y;
            int z;
            int sparse_idx;
        };

        std::vector<NewPeakCandidate> new_candidates;
        new_candidates.reserve(peak_candidates.size());

        for (size_t p_idx = 0; p_idx < peak_candidates.size(); p_idx++) {
            float peak_residual = std::get<3>(peak_candidates[p_idx]);
            float max_obs_in_region = region_max_observed[p_idx];
            int voxel_count = region_voxel_count[p_idx];

            // Require minimum region size to avoid spurious peaks
            if (voxel_count < 3) continue;

            float norm_val = peak_residual / std::max(max_obs_in_region, 1e-10f);
            if (norm_val <= nor_res_cutoff) continue;

            int x = std::get<0>(peak_candidates[p_idx]);
            int y = std::get<1>(peak_candidates[p_idx]);
            int z = std::get<2>(peak_candidates[p_idx]);
            int gidx = cube_flat_index(prefit_cube, x, y, z);
            if (gidx < 0 || gidx >= (int)prefit_cube.grid_to_sparse_idx.size()) continue;
            int sparse_idx = prefit_cube.grid_to_sparse_idx[(size_t)gidx];
            if (sparse_idx < 0) continue;

            new_candidates.push_back({norm_val, peak_residual, x, y, z, sparse_idx});
        }

        std::sort(new_candidates.begin(), new_candidates.end(),
                  [](const NewPeakCandidate& a, const NewPeakCandidate& b) {
                      return a.norm_val > b.norm_val;
                  });

        int added_this_round = 0;
        std::vector<std::tuple<float, float, float>> new_peak_centers;
        new_peak_centers.reserve(new_candidates.size());

        for (const auto& cand : new_candidates) {
            if (active_peak_count() + (size_t)added_this_round >= (size_t)max_peaks) {
                break;
            }

            float cx = (float)(x_coords[cand.sparse_idx]);
            float cy = (float)(y_coords[cand.sparse_idx]);
            float cz = (float)(z_coords[cand.sparse_idx]);

            bool too_close = false;
            for (const auto& center : new_peak_centers) {
                float dx = cx - std::get<0>(center);
                float dy = cy - std::get<1>(center);
                float dz = cz - std::get<2>(center);
                if (dx * dx + dy * dy + dz * dz < min_new_peak_dist_sq) {
                    too_close = true;
                    break;
                }
            }
            if (too_close) continue;

            Peak3D new_peak;
            new_peak.amplitude = cand.peak_residual;
            new_peak.x_center = cx;
            new_peak.y_center = cy;
            new_peak.z_center = cz;
            new_peak.x_width = 2.0f;
            new_peak.y_width = 2.0f;
            new_peak.z_width = 2.0f;
            new_peak.x_lfrac = 0.5f;
            new_peak.y_lfrac = 0.1f;
            new_peak.z_lfrac = 0.1f;
            pre_fit_peaks.push_back(new_peak);
            peak_initial_locations.emplace_back((int)std::round(new_peak.x_center),
                                                (int)std::round(new_peak.y_center),
                                                (int)std::round(new_peak.z_center));
            peak_removed.push_back(false);
            new_peak_centers.emplace_back(cx, cy, cz);
            added_this_round++;

            // Calculate global coordinates and PPM values
            double ppm_f3 = begin1 + new_peak.x_center * step1;
            double ppm_f2 = begin2 + new_peak.y_center * step2;
            double ppm_f1 = begin3 + new_peak.z_center * step3;

            if (verbose_level >= 2) {
                std::cout << "  Added new peak at idx(" << new_peak.x_center << ", " << new_peak.y_center << ", " << new_peak.z_center
                        << ") ppm(" << std::fixed << std::setprecision(3) << ppm_f3 << ", " << ppm_f2 << ", " << ppm_f1
                        << ") norm_val=" << std::fixed << std::setprecision(3) << cand.norm_val
                        << ", raw_val=" << std::scientific << cand.peak_residual << std::endl;
            }
        }

        if (added_this_round > 0) {
            peak_added = true;
        }
        if (kDebugDumpEmRoundPeaks && verbose_level >= 2) {
            std::cout << "    Added " << added_this_round << " new peak(s) this round" << std::endl;
        }
        // --- END NEW PEAK PICKING LOGIC ---
    }

    size_t removed_at_end = compact_removed_peaks();
    total_removed_peaks += removed_at_end;
    
    if (verbose_level >= 1) {
        std::cout << "  Completed prefitting of " << pre_fit_peaks.size() << " peaks with " << total_em_rounds
                << " total EM rounds (removed " << total_removed_peaks << " peaks)" << std::endl;
    }
    
    return pre_fit_peaks;
}

/**
 * Fit a single peak using Levenberg-Marquardt optimization with the provided fitting data and initial peak parameters.
 * Returns true if fitting succeeded, false otherwise.
 * @param fit_peak_data: vector of intensity values for fitting
 * @param fit_x_coords: vector of x coordinates corresponding to fit_peak_data
 * @param fit_y_coords: vector of y coordinates corresponding to fit_peak_data
 * @param fit_z_coords: vector of z coordinates corresponding to fit_peak_data
 * @param nx: size of the partition in x dimension
 * @param ny: size of the partition in y dimension
 * @param nz: size of the partition in z dimension
 * @param peak_to_fit: Peak3D object containing initial parameters, which will be updated with fitted parameters
 * @return: true if fitting succeeded, false otherwise
 */
bool spectrum_fit_3d::fit_one_peak(const std::vector<float> &fit_peak_data,
                                   const std::vector<int> &fit_x_coords,
                                   const std::vector<int> &fit_y_coords,
                                   const std::vector<int> &fit_z_coords,
                                   const int nx, const int ny, const int nz,
                                   Peak3D &peak_to_fit)
{
    /** Run fitting */
    SinglePeak3DCostFunction cost_func;
    cost_func.data = &fit_peak_data;
    cost_func.x_coords = &fit_x_coords;
    cost_func.y_coords = &fit_y_coords;
    cost_func.z_coords = &fit_z_coords;
    cost_func.n_params = kPeakParamCount;
    cost_func.alpha = 1.0f;  // symmetric loss for positive and negative residuals
    cost_func.lambda = 0.0f; // No regularization for EM pre-fitting
    cost_func.noise_floor = noise_level;

    std::vector<double> param_values(kPeakParamCount);
    std::vector<double *> par(kPeakParamCount);

    // Initialize parameters from current peak
    param_values[0] = (double)peak_to_fit.amplitude;
    param_values[1] = (double)peak_to_fit.x_center;
    param_values[2] = (double)peak_to_fit.y_center;
    param_values[3] = (double)peak_to_fit.z_center;
    param_values[4] = (double)peak_to_fit.x_width;
    param_values[5] = (double)peak_to_fit.y_width;
    param_values[6] = (double)peak_to_fit.z_width;
    param_values[7] = (double)peak_to_fit.x_lfrac;
    param_values[8] = (double)peak_to_fit.y_lfrac;
    param_values[9] = (double)peak_to_fit.z_lfrac;

    for (int i = 0; i < kPeakParamCount; i++)
    {
        par[i] = &param_values[i];
    }

    std::vector<double> weights(fit_peak_data.size() + 1, 1.0); // +1 residual for regularization
    levmarq optimizer;
    std::array<double, kPeakParamCount> lower_bounds = {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0};
    std::array<double, kPeakParamCount> upper_bounds = {
        std::numeric_limits<double>::max(),
        (double)(nx - 1), (double)(ny - 1), (double)(nz - 1),
        10.0, 10.0, 10.0,
        1.0, 1.0, 1.0};
    optimizer.set_bounds(std::vector<double>(lower_bounds.begin(), lower_bounds.end()), std::vector<double>(upper_bounds.begin(), upper_bounds.end()));
    int opt_result = optimizer.solve(kPeakParamCount, par.data(), (int)fit_peak_data.size() + 1, weights.data(), cost_func);

    // Update peak with optimized parameters
    peak_to_fit.amplitude = (float)param_values[0];
    peak_to_fit.x_center = (float)param_values[1];
    peak_to_fit.y_center = (float)param_values[2];
    peak_to_fit.z_center = (float)param_values[3];
    peak_to_fit.x_width = std::min(10.0f, std::max(0.5f, (float)param_values[4]));
    peak_to_fit.y_width = std::min(10.0f, std::max(0.5f, (float)param_values[5]));
    peak_to_fit.z_width = std::min(10.0f, std::max(0.5f, (float)param_values[6]));
    peak_to_fit.x_lfrac = std::min(1.0f, std::max(0.0f, (float)param_values[7]));
    peak_to_fit.y_lfrac = std::min(1.0f, std::max(0.0f, (float)param_values[8]));
    peak_to_fit.z_lfrac = std::min(1.0f, std::max(0.0f, (float)param_values[9]));

    return true;
}

#ifdef WEBASSEMBLY
using namespace emscripten;

EMSCRIPTEN_BINDINGS(dp_module_spectrum_fit_3d)
{
    class_<spectrum_fit_3d, base<fid_3d>>("spectrum_fit_3d")
        .constructor()
        .function("set_noise_scales", &spectrum_fit_3d::set_noise_scales)
        .function("set_new_peak_cutoff", &spectrum_fit_3d::set_new_peak_cutoff)
        .function("set_noise_level_for_nus", &spectrum_fit_3d::set_noise_level_for_nus)
        .function("peak_picking", &spectrum_fit_3d::peak_picking)
        .function("partition_signal_regions", &spectrum_fit_3d::partition_signal_regions)
        .function("iterative_fit_all_partitions", &spectrum_fit_3d::iterative_fit_all_partitions)
        .function("get_fitted_peaks_string", &spectrum_fit_3d::get_fitted_peaks_string)
        .function("set_verbose", &spectrum_fit_3d::set_verbose);
}
#endif
