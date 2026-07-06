#ifndef SPECTRUM_FIT_3D_H
#define SPECTRUM_FIT_3D_H

#include <cstdio>
#include <string>
#include <vector>
#include <tuple>
#include <set>

#include "fid_3d.h"
#include "spectrum_fit_3d_cost.h"

// ============================================================================
// spectrum_fit_3d class definition
// ============================================================================

class spectrum_fit_3d : public fid_3d
{
protected:
    int verbose_level;
    float partition_threshold;
    float scale1;
    float scale2;
    std::vector<int> partition_labels;
    float nor_res_cutoff;
    // Array of detected partitions. Each entry stores sparse voxel indices (storage-order),
    // bounding box and intensity statistics for a single connected region.
    std::vector<fid_3d::partition> partitionArray;
    // Mapping between pre-sort (signal regions file) indices and sorted indices.
    std::vector<int> partition_old_to_new;
    std::vector<int> partition_new_to_old;
    // Mapping from region label ID (1-based) to partitionArray index (pre-sort).
    std::vector<int> partition_label_to_index;
    // Cached list of picked peaks read from the peak-picking output (3d_peaks.txt)
    std::vector<std::tuple<int,int,int,float>> picked_peaks_all;
    // List of all fitted peaks after iterative fitting of all partitions. Each entry is a Peak3D struct containing fitted parameters and metadata.
    std::vector<Peak3D> fitted_peaks_all;

    std::vector<Peak3D> prefit_partition_peaks(
        const std::vector<float>& partition_data,
        const std::vector<int>& x_coords,
        const std::vector<int>& y_coords,
        const std::vector<int>& z_coords,
        const fid_3d::partition& coor,
        double begin1, double step1,
        double begin2, double step2,
        double begin3, double step3,
        int x0, int y0, int z0);

    void watershed_fitting_region(
        const std::vector<float>& peak_data,
        const std::vector<int>& x_coords,
        const std::vector<int>& y_coords,
        const std::vector<int>& z_coords,
        float peak_x_center,
        float peak_y_center,
        float peak_z_center,
        std::set<int>& watershed_region);

    
    bool fit_one_peak(const std::vector<float>& fit_peak_data,
                                 const std::vector<int>& fit_x_coords,
                                 const std::vector<int>& fit_y_coords,
                                 const std::vector<int>& fit_z_coords,
                                 const int nx, const int ny, const int nz,
                                 Peak3D &peak_to_fit);

public:
    spectrum_fit_3d();
    ~spectrum_fit_3d();

    // Load picked peaks from a file into `picked_peaks_all` (returns true on success)
    bool load_picked_peaks_file(const std::string& fname = "3d_peaks.txt");

    bool peak_picking(std::string outfname);
    bool partition_signal_regions(std::string outfname = "");

    bool dump_partition_voxels(int partition_idx, const std::string& outfname) const;

    bool iterative_fit_one_partition(int partition_idx);
    bool iterative_fit_all_partitions(int top_n);

    /**
     * @brief Save the current fitted peaks list to a formatted string.
     * The format matches the output previously written to prefit_fp.
     */
    std::string get_fitted_peaks_string() const;

    inline void set_noise_scales(double noise, double s1, double s2) {
        if (noise > 1e-20) {
            noise_level = static_cast<float>(noise);
        }
        if (s1 > 0.0) {
            scale1 = static_cast<float>(s1);
        }
        if (s2 > 0.0) {
            scale2 = static_cast<float>(s2);
        }
    }
    inline float get_noise_level() const { return noise_level; }
    inline float get_scale1() const { return scale1; }
    inline float get_scale2() const { return scale2; }
    inline void set_verbose(int level) { verbose_level = level; }
    inline void set_new_peak_cutoff(float value) {
        if (value > 0.0f) {
            nor_res_cutoff = value;
        }
    }
};

#endif
