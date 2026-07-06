
#ifndef FID_3D_H
#define FID_3D_H

#include <vector>
#include <string>
#include <map>
#include <string>
#include <cstddef>
#include <cstdint>
#include <array>

#include "fid_1d.h"
#include "fid_2d.h" // Including fid_2d.h might be useful for some shared utilities or if we want to follow similar patterns

class fid_3d : public fid_base
{
public:
    // Representation of a detected signal partition (connected region) in storage-order
    // Coordinates are stored in storage-order indexing: x = inner (fastest),
    // y = intermediate, z = outer (slowest). All voxel lists and bounds refer
    // to storage-order indices. Physical-axis (ppm) mapping is applied later
    // when producing user-visible outputs.
    struct partition {
        // Sparse lists of voxel coordinates that belong to this partition
        std::vector<int> x; // inner-axis indices
        std::vector<int> y; // intermediate-axis indices
        std::vector<int> z; // outer-axis indices

        // Partition statistics
        double total_intensity; // sum of voxel intensities for this partition
        float max_intensity;    // maximum single-voxel intensity in the partition
        int min_x, max_x, min_y, max_y, min_z, max_z; // bounding box in storage-order

        partition() : total_intensity(0.0), max_intensity(-1e6f),
                      min_x(0), max_x(0), min_y(0), max_y(0), min_z(0), max_z(0) {}
    };

    double begin1, step1, stop1; // Direct dimension
    double begin2, step2, stop2; // 2nd dimension
    double begin3, step3, stop3; // 3rd dimension

protected:
    /**
     * Flags
     */
    bool b_read_bruker_acqus_and_fid; 
    bool b_read_nmrpipe_fid;

    std::string aqseq; // "321" etc.

    bool b_negative; 

    double receiver_gain;

    /**
     * Dimensions and acquisition parameters
     */
    int fnmode2; // Encoding for 2nd dimension
    int fnmode3; // Encoding for 3rd dimension
    double grpdly;

    FID_DATA_TYPE data_type;
    FID_DATA_COMPLEXITY data_complexity;

    // Dictionaries
    std::map<std::string, std::string> udict_acqus_direct;   // F3
    std::map<std::string, std::string> udict_acqus_indirect1; // F2
    std::map<std::string, std::string> udict_acqus_indirect2; // F1

    // Phase corrections from pulse program
    double indirect1_p0, indirect1_p1; // F2
    double indirect2_p0, indirect2_p1; // F1

    std::string nucleus;
    std::string nucleus_indirect1;
    std::string nucleus_indirect2;

    // F3 (Direct)
    double spectral_width;
    double observed_frequency;
    double carrier_frequency;
    double origin_frequency;

    // F2 (Indirect 1)
    double spectral_width_indirect1;
    double observed_frequency_indirect1;
    double carrier_frequency_indirect1;
    double reference_frequency_indirect1;
    double origin_frequency_indirect1;

    // F1 (Indirect 2)
    double spectral_width_indirect2;
    double observed_frequency_indirect2;
    double carrier_frequency_indirect2;
    double reference_frequency_indirect2;
    double origin_frequency_indirect2;
    
    // Processing status flags
    bool b_frq;      // FFT done on F3
    bool b_frq_indirect1; // FFT done on F2
    bool b_frq_indirect2; // FFT done on F1

    bool b_imaginary; // Keep imaginary data?
    bool b_imaginary_indirect;
    bool b_imaginary_indirect2;

    // User controls: whether to keep imaginary channels after each axis processing step.
    // These are processing directives and are independent of b_imaginary* state flags.
    bool keep_imag_direct_step;
    bool keep_imag_indirect1_step;
    bool keep_imag_indirect2_step;

    // Optional sign inversion for imaginary traces in indirect dimensions.
    bool b_inverse_indirect1;
    bool b_inverse_indirect2;

    // Optional polynomial baseline correction order per physical axis (F2, F1, F3).
    // -1: disabled, 0..4: polynomial order.
    std::array<int, 3> time_domain_polynorminal_order;
    std::array<int, 3> frq_domain_polynorminal_order;
    // Final frequency-domain polynomial baseline orders applied after full FFT (trace-by-trace on spectrum_rrr)
    // -1: disabled, 0..4: polynomial order
    std::array<int, 3> final_frq_polynorminal_order;

    // Segment size used by noise estimation and signal-free region detection.
    int segment_size;
    
    // Direct-dimension extraction controls (same semantics as fid_2d)
    bool extraction_use_ppm;
    double extraction_from;
    double extraction_to;
    double extraction_from_ppm;
    double extraction_to_ppm;
    int extraction_from_pos;
    int extraction_to_pos;

    // Data sizes
    // Bruker raw sizes
    int ndata_bruker;
    int ndata_bruker_indirect1;
    int ndata_bruker_indirect2;

    // Logical sizes (points)
    int ndata;
    int ndata_indirect1;
    int ndata_indirect2;

    // Center points used for nmrPipe header metadata.
    int n_center;
    int n_center_indirect1;
    int n_center_indirect2;

    // Power of 2 sizes
    int ndata_power_of_2;
    int ndata_power_of_2_indirect1;
    int ndata_power_of_2_indirect2;

    // Zero filling
    int zf;
    int zf_indirect1;
    int zf_indirect2;

    // Bruker in-memory loading state for WebAssembly path.
    bool use_nus_input;
    bool nus_input_is_inflated;
    std::size_t expected_bruker_raw_points;
    std::vector<std::array<int, 2>> bruker_nus_pairs;

    // NUS simulation mode: when enabled, reads fully sampled data then zeros out traces not in NUS list
    bool simulate_nus_mode;
    std::vector<std::array<int, 2>> simulate_nus_pairs;

    // Current axis order in the working buffer: positions 0,1,2 map to physical axes.
    std::array<int, 3> current_axis_order;

    // Frequency domain sizes after ZF
    int ndata_frq;
    int ndata_frq_indirect1;
    int ndata_frq_indirect2;

    // Apodization codes
    int apodization_code;
    int apodization_code_indirect1;
    int apodization_code_indirect2;

    // Time domain data (all valid data from Bruker read)
    // Flattened: [Indirect2][Indirect1][Direct]
    std::vector<float> fid_data_float;

    /**
     * Frequency Domain Data
     * For 3D complex data, we have 2^3 = 8 components.
     * Dimension order is always: Indirect2 (F3), Indirect1 (F1), Direct (F2: as in nmrPipe FDF2???)
     * So direct dimension is always the most inner dimension, and indirect2 is the outer dimension.
     */
    // Real-Real-Real
    std::vector<float> spectrum_rrr; 
    // Real-Real-Imag
    std::vector<float> spectrum_rri;
    // Real-Imag-Real
    std::vector<float> spectrum_rir;
    // Real-Imag-Imag
    std::vector<float> spectrum_rii;
    // Imag-Real-Real
    std::vector<float> spectrum_irr;
    // Imag-Real-Imag
    std::vector<float> spectrum_iri;
    // Imag-Imag-Real
    std::vector<float> spectrum_iir;
    // Imag-Imag-Imag
    std::vector<float> spectrum_iii;

    float spectrum_rrr_max; // Max value in the real-real-real component, used for normalization and noise estimation.
    
    // Pointer alias for the main real spectrum (spectrum_rrr)
    float *spect; 

    // User phase corrections
    double user_p0_direct, user_p1_direct;
    double user_p0_indirect1, user_p1_indirect1;
    double user_p0_indirect2, user_p1_indirect2;

    // NMRPipe Header
    std::map<std::string, std::string> nmrpipe_dict_string;
    std::map<std::string, float> nmrpipe_dict_float;
    std::vector<float> nmrpipe_header_data;

    // Apodization objects
    class apodization *apodization_direct;
    class apodization *apodization_indirect1;
    class apodization *apodization_indirect2;

    float noise_level;
    bool noise_level_for_nus;

    // Shared serializer used by write_ft3(file) and write_ft3_to_buffer(memory).
    bool serialize_ft3_bytes(std::vector<unsigned char> &out_bytes);
    std::vector<unsigned char> ft3_serialized_buffer;


public:
    fid_3d();
    ~fid_3d();



    bool set_up_apodization(apodization *a1, apodization *a2, apodization *a3);

    bool set_up_apodization_from_string(const std::string &apodization_string_direct,
                                        const std::string &apodization_string_indirect1,
                                        const std::string &apodization_string_indirect2);

    bool run_zf(int n_zf_direct, int n_zf_indirect1, int n_zf_indirect2);

    /**
     * @brief read_bruker_folder: read acqus/acqu2s/acqu3s and a single fid/ser file from a Bruker folder.
     */
    bool read_bruker_folder(std::string folder_name);

    /**
     * @brief read_bruker_files: read Bruker acqus/acqu2s/acqu3s and one raw fid/ser file.
     */
    bool read_bruker_files(const std::string &pulse_program_name,
                           const std::string &acqus_file2_name,
                           const std::string &acqus_file3_name,
                           const std::string &acqus_file_name,
                           const std::string &fid_data_file_name,
                           const std::string &nuslist_contents = "",
                           bool simulate_nus = false,
                           bool nus_ser_inflated = false);

    // Two-step in-memory Bruker loader for WebAssembly:
    // 1) parse text files, 2) provide binary payload from memory.
    bool read_bruker_files_as_strings(const std::string &contents_pulse,
                                      const std::string &contents_acqus,
                                      const std::string &contents_acqus2,
                                      const std::string &contents_acqus3,
                                      const std::string &contents_nuslist = "",
                                      bool simulate_nus = false,
                                      bool nus_ser_inflated = false);
    bool read_bruker_fid_data(const std::vector<float> &fid_data_float_in);
    bool read_bruker_fid_data_bytes(const std::vector<unsigned char> &fid_data_bytes);
    bool read_ft3_from_buffer(const std::vector<unsigned char> &ft3_bytes);
    bool postprocess_loaded_ft3();



    /**
     * Helper to set variables from dictionary
     */
    bool set_varibles_from_nmrpipe_dictionary();

    /**
     * Print F2/F1/F3 nuclei, ppm ranges, and sizes after input is loaded.
     */
    void print_axis_summary() const;

    /**
     * @brief Estimate noise level of the final `spectrum_rrr` buffer using
     * 16x16x16 segment RMSD and returning the median RMS across segments.
     * If the data dimensions are smaller than 16 in any axis, falls back to
     * `fid_3d_aux::estimate_noise_mad(spectrum_rrr)`.
     */
    float estimate_final_rrr_segment_noise() const;

    // ========================================================================
    // Private FID processing pipeline helpers
    // ========================================================================
    
    /**
     * @brief Read all time-domain .fid planes from indexed filename range
     * @param path_name: directory path
     * @param prefix: filename prefix
     * @param file_ext: file extension
     * @param start_idx: starting index
     * @param end_idx: ending index
     * @param digit_width: zero-padding width
     * @return: true if successfully read all planes
     */
    bool read_time_domain_planes(const std::string &path_name,
                                 const std::string &prefix,
                                 const std::string &file_ext,
                                 int start_idx,
                                 int end_idx,
                                 int digit_width);

    /**
     * @brief Apply optional time-domain baseline correction (water suppression)
     * along the direct dimension before direct FFT.
     */
    bool apply_time_domain_polynorminal_baseline_axis(int axis_id);
    bool apply_frq_domain_polynorminal_baseline(float *trace_real,
                                                float *trace_imag,
                                                int n_freq,
                                                int axis_id,
                                                double threshold = -1.0) const;

    /**
     * @brief Apply final frequency-domain polynomial baseline on the fully processed 3D spectrum.
     * This operates trace-by-trace on the `spectrum_rrr` buffer along each axis where
     * `final_frq_polynorminal_order` is >= 0. Returns true on success.
     */
    bool apply_final_frq_polynomial_baseline();

    /**
     * @brief Print current storage axis ordering (position -> physical axis).
     * Example output: "pos0->F2 pos1->F1 pos2->F3"
     */
    void print_current_axis_order() const;

    /**
     * @brief Apply FFT to direct dimension (innermost axis)
     * Processes all 8 complex components with apodization and phase correction
     */
    bool process_direct_dimension();
    bool apply_direct_extraction_after_direct_fft();
    bool apply_nus_simulation_mask(const std::vector<std::array<int, 2>> &nus_pairs_list);
    
    bool extract_region(double from, double to);
    bool extract_region_ppm(double from_ppm, double to_ppm);

    /**
     * @brief Transpose/gather for indirect1 axis preparation
     * Rearranges data from direct-fast layout to indirect1-fast layout for FFT
     */
    bool process_indirect1_dimension();

    /**
     * @brief Transpose/gather for indirect2 axis preparation
     * Rearranges data from indirect1-fast layout to indirect2-fast layout for FFT
     */
    bool process_indirect2_dimension();

    /**
     * @brief Clean up unused complex components based on axis flags
     */
    void cleanup_unused_components();

    /**
     * @brief Transpose any two axes in the processed 3D buffers.
        * Axis ids refer to current buffer positions.
     */
    bool transpose_dimensions(int axis_a, int axis_b);

    /**
     * @brief Apply a 3-axis permutation to the processed 3D buffers.
        * Axis ids refer to current buffer positions.
     */
    bool transpose_dimensions(int axis0, int axis1, int axis2);

        int current_axis_length(int position) const;
        std::array<int, 3> current_position_lengths() const;

    /**
     * Load user provided phase correction values from a file.
     * 6 values:
     * p0 (direct), p1 (direct),
     * p0 (indirect1), p1 (indirect1),
     * p0 (indirect2), p1 (indirect2)
     */
    bool read_phase_correction(std::string fname);
    bool read_phase_correction_from_string(const std::string &phase_correction_text);

    inline void set_time_domain_polynorminal_order(int f2_order, int f1_order, int f3_order)
    {
        std::cout<< "Set time-domain polynomial baseline order: F2 = " << f2_order
                  << ", F1 = " << f1_order
                  << ", F3 = " << f3_order << std::endl;
        time_domain_polynorminal_order = {f2_order, f1_order, f3_order};
    }

    inline void set_frq_domain_polynorminal_order(int f2_order, int f1_order, int f3_order)
    {
        std::cout<< "Set frequency-domain polynomial baseline order: F2 = " << f2_order
                  << ", F1 = " << f1_order
                  << ", F3 = " << f3_order << std::endl;
        frq_domain_polynorminal_order = {f2_order, f1_order, f3_order};
    }

    inline void set_final_frq_polynorminal_order(int f2_order, int f1_order, int f3_order)
    {
        std::cout<< "Set final frequency-domain polynomial baseline order: F2 = " << f2_order
                  << ", F1 = " << f1_order
                  << ", F3 = " << f3_order << std::endl;
        final_frq_polynorminal_order = {f2_order, f1_order, f3_order};
    }

    inline void set_segment_size(int value)
    {
        segment_size = value;
    }

    // Combined inverse switch by axis (F2, F1, F3). F2 entry is accepted for
    // interface consistency and ignored because direct-axis inversion is unsupported.
    inline void set_inverse(bool f2_inverse, bool f1_inverse, bool f3_inverse)
    {
        (void)f2_inverse;
        b_inverse_indirect1 = f1_inverse;
        b_inverse_indirect2 = f3_inverse;
        std::cout<< "Set inverse: F2 (direct) ignored, F1 (indirect1) = " << (f1_inverse ? "y" : "n")
                  << ", F3 (indirect2) = " << (f3_inverse ? "y" : "n") << std::endl;
    }


    // Combined delete-image switch by axis (F2, F1, F3).
    // true means delete imaginary channel after that axis processing step.
    inline void set_delete_image(bool f2_delete, bool f1_delete, bool f3_delete)
    {
        keep_imag_direct_step = !f2_delete;
        keep_imag_indirect1_step = !f1_delete;
        keep_imag_indirect2_step = !f3_delete;
    }

    inline void set_noise_level_for_nus(bool value)
    {
        noise_level_for_nus = value;
    }

    float calculate_noise_level_for_nus()
    {
        if(spectrum_rrr.empty())
        {
            std::cout << "Cannot calculate noise level for NUS simulation: spectrum_rrr is empty." << std::endl;
            return 0.0f;
        }
        spectrum_rrr_max = 0.0f;
        for (const auto& val : spectrum_rrr)
        {
            float abs_val = std::abs(val);
            if (abs_val > spectrum_rrr_max)
            {
                spectrum_rrr_max = abs_val;
            }
        }
        std::cout<< "Calculated spectrum_rrr_max for NUS noise estimation: " << spectrum_rrr_max << std::endl;
        return spectrum_rrr_max * 0.001f; // Set noise level to 1% of max value
    }


    /**
     * @brief Read the input file range and load data into memory.
     */
    bool read(const std::string &first_file, const std::string &last_file);

    /**
     * @brief Read a single stacked .ft3 file and load data into memory.
     */
    bool read(const std::string &input_file);

    /**
     * @brief Run all in-memory FFT processing steps after read().
     */
    bool full_process();

    /**
     * @brief Run only direct-dimension processing, then reorder storage to y,z,x
     * (indirect1, indirect2, direct) for intermediate export.
     */
    bool direct_only_process();

    /**
     * @brief Continue processing from direct-only intermediate stage to final stage.
     */
    bool indirect_only_process();

    /**
     * @brief Write processed spectrum outputs.
     */
    bool write(const std::string &out_ft3 = "",
               const std::string &out_ft2_first = "");

    // ========================================================================
    // Output/serialization methods
    // ========================================================================

    /**
        * @brief create nmrPipe dictionary from current 3D state
     */
        bool create_nmrpipe_dictionary(std::map<std::string, std::string> &dict_string,
                                std::map<std::string, float> &dict_float,
                                bool b_init) const;

        /**
        * @brief prepare nmrPipe header array in nmrpipe_header_data
        */
        bool prepare_header_for_nmrpipe();

    /**
     * @brief Write processed spectrum as single stacked .ft3 file
     * @param out_fname: output filename
     * @return: true if successfully written
     */
    bool write_ft3(const std::string &out_fname);

    /**
     * @brief Serialize processed spectrum as single stacked .ft3 byte buffer
     * @param out_bytes: output byte buffer (header + stacked payload)
     * @return: true if successfully serialized
     */
    bool write_ft3_to_buffer(std::vector<unsigned char> &out_bytes);

    /**
     * @brief Write processed spectrum as series of indexed .ft2 plane files
     * @param first_plane_fname: first output plane filename (e.g., proc001.ft2)
     * @return: true if successfully written all planes
     */
    bool write_ft2_series(const std::string &first_plane_fname);

    inline float* get_spect_data() { return spect; }

    inline void get_dim(int *n1, int *n2, int *n3)
    {
        *n1 = get_ndata_direct();
        *n2 = get_ndata_indirect1();
        *n3 = get_ndata_indirect2();
    }

    inline int get_ndata_direct() const
    {
        return (b_frq && ndata_frq > 0) ? ndata_frq : ndata;
    }

    inline int get_ndata_indirect1() const
    {
        return (b_frq_indirect1 && ndata_frq_indirect1 > 0) ? ndata_frq_indirect1 : ndata_indirect1;
    }

    inline int get_ndata_indirect2() const
    {
        return (b_frq_indirect2 && ndata_frq_indirect2 > 0) ? ndata_frq_indirect2 : ndata_indirect2;
    }

#ifdef WEBASSEMBLY
    uintptr_t get_nmrpipe_header_data();
    uintptr_t get_data_of_rrr();

    // Zero-copy optimizations
    bool read_bruker_fid_data_bytes_raw(uintptr_t ptr, size_t size);
    bool read_ft3_from_buffer_raw(uintptr_t ptr, size_t size);
    bool serialize_ft3_to_internal_buffer();
    uintptr_t get_ft3_buffer_ptr() const;
    size_t get_ft3_buffer_size() const;
#endif
};

#endif
