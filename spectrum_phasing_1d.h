#include <array>

#include "spectrum_fwhh_1d.h" //spectrum_io_1d is included in spectrum_fwhh_1d.h

#ifndef SPECTRUM_PHASING_1D_H
#define SPECTRUM_PHASING_1D_H

class spectrum_phasing_1d : public spectrum_fwhh_1d // spectrum_fwhh_1d is a derived class of spectrum_io_1d
{
private:
    std::array<double, 2> phase_correction; // phase_correction[0] is total left phase, phase_correction[1] is total right phase

    float fwhh_1d_median; // median of peak widthes

    int max_loop;
    int max_peak;
    int max_dist;
    bool b_end; //use both ends of the spectrum to assess phase correction
    bool b_smooth_baseline; //at last, adjust phase correction to make baseline smooth (minimal entropy)

    /**
     * @brief Define validation_dnn here
     * The order or layers are same as order defination here
     */
    conv1d validation_dnn_c1, validation_dnn_c2, validation_dnn_c3;
    pool1d validation_pool1;
    conv1d validation_dnn_c4;
    pool1d validation_pool2;
    dense validation_dnn_d1, validation_dnn_d2, validation_dnn_d3;

    /**
     * @brief Define phasing_dnn here
     * The order of layers are same as order defination here
     */
    conv1d phase_dnn_c1, phase_dnn_c2, phase_dnn_c3;
    pool1d phase_pool1;
    conv1d phase_dnn_c4;
    pool1d phase_pool2;
    dense phase_dnn_d1, phase_dnn_d2, phase_dnn_d3;

protected:
    std::vector<int> p1;            // peak position
    std::vector<float> p_intensity; // peak intensity
    double max_intensity;           // maximum of p_intensity (used to normalize p_intensity)
    int water_pos;                  // water peak position

    bool normalize_spectrum(std::vector<float> &y) const;         // normalize spectrum to be between 0 and 1. This is required by DNN
    bool is_valid_from_all_peaks(int p_inten,int start, int end, int direction) const; // return false if phase cannot be determined, using peaks information only
    bool is_valid(std::vector<float> spectrum_part) const;        // return false if phase cannot be determined
    bool has_phase_error(std::vector<float> spectrum_part) const; // return true if phase is incorrect

    bool simple_peak_picking();
    double calculate_entropy_of_spe(const std::vector<float> &s) const;
    bool calculate_phased_spectrum(const std::vector<float> &spe_real, const std::vector<float> &spe_imag, const double phase_left, const double phase_right, std::vector<float> &spe_real_phased, std::vector<float> &spe_image_phased) const;
    bool entropy_minimization(double &, double &) const;
    bool entropy_minimization_grid(double &,double &) const;
    bool assess_phase_at_peaks(std::vector<int> &left_cross, std::vector<int> &right_cross, int &npeak_tested, int stride) const;
    bool assess_two_end_phase_error(const int,const int,std::vector<float> &left_stds,std::vector<float> &right_stds) const;
    bool gd_optimization_from_cross(const int, const float,const int, const float, const std::vector<int> left_cross, const std::vector<int> right_cross, const int npeak_tested, double &, double &) const;
    
    double test_consistence_with_peak_based_method(const double left_end,const double right_end,const std::vector<int> left_cross, const std::vector<int> right_cross,const int npeak_tested) const;

public:
    spectrum_phasing_1d();
    ~spectrum_phasing_1d();
    bool phase_spectrum(const double phase_left, const double phase_right);
    bool set_up_parameters(const int max_loop, const int max_peak, const int max_dist,const bool b_end, const bool b_smooth_baseline=false);
    bool auto_phase_correction();                                                    // main working function
    std::array<double, 2> get_phase_correction() const { return phase_correction; }; // phase_correction[0] is total left phase, phase_correction[1] is total right phase
};

#endif