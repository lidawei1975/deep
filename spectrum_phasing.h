#include <array>

#include "spectrum_fwhh.h" //spectrum_io_1d is included in spectrum_fwhh_1d.h
#include "phase_dnn.h"

#ifndef SPECTRUM_PHASING_H
#define SPECTRUM_PHASING_H

class spectrum_phasing : public spectrum_io // spectrum_fwhh_1d is a derived class of spectrum_io_1d
{
private:
    /**
     * The following 4 variables are used to store the final phase correction values.
    */
    float final_p0_direct, final_p1_direct, final_p0_indirect, final_p1_indirect;


protected:
    std::vector<int> p1;            // peak position
    std::vector<float> p_intensity; // peak intensity
    double max_intensity;           // maximum of p_intensity (used to normalize p_intensity)
    int water_pos;                  // water peak position

    bool phase_correction_worker(int n_dim1,int n_dim2, double p0, double p1, bool b_second,
                                 float *pectrum_real, float * spectrum_imag,
                                float *spectrum_real_out, float * spectrum_image_out) const;

    float entropy_based_p0_correction( int ndim1,int ndim2,int nstep, 
                                       float step_size, float p0_center,
                                       float * spectrum_real, float * spectrum_imag,
                                       std::vector<float> &p0s, std::vector<float> &entropies) const;

    bool entropy_based_p0_p1_correction( int ndim1,int ndim2,
                                         int nstep, float step_size, float p0_center,
                                         int nstep2, float step_size2, float p1_center,
                                         float * spectrum_real, float* spectrum_imag,
                                         std::vector<float> &p0s, std::vector<float> &p1s, std::vector<float> &entropies) const;

    /**
     * User segment by segment variance method to estimate the noise level of the spectrum.
     * size of spectrum is ydim * xdim. xdim (row) is major dimension.
    */
    float estimate_noise_level(int ydim, int xdim, float * spectrum) const;
   
public:
    spectrum_phasing();
    ~spectrum_phasing();

    bool auto_phase_correction_v2();
    bool set_user_phase_correction(double p0_direct, double p1_direct, double p0_indirect, double p1_indirect);
    bool save_phase_correction_result(std::string fname) const;
};

#endif