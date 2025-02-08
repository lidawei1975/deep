
#include <vector>
#include <iostream>


#include "kiss_fft.h"
#include "fid_1d.h"
#include "spectrum_pick_1d.h"
#include "spectrum_fit_1d.h"

#ifndef SPECTURM_SIMULATION_H
#define SPECTURM_SIMULATION_H

namespace spectrum_simulation_helper{
    size_t split(const std::string &txt, std::vector<std::string> &strs, char ch);
}

class spectrum_simulation
{
public:
    spectrum_simulation();
    ~spectrum_simulation();

    bool simualte_fid(int ndata, const std::vector<double> spin_ppm, const std::vector<double> spin_height,double b0, double delta_t, double r2, const std::string apodization_method, std::vector<float> &fid_real, std::vector<float> &fid_imag);

    bool run(int ndata, int ndata_frq, const std::vector<double> &ppm, const std::vector<float> &fid_real, const std::vector<float> &fid_imag, spectrum_1d_peaks &peak_list,double &step2,int n);

    bool run_fft(int ndata, int ndata_frq, std::vector<float> &spectrum_real,std::vector<float> &spectrum_imag, const std::vector<float> &fid_real, const std::vector<float> &fid_imag, int n);

    bool run_peaks(int ndata_frq, const std::vector<float> &ppm, const std::vector<float> spectrum, spectrum_1d_peaks &peak_list,double &step2);

};
#endif
