
#include <vector>
#include <iostream>


#include "kiss_fft.h"
#include "spectrum_io_1d.h"
#include "spectrum_pick_1d.h"
#include "spectrum_fit_1d.h"

#ifndef SPECTURM_SIMULATION_H
#define SPECTURM_SIMULATION_H
class spectrum_simulation
{
public:
    spectrum_simulation();
    ~spectrum_simulation();

    bool run(int ndata, int ndata_frq, const std::vector<double> &ppm, const std::vector<float> &fid_real, const std::vector<float> &fid_imag, spectrum_1d_peaks &peak_list,double &step2,int n);

};
#endif
