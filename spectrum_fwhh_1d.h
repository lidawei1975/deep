

#include "spectrum_io_1d.h"
#include "fwhh_dnn.h"

#ifndef SPEC_FWHH_1D_H
#define SPEC_FWHH_1D_H

class spectrum_fwhh_1d: public spectrum_io_1d
{
private:
    std::vector<int> pos;
    std::vector<float> wids;
   
public:

    spectrum_fwhh_1d();
    ~spectrum_fwhh_1d();
    float get_median_peak_width();
    void print_result(std::string fname);
};

#endif