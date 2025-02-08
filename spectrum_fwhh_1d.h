

#include "fid_1d.h"
#include "fwhh_dnn.h"

#ifndef SPEC_FWHH_1D_H
#define SPEC_FWHH_1D_H

class spectrum_fwhh_1d: public fid_1d
{
private:
    std::vector<int> fwhh_1d_pos;
    std::vector<float> fwhh_1d_wids;
   
public:

    spectrum_fwhh_1d();
    ~spectrum_fwhh_1d();
    float get_median_peak_width();
    void print_result(std::string fname);
};

#endif