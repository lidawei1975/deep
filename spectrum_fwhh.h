

#include "spectrum_io.h"
#include "fwhh_dnn.h"

#ifndef SPEC_FWHH_1D
#define SPEC_FWHH_1D

class spectrum_fwhh: public spectrum_io
{
private:
    std::vector<int> fwhh_pos_direct, fwhh_pos_indirect;
    std::vector<float> fwhh_wids_direct, fwhh_wids_indirect;
   
public:

    spectrum_fwhh();
    ~spectrum_fwhh();
    bool get_median_peak_width(float &ppp_direct, float &ppp_indirect);
    void print_result(std::string fname);
};

#endif