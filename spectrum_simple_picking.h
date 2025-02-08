#include <vector>
#include <set>
#include <deque>
#include <string>

#ifndef SPECTRUM_IO_HEAD
#define SPECTRUM_IO_HEAD
#include "fid_2d.h"
#endif

class spectrum_simple_pick : public fid_2d
{
private:
protected:
    std::vector<int> p_type;
    int peak_diag;

    bool b_negative;

    bool sub_pixel_0();
    bool sub_pixel();
    bool normal_peak_picking();
    bool shoulder_peak_picking();
    bool get_mean_width_from_picking();
    bool estimate_sigma(float xc, float yc, float a1, float a2, float a3, float a4, float a5, float aa, float &sx, float &sy);
    bool laplacing_of_gaussian_convolution(std::vector< std::vector<double> > &s,double width);

public:
    spectrum_simple_pick();
    ~spectrum_simple_pick();

    bool simple_peak_picking(bool b_negative = false, bool b_shoulder = false);
    bool print_peaks_picking(std::string);
};
