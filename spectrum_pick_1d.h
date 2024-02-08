

#include "spectrum_fwhh_1d.h"
#include "dnn_picker.h"
#include "cubic_spline.h"

#ifndef SPEC_PICK_1D_H
#define SPEC_PICK_1D_H



class spectrum_pick_1d: public spectrum_fwhh_1d
{
private:
   
    int mod_selection;
    class peak1d p1;
    bool b_negative; //if true, we will pick negative peaks besides positive peaks

    /**
     * when we adjust ppp, we need to interpolate the spectrum to a new ppp
     * this is the step size of interpolation, in pixel. That is, old step is 1.0 pixel.
    */
    double interpolation_step;

    std::vector<int> final_segment_begin,final_segment_stop; //updated in peak_partition_step2 only.

    bool substract_baseline();
    bool peak_partition_1d();
    bool peak_partition_step2();
    bool run_ann(bool b_neg=false);

public:

    //picked peaks
    std::vector<double> a;          //peak intensity 
    std::vector<double> x;          //peak coordinates
    std::vector<double> sigmax;     //Gaussian peak shape parameter. IMPORTANT: in Gaussian fit, this is actually 2*sigma*sigma
    std::vector<double> gammax;     //Lorentzian peak shape parameter
    std::vector<double> confidence; //confidence level of peak

    spectrum_pick_1d();
    ~spectrum_pick_1d();
    bool init_mod(int);
    bool spectrum_pick_1d_work(bool b_negative=false);
    bool get_peak_pos(std::vector<int> &);
    bool print_peaks(std::string outfname);
    bool get_peaks(struct spectrum_1d_peaks & );

    bool interpolate_spectrum(const double interpolation_step); //interpolate spectrum to a new ppp
    bool adjust_ppp_of_spectrum(const double ppp); //adjust ppp of spectrum to a given value, using cubic spline interpolation

};

#endif