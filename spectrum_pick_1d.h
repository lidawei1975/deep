

#include "spectrum_io_1d.h"
#include "dnn_picker.h"

#ifndef SPEC_FWHH_1D_H
#define SPEC_FWHH_1D_H

struct spectrum_1d_peaks
{
    std::vector<double> a;          //peak intensity 
    std::vector<double> x;          //peak coordinates
    std::vector<double> sigmax;     //Gaussian peak shape parameter. IMPORTANT: in Gaussian fit, this is actually 2*sigma*sigma
    std::vector<double> gammax;     //Lorentzian peak shape parameter
    std::vector<double> confidence; //confidence level of peak
    std::vector<double> intens;     //peak intensity
};

class spectrum_pick_1d: public spectrum_io_1d
{
private:
   
    int mod_selection;
    class peak1d p1;
    bool b_negative; //if true, we will pick negative peaks besides positive peaks

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
    bool work(std::string outfname);
    bool work2(bool b_negative=false);
    bool get_peak_pos(std::vector<int> &);
    bool print_peaks(std::string outfname);
    bool get_peaks(struct spectrum_1d_peaks & );

};

#endif