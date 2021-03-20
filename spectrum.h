#include <vector>
#include <set>
#include <deque>
#include <string>



#ifndef SPECTRUM_TYPE
#define SPECTRUM_TYPE
enum spectrum_type 
{
    null_spectrum,
    hsqc_spectrum,
    tocsy_spectrum
};
#endif

#include "spectrum_io.h"

class spectrum_picking : public spectrum_io
{
private:

protected:
   
    std::string infname;
    int model_selection; //1 for protein (wide peak 4-22) 2 for meta (narrow peak 2-13)
    double user_scale,user_scale2; //minimal peak intesntiy in picking and fitting
    double median_width_x,median_width_y; //peak width median from either picking or fitting
    
    std::vector<int> peak_index; //used when read in peak list to keep track of all peaks.
    std::vector<double> p1,p2,p_intensity;  //grid peak position, peak amp 
    std::vector<double> p1_ppm,p2_ppm; //ppm value of peak pos, obtained from p1 and p2
    std::vector<int> p_type;  //normal peak (1) or shoulder peak (2) flag.      
    std::vector<double> p_confidencex,p_confidencey; //confidence level of peaks
    std::vector<int> p_noise; //true or noise peak                           
    std::vector<std::string> user_comments; //user comment of about a peak!
    std::vector<double> sigmax,sigmay;  //for Gaussian and Voigt fitting, est in picking too
    std::vector<double> gammax,gammay; //for voigt fittting and ann picking, set to 0 in picking!

    bool get_ppm_from_point();
    bool zero_negative();
    
public:

    spectrum_picking();
    ~spectrum_picking();  

    bool ann_peak_picking();
    bool output_picking(std::string);

    


    inline void set_scale(double x,double y)
    {
        user_scale=x;
        user_scale2=y;
    } 
    
    inline void get_median_width(double *x, double *y)
    {
        *x=median_width_x;
        *y=median_width_y;
    }

    inline void set_model_selection(int n)
    {
        model_selection=n;
    }
};

