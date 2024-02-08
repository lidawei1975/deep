#include <vector>
#include <set>
#include <deque>
#include <string>


#include "spectrum_fwhh.h"
#include "cubic_spline.h"

class spectrum_pick : public spectrum_fwhh
{
private:

protected:
   
    //1 for protein (wide peak 8-20, optimal 12) 2 for meta (narrow peak: 4-12, optimal 8)
    int model_selection;

    /**
     * when we adjust ppp, we need to interpolate the spectrum to a new ppp
     * this is the step size of interpolation, in pixel. That is, old step is 1.0 pixel.
     * We save them because we need to restore original peak positions (in pixel) after interpolation and picking
    */
    double interpolation_step_direct,interpolation_step_indirect;
    
    
public:

    spectrum_pick();
    ~spectrum_pick();  

    bool simple_peak_picking(bool b_negative=false);
    bool ann_peak_picking(int flag=0,int flag_t1_noise=0, bool b_negative=false);
    bool print_peaks_picking(std::string);
    bool voigt_convolution(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> &kernel,int &i0,int &i1, int &j0, int &j1) const;

    /**
     * Adjust ppp of spectrum to a given value, using cubic spline interpolation
    */
    bool adjust_ppp_of_spectrum(const double ppp); 
    

    inline void set_model_selection(int n)
    {
        model_selection=n;
    }
};

