#include <vector>
#include <set>
#include <deque>
#include <string>


#ifndef SPECTRUM_IO_HEAD
  #define SPECTRUM_IO_HEAD
  #include "spectrum_io.h"
#endif

class spectrum_pick : public spectrum_io
{
private:

protected:
   
    int model_selection; //1 for protein (wide peak 4-22) 2 for meta (narrow peak 2-13)
    
    
public:

    spectrum_pick();
    ~spectrum_pick();  

    bool simple_peak_picking();
    bool ann_peak_picking(int flag=0,int expand=0,int flag_t1_noise=0);
    bool linear_regression();
    bool print_peaks_picking(std::string);
    bool clear_memory();
    bool voigt_convolution(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> &kernel,int &i0,int &i1, int &j0, int &j1);

    

    inline void set_model_selection(int n)
    {
        model_selection=n;
    }
};

