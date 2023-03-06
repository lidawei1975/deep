#include "spectrum_io_3d.h"

extern "C"  
{
    double voigt(double x, double sigma, double gamma);
    void re_im_w_of_z( double x, double y ,double *r, double *i); //re_im_w_of_z
};

struct ldw_triple
{
    int first,second,third;

    inline ldw_triple(int x, int y, int z)
    {
        first = x;
        second = y;
        third = z;
    };

    void inline operator = (const ldw_triple &D )
    { 
        first = D.first;
        second = D.second;
        third = D.third;
    }
};

#ifndef FIT_TYPE
#define FIT_TYPE
enum fit_type
{
    gaussian_type,
    voigt_type,
    null_type
};
#endif

//for 3D Gaussian fitting, analytical derivative
class mycostfunction_gaussian_3d : public ceres::CostFunction
{

private:
  int nx,ny,nz;
  double *data; //z-> spectra data

public:
  ~mycostfunction_gaussian_3d();
  mycostfunction_gaussian_3d(int,int,int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};

//for 3D voigt fitting, analytical derivative
class voigt_fit_3d : public ceres::CostFunction
{

private:
    int n_datapoint;
    int nx, ny, nz;
    double *data;

    void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
    void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
    ~voigt_fit_3d();
    voigt_fit_3d(int, int, int, double *);
    bool Evaluate(double const *const *, double *, double **) const;
    inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
    inline void set_n_residuals(int n) { set_num_residuals(n); };
};

class gaussian_fit_3d
{
private:

    std::vector<double> spe;
    enum fit_type peak_shape;
    int rmax;                          
    double noise_level;      
    double user_scale2;    
    std::array<int,3> dims,start_pos;
    std::array<double,3> median_widths;
    std::vector<std::array<int,6>> valid_fit_region;

    ceres::Solver::Options options;

    bool one_fit_gaussian(double *data, std::array<int,3> dims, std::array<double,3> &coors,std::array<double,3> &sigmas, double &inten, double &e);
    bool one_fit_voigt(double *data, std::array<int,3> dims, std::array<double,3> &coors, std::array<double,3> &sigmas,std::array<double,3> &gammas,double &inten, double &e, int loop);
    bool run_single_peak();
    bool run_multiple_peaks();
    bool limit_fitting_region_of_each_peak();
    bool find_highest_neighbor(int xx,int yy,int zz, int &mm,int &nn, int &kk);

public:
    std::vector<std::array<double,3> > x,sigma,gamma; //peak para
    std::vector<std::array<int,3> > x_int; //coor at grid
    std::vector<int> to_remove; //lable to_be_remvoed peaks identifed in fitting process
    std::vector<double> intensity;   //peak intensity  a[spe index][peak index]
    std::vector<double> err;                //fitting residual (RMSD)
    std::vector<double> numerical_sum;
    std::vector<int> cannot_move;
    std::vector<double> original_ratio;


    gaussian_fit_3d();
    ~gaussian_fit_3d();

    bool init_optional(std::array<int,3>);
    bool init(fit_type,int,std::array<int,3>,std::vector<double>,std::vector<std::array<double,3> >,std::vector<std::array<double,3> >,std::vector<double>,std::vector<int>);

    bool set_peak_paras(std::array<double,3>,double,double);
    bool run();
};


class spectrum_fit_3d : public spectrum_io_3d
{
protected:
    enum fit_type peak_shape;
    int maxround;
    std::array<double,3> median_widths,w;

    //fitted peaks
    //keep one to one corresponse to picked (or loaded) peaks
    std::vector< std::array<double,3> > fitted_peaks_pos;
    std::vector< std::array<double,3> > fitted_sigma,fitted_gamma;
    std::vector<double> fitted_intensity;
    std::vector<double> numerical_sum;
    std::vector<int> peak_updated; 

    /**
     * @brief We treate positive and negative peaks separately. 
     * In other words, we first fit all positive peaks, then all negative peaks.
     * We don't fit positive and negative peaks together because it is inherently unstable.
     */

    //These 3 varibles are shared between peak_partition and work!!
    std::vector< std::vector<int> > begin_array,end_array; 
    std::vector< std::deque<ldw_triple> > clusters2; //peak cluster in peak partition
    std::vector<std::vector< std::vector<int> > > peaks_in_this_line;

    //same as above, but for negative peaks.
    std::vector< std::vector<int> > begin_array_neg,end_array_neg; 
    std::vector< std::deque<ldw_triple> > clusters2_neg; //peak cluster in peak partition
    std::vector<std::vector< std::vector<int> > > peaks_in_this_line_neg;

    

    bool get_median_width_of_peaks();
    bool peak_partition();
    bool peak_partition_neg();
    

public:
    spectrum_fit_3d();
    ~spectrum_fit_3d();
    bool read_for_fitting(std::string fname1, std::string fname2);
    bool work();
    bool init_fitting_parameter(int,int);
    bool print_fitted_peaks(std::string fname);
};

