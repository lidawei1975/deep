//fucntions from libcerf
extern "C"  
{
    double voigt(double x, double sigma, double gamma);
    void re_im_w_of_z( double x, double y ,double *r, double *i); //re_im_w_of_z
};

#include "spectrum_io.h"


#ifndef FIT_TYPE
    #define FIT_TYPE
    enum fit_type
    {
        gaussian_type,
        voigt_type,
        null_type
    };
#endif

#ifndef LDW_PARAS
    #define LDW_PARAS
    #define SMALL 1e-10
    #define PI 3.14159265358979323846
    #define M_SQRT_PI 1.772453850905516
    #define M_SQRT_2PI 2.506628274631000
    #define M_1_SQRT_PI 0.564189583547756
#endif


//for 2D voigt fitting, analytical derivative
class mycostfunction_voigt : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  int xdim,ydim;
  double *z; //x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt();
  mycostfunction_voigt(int,int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};

class mycostfunction_voigt_v2 : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  double *x,*y,*z; //x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt_v2();
  mycostfunction_voigt_v2(int,double *,double *, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};

//for 2D voigt fitting, intensity only, analytical derivative
/*
class mycostfunction_voigt_a : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  int xdim,ydim;
  double *z; //x,y -> coor, z-> spectra data
  double x0,y0,sigmax,sigmay,gammax,gammay;

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt_a();
  mycostfunction_voigt_a(int,int, double *,double,double,double,double,double,double);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};
*/

//for 2D Gaussian fitting, analytical derivative
class mycostfunction_gaussian : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  int xdim,ydim;
  double *z; //x,y -> coor, z-> spectra data

public:
  ~mycostfunction_gaussian();
  mycostfunction_gaussian(int,int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};

class mycostfunction_gaussian_v2 : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  double *x,*y,*z; //x,y -> coor, z-> spectra data

  double gaussian_funtion(double x,double y,double sx,double sy) const;


public:
  ~mycostfunction_gaussian_v2();
  mycostfunction_gaussian_v2(int,double *,double *, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};



class gaussian_fit
{
  private:
    int my_index;
    int rmax;                                             //max round in fitting
    double noise_level;                                   //noise level of spectrum.
    std::vector<int> x_int, y_int;                        // peak coordinate at grid
    std::vector<std::vector<double>> x_old, y_old; //used for check convergence
    double too_near_cutoff; //peaks near to another bigger peak will be removed in fititng, if d<sqrt(wx*wy)*too_near_cutoff
    double median_width_x, median_width_y; //median values of whole spectrum, not median of this fitting region
    double minimal_height; //minimal peak height

    std::vector<double> min_x,max_x,min_y,max_y; //region of fitting area of individual peak
    std::vector<std::vector<double>> peaks_fitting_data;  //data used to fit each peak. 
    
    int nround;                             //round number when exit
    double gaussian_fit_wx, gaussian_fit_wy;                          //maximal half peak width , used to define fitting area of each peak!!
    enum fit_type type;                     //gaussian or voigt
    std::vector<std::array<int,4>> valid_fit_region;

    int i0,i1,j0,j1; //used in convolution and mixed gaussian model

    ceres::Solver::Options options;

    bool one_fit_gaussian(int,int,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay, double *e);
    bool one_fit_gaussian_v2(std::vector<double> *xx,std::vector<double> *yy,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay, double *e);
    bool one_fit_gaussian_intensity_only(int,int,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay, double *e);
    bool multiple_fit_gaussian(int,int, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double *e);
    bool multiple_fit_gaussian_v2(std::vector<double> &xx,std::vector<double> &yy, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double *e);
    

    bool one_fit_voigt(int,int, std::vector<double> *, double &, double &, double &, double &, double &, double &, double &, double *, int n=0);
    bool one_fit_voigt_core(int,int, std::vector<double> *, double &, double &, double &, double &, double &, double &, double &, double *);
    
    bool one_fit_voigt_v2(std::vector<double> *xx,std::vector<double> *yy,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay,double &gammax,double &gammay,double *e, int n);
    bool one_fit_voigt_intensity_only(int,int,std::vector<double> *, double &, double, double, double, double, double, double, double *);
    bool multiple_fit_voigt(int,int, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay,double &gammax, double &gammay, double *e, int n=0);
    bool multiple_fit_voigt_v2(std::vector<double> &xx,std::vector<double> &yy, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay,double &gammax, double &gammay, double *e, int n=0);
    bool multiple_fit_voigt_core(int,int, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay,double &gammax, double &gammay, double *e);
 


    bool voigt_convolution_region(double x, double y, double sigmax, double sigmay, double gammax, double gammay,double scale=1.5);
    bool gaussain_convolution_region(double x, double y, double sigmax, double sigmay,double scale=1.5);

    bool gaussain_convolution(double a, double x, double y, double sigmax, double sigmay, std::vector<double> *kernel,double scale=1.5);
    bool gaussain_convolution_within_region(int ndx,double a, double x, double y, double sigmax, double sigmay, std::vector<double> *kernel,double scale=1.0);
    
    bool voigt_convolution(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> *kernel,double scale=1.5);
    bool voigt_convolution_within_region(int ndx,double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> *kernel,double scale=1.0);
    
    //bool voigt_convolution_2(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> *kernel,int xdim2,int ydim2);

    bool find_highest_neighbor(int xx,int yy,int &m,int &n);
    bool limit_fitting_region_of_each_peak();
    bool limit_fitting_region_of_one_peak();
    bool run_multi_peaks(int rmax);
    bool run_single_peak();
    bool multi_spectra_run_multi_peaks(int rmax);
    bool multi_spectra_run_single_peak();

  public:
    //will be read directly in fit_gather step of spectrum_fit.cpp
    //updated in fitting at each iteration
    std::vector<int> to_remove; //lable to_be_remvoed peaks identifed in fitting process
    std::vector< std::vector<double> > a;   //peak intensity  a[spe index][peak index]
    std::vector<double> x, y;               //peak coordinates
    std::vector<double> sigmax, sigmay;     //peak width
    std::vector<double> gammax, gammay;     //Lorentzian peak width
    std::vector<double> err;                //fitting residual (RMSD)
    std::vector<int> original_ndx;          //for debug only.
    std::vector<int> cannot_move;
    std::vector<double> original_ratio;
    

    std::vector<std::vector<double>> analytical_spectra;    //analytical spectrum for each peak
    std::vector<double> peaks_total;                        //reconstructed 2D matrix from fitting: sum of analytical_spectra
    
    //these 2 will be re-calcualted from above varibles
    // std::vector<double> x_ppm, y_ppm;               //peak coor in ppm 
    std::vector< std::vector<double> > num_sum;     // numerical integral of each peak num_sum[spe index][peak index]
    
    //other varibles need direct access
    int xstart, ystart;
    int xdim, ydim;
    std::vector< std::vector<double> > surface;      //2D spectrum matrix, column by column order. outlayer: list of spectra

    
    gaussian_fit();
    ~gaussian_fit();

    bool init(int, int, int, int, std::vector<std::vector<double>>, std::vector<double>, std::vector<double>, std::vector< std::vector<double>>, std::vector<double>, std::vector<double>, std::vector<int>, std::vector<int>);
    bool run();
    bool assess_size();
    int get_my_index();
    int get_nround();
    void set_everything(fit_type t, int r, int index);
    void set_peak_paras(double x, double y, double noise, double height, double near);
    
};


class spectrum_fit : public spectrum_io
{
  private:
    std::vector<std::string> fnames;
    int maxround; //max round number 
    int zf;
    enum fit_type peak_shape;
    double too_near_cutoff;
    double wx,wy; //max peak width (in pixel) used in peak partition and peak fitting (how many area to included in one peak fitting!)

    //OMP 
    int nthread;

    //new version 
    std::vector< std::deque<std::pair<int,int> > > clusters2; //peak cluster in peak partition
    double *pos_x_correction,*pos_y_correction; //diff between double and int coor of all peak
    std::vector< std::vector<int> > b,s;     //shared between peak_partition2 and prepare_fit2
    std::vector<int> peak_map;

    //peak can move in fitting process?
    std::vector<int> peak_cannot_move_flag;

private:


    bool gather_width();
    bool real_peak_fitting();
    // bool gaussain_convolution(std::vector<double> &,double,double,std::vector<double> &,std::vector<double> &,std::vector< std::vector<double> > &);
    bool generate_spectrum_gaussian(std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector< std::vector<double> > &);
    bool generate_spectrum_voigt(std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector< std::vector<double> > &);


    bool peak_partition(); 
    bool prepare_fit();
    bool assess_size();


    bool peak_reading_sparky(std::string);
    bool peak_reading_pipe(std::string);
    bool peak_reading_json(std::string);

public:
    std::vector<float *> spects;  //In peak fitting, float * spect (defined in spectrum_io) will still be used for first spectrum 
    std::vector<int> group; //peak groups, overlapped peaks are grouped together
    std::vector<int> nround; //how many round to get converged fitting?
    std::vector<double> err; //fitting RMSD 

    int nspect;
    std::vector< std::vector<double> > p_intensity_all_spectra; //p_intensity_all_spectra[peak_index][spec_index]; p_intensity_all_spectra[?][0] === p_intensities
    std::vector< std::vector<double> >num_sums; // numerical integral (volume) of each peak of each spectrum. num_sums[peak_index][spect_index]
    
    std::vector<gaussian_fit> fits; 


    spectrum_fit();
    ~spectrum_fit();

    bool init_all_spectra(std::vector<std::string> fnames);

    bool initflags_fit(int n,double c,int im, int zf_);
    bool print_peaks(std::string fname);
    // bool print_intensities(std::string outfname);
    bool generate_recon_and_diff_spectrum();
    bool clear_memory();
    bool fit_gather();
    bool peak_reading(std::string infname);
    bool peak_fitting();

    inline void set_peak_width(double x,double y) {wx=x;wy=y;}
};
