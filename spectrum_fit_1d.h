extern "C"
{
  double voigt(double x, double sigma, double gamma);
  void re_im_w_of_z(double x, double y, double *r, double *i); // re_im_w_of_z
};

#include "fid_1d.h"

#ifndef LDW_PARA
#define LDW_PARA
#define SMALL 1e-10
#define PI 3.14159265358979323846
#define M_SQRT_PI 1.772453850905516
#define M_SQRT_2PI 2.506628274631000
#define M_1_SQRT_PI 0.564189583547756
#endif

#ifndef FIT_TYPE
#define FIT_TYPE
enum fit_type
{
  gaussian_type,
  voigt_type,
  lorentz_type,
  null_type
};
#endif

#ifdef LMMIN
    #include "lmminimizer.h"
#else
    #include "ceres/ceres.h"
    #include "glog/logging.h"
    using ceres::CostFunction;
    using ceres::AutoDiffCostFunction;
    using ceres::NumericDiffCostFunction;
    using ceres::Problem;
    using ceres::Solver;
    using ceres::Solve;
#endif

class gaussian_fit_1d: public shared_data_1d
{
private:
  int n_patch;

  bool b_negative;

  double median_width_x; // median peak width in x direction. Copied from spectrum_fit_1d

  double spectrum_scale;
  int rmax;                               // max round in fitting
  std::vector<int> x_int;                 // peak coordinate at grid
  std::vector<std::vector<double>> x_old; // used for check convergence
  double too_near_cutoff;                 // peaks near to another bigger peak will be removed in fititng, if d<sqrt(wx*wy)*too_near_cutoff
  double minimal_height;                  // minimal peak height
  double noise_level;                     // level of noise
  int nround;                             // round number when exit
  double wx;                              // maximal half peak width , used to define fitting area of each peak!!
  enum fit_type type;                     // gaussian or voigt
  int nspect; // number of input spectra


#ifndef LMMIN
  ceres::Solver::Options options;
#endif

  bool one_fit_gaussian(int, std::vector<double> *zz, double &x0, double &a, double &sigmax, double &e);
  bool one_fit_lorentz(int, std::vector<double> *zz, double &x0, double &a, double &gammax, double &e);
  bool one_fit_voigt(int, std::vector<double> *zz, double &, double &, double &, double &, double &, int n = 0, int n2 = 0);
  bool one_fit_voigt_core(int, std::vector<double> *zz, double &x0, double &a, double &sigmax, double &gammax, double &e);

  bool multi_fit_gaussian(int xdim, std::vector<std::vector<double>> &zz, double &x0, std::vector<double> &a, double &sigmax, double &e);
  bool multi_fit_lorentz(int xdim, std::vector<std::vector<double>> &zz, double &x0, std::vector<double>  &a, double &gammax, double &e);
  bool multi_fit_voigt(int xdim, std::vector<std::vector<double>> zz, double &x0, std::vector<double> &a, double &sigmax, double &gammax, double &e, int n, int n2);
  bool multi_fit_voigt_core(int xdim, std::vector<std::vector<double>> &zz, double &x0, std::vector<double> &a, double &sigmax, double &gammax, double &e);

  bool multi_fit_voigt_doesy(const int xdim,std::vector<std::vector<double>> zz, double &x0, double &a, double &d, double &sigmax, double &gammax, double &e, int n, int n2);
  bool multi_fit_voigt_core_doesy(const int xdim,std::vector<std::vector<double>> &zz, double &x0, double &a, double &d, double &sigmax, double &gammax, double &e);

  bool gaussain_convolution(double a, double x, double sigmax, std::vector<double> *kernel, int &i0, int &i1, double scale);
  bool gaussain_convolution_with_limit(int ndx, double a, double x, double sigmax, std::vector<double> *kernel, int &i0, int &i1, double scale);
  bool voigt_convolution(double a, double x, double sigmax, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale);
  bool voigt_convolution_with_limit(int ndx, double a, double x, double sigmax, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale);
  bool lorentz_convolution(double a, double x, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale);
  bool lorentz_convolution_with_limit(int ndx, double a, double x, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale);
  bool find_highest_neighbor(int xx, int &mm);
  bool limit_fitting_region_of_each_peak();
  int test_possible_removal(double, double, double, double, double, double, double, double);
  bool run_single_peak();
  bool run_multi_peaks();
  bool run_single_peak_multi_spectra();
  bool run_multi_peaks_multi_spectra();
  bool generate_random_noise(int m, int m2, std::vector<float> &noise_spectrum);

public:
  // will be read directly in fit_gather step of spectrum_fit_1d.cpp
  // updated in fitting at each iteration
  std::vector<int> to_remove;                       // lable to_be_remvoed peaks identifed in fitting process
  std::vector<std::vector<double>> a;               // peak intensity a[peak_index][spectra_index]
  std::vector<double>   a_at_time_zero;             // peak intensity at time zero. used in doesy fitting only
  std::vector<double> x;                            // peak coordinates
  std::vector<double> x_as_input;                   // peak coordinates before fitting
  std::vector<double> sigmax;                       // Gaussian peak shape parameter. IMPORTANT: in Gaussian fit, this is actually 2*sigma*sigma
  std::vector<double> gammax;                       // Lorentzian peak shape parameter
  std::vector<double> err;                          // fitting residual (RMSD)
  std::vector<int> original_ndx;                    // for debug only.
  std::vector<std::vector<double>> num_sum;         // numerical integral of each peak num_sum[peak_index][spectra_index]
  std::vector<double> surface;                      // 1D spectrum
  std::vector<std::vector<double>> surfaces;        // all 1D spectra
  std::vector<double> diffusion_coefficient;        // diffusion coefficient. used in doesy fitting only
  int xdim;                                         // size of spectrum part
  std::vector<std::array<int, 2>> valid_fit_region; // limit fitting region of overlappin gpeaks.
  std::vector<int> x_range_left, x_range_right;

  // initial peak position and spectral height at that position
  std::vector<double> original_spectral_height;
  std::vector<double> original_peak_pos;

  // for error estimation
  std::vector<std::vector<double>> batch_x, batch_sigmax, batch_gammax;
  std::vector<std::vector<std::vector<double>>> batch_a;

  // positonal information
  int begin, stop, left_patch, right_patch, n_initial;

  gaussian_fit_1d();
  ~gaussian_fit_1d();

  bool save_postion_informations(int, int, int, int, int);
  bool gaussian_fit_init(std::vector<std::vector<float>> &d, std::vector<double> p1, std::vector<double> sigma, std::vector<double> gamma, std::vector<std::vector<double>> inten, std::vector<int> ndx);
  bool run_peak_fitting(bool flag_first = true);
  bool run_with_error_estimation(int zf1, int n_error_round);
  int get_nround();
  void set_up(fit_type, int, double, double, double,bool,double);
};

class spectrum_fit_1d : public fid_1d
{
private:

  std::vector<std::string> fnames; // file names of all input spectra
  std::vector<std::vector<float>> spects; // all input spectra
  int nspect; // number of input spectra

  bool b_negative;

  int n_patch;
  int zf, error_nround;
  int rmax;
  double median_width_x;
  double to_near_cutoff;
  double w_smooth, w_positive, w_fit;

  std::vector<double> p1, p1_ppm, sigmax, gammax, p_intensity;
  std::vector<std::vector<double>> p_intensity_all_spectra; //for pseudo 2D. [peak_index][spectra_index]
  std::vector<std::string> user_comments;
  std::vector<double> confident_level;

  // for error estimation
  std::vector<std::vector<double>> batch_p1, batch_sigmax, batch_gammax, batch_p_intensity;

  enum fit_type peak_shape;

  std::vector<gaussian_fit_1d> fits;

  bool peak_reading_pipe(const std::string &content);
  bool peak_reading_json(const std::string &content);
  bool peak_reading_sparky(const std::string &content);
  bool assess_size();
  bool gather_result();
  bool gather_result_with_error_estimation(int);
  bool peak_partition_1d_for_fit(double,double);
  bool label_baseline_peaks();

public:
  std::vector<int> fit_peak_index;
  std::vector<double> fit_p1, fit_p1_ppm, fit_sigmax, fit_gammax, fit_p_intensity; // lazy. should be private
  std::vector<std::vector<double>> fit_num_sum,fit_p_intensity_all_spectra; //for pseudo 2D. [peak_index][spectra_index]
  std::vector<double> fit_err;
  std::vector<int> fit_nround;
  std::vector<int> background_peak_flag; // 0: not background peak, 1: background peak

  spectrum_fit_1d();
  ~spectrum_fit_1d();
  bool init_all_spectra(std::vector<std::string> finames,bool);
  bool set_for_one_spectrum();
  bool init_fit(int, int, double);
  bool init_error(int, int);
  bool peak_fitting(double spectrum_begin=100.0,double spectrum_end=-100.0);
  bool output(std::string outfname);
  bool peak_reading(std::string outfname); //default is allow negative peaks.
  bool set_peaks(const spectrum_1d_peaks);
  bool get_fitted_peaks(spectrum_1d_peaks &);
  bool output_json(std::string outfname,bool b_individual_peaks);
  bool write_recon(std::string folder_name); 

  /**
   * Below functions are mainly used for web assembly version to read froma and write to buffer
   * that is accessible from JavaScript.
   */
  bool prepare_to_read_additional_spectrum_from_buffer(bool);
  bool read_additonal_spectrum_from_buffer(std::vector<float> &buffer);
  
  std::string  output_as_string(int c);
  std::string output_json_as_string(bool b_individual_peaks);
  bool peak_reading_from_string(const std::string &content, int read_type);
  int get_size_of_recon();
  std::vector<float> spe_recon;
  uintptr_t get_data_of_recon(int);
};