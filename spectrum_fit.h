// fucntions from libcerf
extern "C"
{
  double voigt(double x, double sigma, double gamma);
  void re_im_w_of_z(double x, double y, double *r, double *i); // re_im_w_of_z
};


#include "fid_2d.h"

struct shared_data_2d
{
  static int n_verbose; //0: minimal, 1: normal
  static double error_scale; //scale of error in MC based error estimation
};

#ifndef FIT_TYPE
#define FIT_TYPE
enum fit_type
{ 
  gaussian_type,
  voigt_type,
  exact_type,
  voigt_lorentz_type, // voigt along x and lorentzian along y
  null_type
};
#endif



class gaussian_fit: public shared_data_2d
{
private:
  int my_index;
  int rmax;                                      // max round in fitting
  double noise_level;                            // noise level of spectrum.
  std::vector<int> x_int, y_int;                 // peak coordinate at grid
  std::vector<std::vector<double>> x_old, y_old; // used for check convergence
  double too_near_cutoff;                        // peaks near to another bigger peak will be removed in fititng, if d<sqrt(wx*wy)*too_near_cutoff
  double median_width_x, median_width_y;         // median values of whole spectrum, not median of this fitting region
  double minimal_height;                         // minimal peak height

  std::vector<double> min_x, max_x, min_y, max_y;      // region of fitting area of individual peak
  std::vector<std::vector<double>> peaks_fitting_data; // data used to fit each peak.

  int nround;                              // round number when exit
  double gaussian_fit_wx, gaussian_fit_wy; // maximal half peak width , used to define fitting area of each peak!!
  enum fit_type peak_shape;                // gaussian or voigt
  std::vector<std::array<int, 4>> valid_fit_region;
  double removal_cutoff; // cutoff to remove peak when it is overlapping with a big neighbor

#ifdef LMMIN
#else
  ceres::Solver::Options options;
#endif

  bool one_fit_exact(std::vector<double> &zz, double &x0, double &y0, double &a, double &r2x, double &r2y, double &sx, double &sy, double *e);
  bool one_fit_exact_shell(std::vector<double> &xx, std::vector<double> &yy, std::vector<double> &zz, const double x, const double y, double &aa, double &r2x, double &r2y, double &shiftx, double &shifty, double &phase_x, double &phase_y);

  bool one_fit_gaussian(int, int, std::vector<double> *zz, double &x0, double &y0, double &a, double &sigmax, double &sigmay, double *e);
  bool one_fit_gaussian_intensity_only(int, int, std::vector<double> *zz, double &x0, double &y0, double &a, double &sigmax, double &sigmay, double *e);
  bool multiple_fit_gaussian(int, int, std::vector<std::vector<double>> &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double *e);

  bool one_fit_voigt(int, int, std::vector<double> *, double &, double &, double &, double &, double &, double &, double &, double *, int n = 0);
  bool one_fit_voigt_core(int, int, std::vector<double> *, double &, double &, double &, double &, double &, double &, double &, double *);

  bool one_fit_voigt_intensity_only(int, int, std::vector<double> *, double &, double, double, double, double, double, double, double *);
  bool multiple_fit_voigt(int, int, std::vector<std::vector<double>> &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double &gammax, double &gammay, double *e, int n = 0);
  bool multiple_fit_voigt_core(int, int, std::vector<std::vector<double>> &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double &gammax, double &gammay, double *e);

  bool one_fit_voigt_lorentz(int xsize,int ysize,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay,double &gammax,double &gammay,double *e, int n);
  bool one_fit_voigt_lorentz_core(int xsize,int ysize,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &gammax,double &gammay,double *e);

  bool multiple_fit_voigt_lorentz(int xsize,int ysize, std::vector<std::vector<double> > &zz, double &x0, double &y0, std::vector<double> &a, double &sigmax, double &sigmay, double &gammax, double &gammay, double *e, int n);
  bool multiple_fit_voigt_lorentz_core(int xsize,int ysize, std::vector<std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &gammax, double &gammay, double *e);



  bool gaussain_convolution(const int,const int,const double a,const double x,const double y,const double sigmax,const double sigmay,int&,int&,int&,int&, std::vector<double> *kernel, double scale = 1.5);
  bool gaussain_convolution_within_region(const int ndx,const double a,const double x,const double y,const double sigmax,const double sigmay,int&,int&,int&,int&, std::vector<double> *kernel, double scale = 1.0);

  bool voigt_convolution(const int,const int,const double a,const double x,const double y,const double sigmax,const double sigmay,const double gammax,const double gammay,int&,int&,int&,int&, std::vector<double> *kernel, double scale = 1.5);
  bool voigt_convolution_2(const int,const int,const double a,const double x,const double y,const double sigmax,const double sigmay,const double gammax,const double gammay,int&,int&,int&,int&, std::vector<double> *kernel, double scale = 1.5);
  bool voigt_convolution_within_region(int ndx,const double a,const double x,const double y,const double sigmax,const double sigmay,const double gammax,const double gammay,int&,int&,int&,int&, std::vector<double> *kernel, double scale = 1.0);

  bool voigt_lorentz_convolution(const int,const int,const double a,const double x,const double y,const double sigmax,const double sigmay,const double gammax,const double gammay,int&,int&,int&,int&, std::vector<double> *kernel, double scale1 = 1.5, double scale2 = 3.0);
  bool voigt_lorentz_convolution_within_region(int ndx,const double a,const double x,const double y,const double sigmax,const double sigmay,const double gammax,const double gammay,int&,int&,int&,int&, std::vector<double> *kernel, double scale1 = 1.0, double scale2 = 3.0);

  // bool voigt_convolution_2(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> *kernel,int xdim2,int ydim2);

  bool find_highest_neighbor(int xx, int yy, int &m, int &n);
  bool limit_fitting_region_of_each_peak();
  bool run_multi_peaks(int rmax);
  bool run_single_peak();
  bool run_single_peak_exact();
  bool multi_spectra_run_multi_peaks(int rmax);
  bool multi_spectra_run_single_peak();

  bool generate_random_noise(int m, int n, int m2, int n2, std::vector<std::vector<float>> &);
  bool generate_theoretical_spectra(std::vector<std::vector<double>> &theorical_surface);

  bool get_pair_overlap(const int m, const int n, double &, int &) const;
  std::vector<std::pair<int, int>> get_possible_excess_peaks();
  int test_excess_peaks(int, int);

  bool get_pair_overlap(const int m, const int n, double &, double &) const;


public:
  // will be read directly in fit_gather step of spectrum_fit.cpp
  // updated in fitting at each iteration
  std::vector<int> to_remove;         // lable to_be_remvoed peaks identifed in fitting process
  std::vector<double> amp;              // peak intensity, flattened as 1D vector  a[peak index][spe index] ==> a[peak index * nspectra + spe index]
  std::vector<double> x, y;           // peak coordinates
  int npeak;                 // number of peaks in this gaussian_fit object (before fitting)
  std::vector<double> sigmax, sigmay; // peak width
  std::vector<double> gammax, gammay; // Lorentzian peak width
  std::vector<double> err;            // fitting residual (RMSD)
  std::vector<int> original_ndx;      // for debug only.
  std::vector<int> cannot_move;
  std::vector<double> original_ratio;
  int peak_sign; // peak sign, 1: positive, -1: negative

  std::vector<int> removed_peaks; //original_ndx of removed peaks.

  std::vector<std::vector<double>> batch_x, batch_y, batch_sigmax, batch_sigmay, batch_gammax, batch_gammay;
  std::vector<std::vector<double>> batch_a;

  std::vector<double> delta_x, delta_y, delta_sigmax, delta_sigmay, delta_gammax, delta_gammay;
  std::vector<std::vector<double>> delta_amplitude, delta_volume; // delta[peak_index][spe_index]

  std::vector<std::vector<double>> analytical_spectra; // analytical spectrum for each peak
  std::vector<double> peaks_total;                     // reconstructed 2D matrix from fitting: sum of analytical_spectra

  std::vector<std::vector<double>> num_sum; // numerical integral of each peak num_sum[spe index][peak index]

  // other varibles need direct access
  int xstart, ystart;
  int xdim, ydim;
  int xydim; // xdim*ydim, size of each spectrum
  double xppm_per_step, yppm_per_step;
  /**
   * 2D spectrum matrix flattened, outlayer: list of spectra
   * inner layer; flattened 2D matrix of each spectrum, column major order
   * surface[spec_index][x_index*ydim+y_index]
   */
  std::vector<double> surface; 
  int nspectra; //size of surface, number of spectra in this gaussian_fit object

  gaussian_fit();
  ~gaussian_fit();

  bool init(int, int, int, int, int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<int>, std::vector<int>,double,double);
  bool run(int flag = 1);
  bool run_with_error_estimation(int, int, int nround = 10);
  bool assess_size();
  int get_my_index();
  int get_nround();
  bool change_sign();
  void set_everything(fit_type t, int r, int index);
  void set_everything_wasm(int t,int r,int index);
  void set_peak_paras(double x, double y, double noise, double height, double near, double xppm, double yppm, double cutoff);
};

class spectrum_fit : public fid_2d, public shared_data_2d
{
private:
  std::vector<std::string> fnames;
  int maxround; // max round number
  enum fit_type peak_shape;
  double too_near_cutoff;
  double wx, wy;         // max peak width (in pixel) used in peak partition and peak fitting (how many area to included in one peak fitting!)
  double removal_cutoff; // cutoff to remove peak when it is overlapping with a big neighbor

  // OMP
  int nthread;

  // peak can move in fitting process?
  std::vector<int> peak_cannot_move_flag;
  std::vector<int> peak_map,peak_map2,peak_map3;
  int cluster_counter;

  int flag_with_error;
  int zf1, zf2; // times of zero filling along x and y, needed for generation of realistic noise.
  int err_nround;

  std::vector<int> excluded_peaks;

private:
  bool gather_width();
  bool real_peak_fitting();
  bool real_peak_fitting_with_error(int, int, int);

  bool peak_partition();
  bool peak_partition_core(int flag);
  bool assess_size();

  bool peak_reading_sparky(std::string);
  bool peak_reading_pipe(std::string);
  bool peak_reading_json(std::string);

public:
  std::vector<std::vector<float>> spects;
  std::vector<int> group;      // peak groups, overlapped peaks are grouped together
  std::vector<int> nround;     // how many round to get converged fitting?
  std::vector<double> err;     // fitting RMSD

  int nspect;
  std::vector<std::vector<double>> p_intensity_all_spectra; // p_intensity_all_spectra[peak_index][spec_index]; p_intensity_all_spectra[?][0] === p_intensities
  std::vector<std::vector<double>> num_sums;                // numerical integral (volume) of each peak of each spectrum. num_sums[peak_index][spect_index]

  // std::vector<double> delta_x,delta_y,delta_sigmax,delta_sigmay,delta_gammax,delta_gammay;
  // std::vector< std::vector<double>> delta_amplitude,delta_volume;

  std::vector<gaussian_fit> fits;

  spectrum_fit();
  ~spectrum_fit();

  bool init_all_spectra(std::vector<std::string> fnames);

  bool initflags_fit(int n, double r, double c, int im);
  bool init_error(int, int, int, int);
  bool print_peaks(std::string fnames, bool b_recon, std::string fold_name="./");
  // bool print_intensities(std::string outfname);
  bool generate_recon_and_diff_spectrum(std::string);
  bool fit_gather_original();
  bool fit_gather(int);
  bool peak_reading(std::string);
  bool peak_fitting();

  inline void set_peak_width(double x, double y)
  {
    wx = x;
    wy = y;
  }
};
