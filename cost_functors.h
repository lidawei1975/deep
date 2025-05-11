#ifdef LMMIN
#include "lmminimizer.h"
#endif

// fucntions from libcerf
extern "C"
{
  double voigt(double x, double sigma, double gamma);
  void re_im_w_of_z(double x, double y, double *r, double *i); // re_im_w_of_z
};

#ifndef LDW_PARAS
#define LDW_PARAS
#define SMALL 1e-10
#define PI 3.14159265358979323846
#define M_SQRT_PI 1.772453850905516
#define M_SQRT_2PI 2.506628274631000
#define M_1_SQRT_PI 0.564189583547756
#endif

/**
 * IMPORTANT:
 * In our Voigt definition, total peak volume is 1.0 (constant). Fittted a is the peak volume.
 * In our Lorentz definition, total peak volume gamma*pi. (not constant). Fitted a is the peak height.
 * In our Gaussian definition, total peak volume is sqrt(sigma*pi). Fitted a is the peak height.
*/

/**
 * for 2D voigt fitting, analytical derivative
*/
#ifdef LMMIN
class mycostfunction_voigt : public ldwcostfunction
#else
class mycostfunction_voigt : public ceres::CostFunction
#endif
{

private:
  int n_datapoint; // size of x(y,z)
  int xdim, ydim;
  double const *z; // x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt();
  mycostfunction_voigt(int, int, double const *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

/**
 * A mixed model: Voigt along x and Lorentzian along y
*/
#ifdef LMMIN
class mycostfunction_voigt_lorentz : public ldwcostfunction
#else
class mycostfunction_voigt_lorentz : public ceres::CostFunction
#endif
{
private:
    int n_datapoint; // size of x(y,z)
    int xdim, ydim;
    double *z; // x,y -> coor, z-> spectra data
    void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
    void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

    public:
    ~mycostfunction_voigt_lorentz();
    mycostfunction_voigt_lorentz(int, int, double *);
    bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
    inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
    inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};


// for 2D Gaussian fitting, analytical derivative
#ifdef LMMIN
class mycostfunction_gaussian : public ldwcostfunction
#else
class mycostfunction_gaussian : public ceres::CostFunction
#endif
{

private:
  int n_datapoint; // size of x(y,z)
  int xdim, ydim;
  double *z; // x,y -> coor, z-> spectra data

public:
  ~mycostfunction_gaussian();
  mycostfunction_gaussian(int, int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};


struct Exactshape // with phasing error
{
  Exactshape(double *zz_, int nx_, int zfx_, int ny_, int zfy_) : zz(zz_), nx(nx_), zfx(zfx_), ny(ny_), zfy(zfy_){};
  bool operator()(const double *const a, const double *const x0, const double *const y0, const double *const r2x, const double *const r2y, const double *const sx, const double *const sy, double *residue) const;
  bool val(const double a, const double x0, const double y0, const double r2x, const double r2y, const double sx, const double sy, double *v) const;

private:
  const double *zz;
  const int nx, zfx, ny, zfy;
};

/**
 * 1D part
*/
/**
 * @brief for 1D voigt fitting, analytical derivative
 * fit multiple peaks simultaneously
 */
#ifdef LMMIN
class mycostfunction_nvoigt1d : public ldwcostfunction
#else
class mycostfunction_nvoigt1d : public ceres::CostFunction
#endif
{

private:
  int np;          // number of peaks
  int n_datapoint; // size of x(y,z)
  double *z;       // x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_nvoigt1d();
  mycostfunction_nvoigt1d(int, int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

/**
 * @brief for 1D voigt fitting, analytical derivative
 * fit one peak
 */
#ifdef LMMIN
class mycostfunction_voigt1d : public ldwcostfunction
#else
class mycostfunction_voigt1d : public ceres::CostFunction
#endif
{

private:
  int n_datapoint; // size of x(y,z)
  double *z;       // x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt1d();
  mycostfunction_voigt1d(int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

/**
 * @brief for pseudo 2D voigt fitting, analytical derivative
 * Peak amplitude is defined as A = A0*exp(-t*t*D) where A0 and D are fitting parameters
 * while t=[0,1,2,3,...,n-1] is the time delay
 */
#ifdef LMMIN
class mycostfunction_voigt1d_doesy : public ldwcostfunction
#else
class mycostfunction_voigt1d_doesy : public ceres::CostFunction
#endif
{

private:
  double z_gradient_squared;           // time delay. t=0,1,2,3,...,n-1
  int n_datapoint; // size of z(x)
  double *z;       // x -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt1d_doesy();
  mycostfunction_voigt1d_doesy(double, int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

// for 1D lorentz fitting, analytical derivative
#ifdef LMMIN
class mycostfunction_lorentz1d : public ldwcostfunction
#else
class mycostfunction_lorentz1d : public ceres::CostFunction
#endif
{

private:
  int n_datapoint; // size of x(y,z)
  double *z;       // x -> coor, z-> spectra data

public:
  ~mycostfunction_lorentz1d();
  mycostfunction_lorentz1d(int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

// for 1D Gaussian fitting, analytical derivative
#ifdef LMMIN
class mycostfunction_gaussian1d : public ldwcostfunction
#else
class mycostfunction_gaussian1d : public ceres::CostFunction
#endif
{

private:
  int n_datapoint; // size of x(y,z)
  double *z;       // x -> coor, z-> spectra data

public:
  ~mycostfunction_gaussian1d();
  mycostfunction_gaussian1d(int, double *);
  bool Evaluate(double const *const *, double *, double **) const;
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

