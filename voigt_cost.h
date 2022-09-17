
extern "C"  
{
    double voigt(double x, double sigma, double gamma);
    void re_im_w_of_z( double x, double y ,double *r, double *i); //re_im_w_of_z
};

#define SMALL 1e-10
#define PI 3.14159265358979323846
#define M_SQRT_PI 1.772453850905516
#define M_SQRT_2PI 2.506628274631000
#define M_1_SQRT_PI 0.564189583547756


//for 1D voigt fitting, analytical derivative
class mycostfunction_voigt1d : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  double *x,*z; //x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction_voigt1d();
  mycostfunction_voigt1d(int,double *, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  bool get_residual(const double a, const double x0, const double sigma, const double gamma, double *residual);
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};

//for 2D voigt fitting, analytical derivative
class mycostfunction : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  double *x,*y,*z; //x,y -> coor, z-> spectra data

  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
  void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

public:
  ~mycostfunction();
  mycostfunction(int,double *,double *, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};


//for 2D Gaussian fitting, analytical derivative
class mycostfunction_gaussian : public ceres::CostFunction
{

private:
  int n_datapoint; //size of x(y,z)
  double *x,*y,*z; //x,y -> coor, z-> spectra data

  double gaussian_funtion(double x,double y,double sx,double sy) const;


public:
  ~mycostfunction_gaussian();
  mycostfunction_gaussian(int,double *,double *, double *);
  bool Evaluate(double const *const *, double *, double **) const;
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
};