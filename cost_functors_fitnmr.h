#ifndef COST_FUNCTORS_FITNMR_H
#define COST_FUNCTORS_FITNMR_H

#ifdef LMMIN
#include "lmminimizer.h"
#else
#include "ceres/ceres.h"
#include "glog/logging.h"
#endif

#include <vector>

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifdef LMMIN
class mycostfunction_fitnmr_base : public ldwcostfunction
#else
class mycostfunction_fitnmr_base : public ceres::CostFunction
#endif
{
protected:
  int n_datapoint;
  int xdim, ydim;
  double *z;

  double nx, ny;
  double aqx, offx, endx;
  double aqy, offy, endy;

  void none_helper(const double omega,
                   const double omega0,
                   const double r2,
                   const double aq,
                   double *vv,
                   double *r_omega0,
                   double *r_r2) const;

  void none_helper(const double omega,
                   const double omega0,
                   const double r2,
                   const double aq,
                   double *vv) const;

  void sp1_helper(const double omega,
                  const double omega0,
                  const double r2,
                  const double aq,
                  const double off,
                  const double end,
                  double *vv,
                  double *r_omega0,
                  double *r_r2) const;

  void sp1_helper(const double omega,
                  const double omega0,
                  const double r2,
                  const double aq,
                  const double off,
                  const double end,
                  double *vv) const;

  void sp2_helper(const double omega,
                  const double omega0,
                  const double r2,
                  const double aq,
                  const double off,
                  const double end,
                  double *vv,
                  double *r_omega0,
                  double *r_r2) const;

  void sp2_helper(const double omega,
                  const double omega0,
                  const double r2,
                  const double aq,
                  const double off,
                  const double end,
                  double *vv) const;

  void sp3_helper(const double omega,
                  const double omega0,
                  const double r2,
                  const double aq,
                  const double off,
                  const double end,
                  double *vv,
                  double *r_omega0,
                  double *r_r2) const;

  void sp3_helper(const double omega,
                  const double omega0,
                  const double r2,
                  const double aq,
                  const double off,
                  const double end,
                  double *vv) const;

  void doublet_helper(const double omega,
                      const double omega0,
                      const double r2,
                      const double j,
                      const double aq,
                      const double off,
                      const double end,
                      double *vv,
                      double *r_omega0,
                      double *r_r2,
                      double *r_j) const;

  void doublet_helper(const double omega,
                      const double omega0,
                      const double r2,
                      const double j,
                      const double aq,
                      const double off,
                      const double end,
                      double *vv) const;

  void singlet_helper(const double omega,
                      const double omega0,
                      const double r2,
                      const double n,
                      const double aq,
                      const double off,
                      const double end,
                      double *vv,
                      double *r_omega0,
                      double *r_r2) const;

  void singlet_helper(const double omega,
                      const double omega0,
                      const double r2,
                      const double n,
                      const double aq,
                      const double off,
                      const double end,
                      double *vv) const;

public:
  ~mycostfunction_fitnmr_base();
  mycostfunction_fitnmr_base(int, int, double *,
                             double, double, double,
                             double,
                             double, double, double,
                             double);
#ifndef LMMIN
  inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
  inline void set_n_residuals(int n) { set_num_residuals(n); };
#endif
};

/**
 * FitNMR singlet in x and singlet in y.
 *
 * Parameters:
 *   xx[0][0] = amplitude a
 *   xx[1][0] = x center x0
 *   xx[2][0] = x decay r2x
 *   xx[3][0] = y center y0
 *   xx[4][0] = y decay r2y
 */
#ifdef LMMIN
class mycostfunction_fitnmr_singlet_singlet : public mycostfunction_fitnmr_base
#else
class mycostfunction_fitnmr_singlet_singlet : public mycostfunction_fitnmr_base
#endif
{
public:
  ~mycostfunction_fitnmr_singlet_singlet();
  mycostfunction_fitnmr_singlet_singlet(int, int, double *,
                                        double, double, double,
                                        double,
                                        double, double, double,
                                        double);
  bool Evaluate(double const *const *, double *, double **) const;
};

/**
 * FitNMR doublet in x and singlet in y.
 *
 * Parameters:
 *   xx[0][0] = amplitude a
 *   xx[1][0] = x center x0
 *   xx[2][0] = x decay r2x
 *   xx[3][0] = x doublet splitting j
 *   xx[4][0] = y center y0
 *   xx[5][0] = y decay r2y
 */
#ifdef LMMIN
class mycostfunction_fitnmr_doublet_singlet : public mycostfunction_fitnmr_base
#else
class mycostfunction_fitnmr_doublet_singlet : public mycostfunction_fitnmr_base
#endif
{
public:
  ~mycostfunction_fitnmr_doublet_singlet();
  mycostfunction_fitnmr_doublet_singlet(int, int, double *,
                                        double, double, double,
                                        double,
                                        double, double, double,
                                        double);
  bool Evaluate(double const *const *, double *, double **) const;
};

#endif
