#include <vector>

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#include "peak_manipulation.h"

#ifndef DOSY
#define DOSY

/**
 * Define a functor for fitting a exp1 model
 */
struct exp_cost_functor
{
    exp_cost_functor(std::vector<double> peak_heights_, std::vector<double> gradients_);
    bool operator()(double const* const* parameters, double* residuals) const;

private:
    std::vector<double> peak_heights, gradients;
};


/**
 * Define a exp functor with baseline shift
*/
struct exp_cost_functor_baseline
{
    exp_cost_functor_baseline(std::vector<double> peak_heights_, std::vector<double> gradients_);
    bool operator()(double const* const* parameters, double* residuals) const;

private:
    std::vector<double> peak_heights, gradients;
};



/**
 * Class for fitting
*/
class CDosyFit
{
private:

    /**
     * Peak file manipulation object
    */
    class peak_manipulation pm;

    /**
     * @brief Fitted peaks from 1D DOSY
     * fitted_peak_heights[peak_index] = [1st trace height, 2nd trace height, ...], normalized to 1st trace height
     */
    std::vector<std::vector<double>> fitted_peak_heights;

    /**
     * Used to define peak multiplets and calculate mean decay rate
    */
    std::vector<double> x_ppm, y_ppm, height, volume;

    /**
     * Each element is a vector of peak indices in a multiplet. If a element has a size of 1, it is a singlet
     * peak_multiplets[i] = [peak_index1, peak_index2, ...]
    */
    std::vector<std::deque<int>> peak_multiplets;

    /**
     * Z filed strength gradient of all traces. Must be the same size as fitted_peak_heights[peak_index]
     */
    std::vector<double> z_gradients;

    /**
     * Corresponding trace indices for each z gradient
     * If all are used, trace_indices = [0, 1, 2, ..z_gradients.size()-1]
    */
    std::vector<int> trace_indices;

    /**
     * Diffusion constant D* for each peak
     */
    std::vector<double> diffusion_constants;

    /**
     * Baseline shift for each peak
     */
    std::vector<double> baseline_shifts;

    /**
     * @brief Fitting parameters[0] is 0 gradient peak height
    */
    std::vector<double> zero_gradient_peak_heights;

    /**
     * Include a constant shift in fitting
     */
    bool include_baseline;

public:
    CDosyFit(bool);
    CDosyFit();
    ~CDosyFit();

    bool read_fitted_peaks(const std::string &filename);
    bool read_z_gradients(const std::string &filename);
    bool fit_diffusion_constant(bool b_combine=false);
    bool rescale(double rescale_factor);
    bool write_result(const std::string &filename);
    bool write_peak_file(const std::string &filename);
};

#endif