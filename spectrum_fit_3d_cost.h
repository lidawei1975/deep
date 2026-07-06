#ifndef SPECTRUM_FIT_3D_COST_H
#define SPECTRUM_FIT_3D_COST_H

#include <tuple>
#include <vector>

#include "lmminimizer.h"

// ============================================================================
// Helper structures and functions for 3D peak fitting
// ============================================================================

// Structure to hold 3D peak parameters
struct Peak3D {
    float amplitude;
    float x_center, y_center, z_center;
    float x_width, y_width, z_width;  // FWHH along each axis
    float x_lfrac, y_lfrac, z_lfrac;  // Lorentzian fraction in [0, 1]
    int partition_idx;

    Peak3D() : amplitude(0), x_center(0), y_center(0), z_center(0),
               x_width(1), y_width(1), z_width(1),
               x_lfrac(0.5f), y_lfrac(0.5f), z_lfrac(0.5f),
               partition_idx(-1) {}

    Peak3D(float amp, float xc, float yc, float zc, float xw, float yw, float zw)
        : amplitude(amp), x_center(xc), y_center(yc), z_center(zc),
          x_width(xw), y_width(yw), z_width(zw),
          x_lfrac(0.5f), y_lfrac(0.5f), z_lfrac(0.5f),
          partition_idx(-1) {}

    Peak3D(float amp, float xc, float yc, float zc, float xw, float yw, float zw,
           float xl, float yl, float zl)
        : amplitude(amp), x_center(xc), y_center(yc), z_center(zc),
          x_width(xw), y_width(yw), z_width(zw),
          x_lfrac(xl), y_lfrac(yl), z_lfrac(zl),
          partition_idx(-1) {}
};

constexpr int kPeakParamCount = 10;

struct SparseCubeIndex3D {
    int min_x = 0;
    int min_y = 0;
    int min_z = 0;
    int max_x = -1;
    int max_y = -1;
    int max_z = -1;
    int ny = 0;
    int nz = 0;
    int nx = 0; // can be derived from max_x - min_x + 1, but stored for convenience
    /**
     * Size is (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1)
     * -1 means invalid data point
     * 0,1,2,... means the index of the corresponding voxel in the partition voxel list (sparse index)
     * For example, if grid_to_sparse_idx[cube_flat_index(cube, x, y, z)] == 5,
     * it means the voxel at (x,y,z) in the full grid corresponds to the 5th voxel in the partition voxel list.
     */
    std::vector<int> grid_to_sparse_idx;
};

int cube_flat_index(const SparseCubeIndex3D& cube, int x, int y, int z);

SparseCubeIndex3D build_sparse_cube_index(const std::vector<int>& x_coords,
                                          const std::vector<int>& y_coords,
                                          const std::vector<int>& z_coords);

float eval_axis_voigt_value(float coord, float center, float fwhh, float lfrac, float sqrt2ln2);

void eval_axis_voigt_with_derivatives(
    float coord,
    float center,
    float fwhh,
    float lfrac,
    float sqrt2ln2,
    float &V,
    float &dV_dcenter,
    float &dV_dfwhh,
    float &dV_dlfrac);

void accumulate_peak_model_sparse(
    std::vector<float>& model,
    const Peak3D& peak,
    const SparseCubeIndex3D& cube,
    float sqrt2ln2,
    float support_fwhh_factor);

void calculate_total_3d_model_sparse_with_support(std::vector<float>& model,
                                                   const std::vector<Peak3D>& peaks,
                                                   const std::vector<int>& x_coords,
                                                   const std::vector<int>& y_coords,
                                                   const std::vector<int>& z_coords,
                                                   float support_fwhh_factor);

// Calculate 3D pseudo-Voigt peak model at a single point
float calculate_3d_peak_model(float x, float y, float z, const Peak3D& peak);

// Calculate total model for all peaks using sparse coordinates
void calculate_total_3d_model_sparse(std::vector<float>& model,
                                     const std::vector<Peak3D>& peaks,
                                     const std::vector<int>& x_coords,
                                     const std::vector<int>& y_coords,
                                     const std::vector<int>& z_coords);

// Objective function for 3D peak fitting with asymmetric weighting and regularization
std::vector<float> objective_function_3d(
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    float noise_floor,
    float alpha,
    float lambda);

// Calculate analytical Jacobian for 3D pseudo-Voigt peak model (data term only)
void calculate_data_jacobian(
    std::vector<float>& jacobian,
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    const std::vector<float>& model,
    const std::vector<float>& residuals,
    float alpha);

// Calculate numerical Jacobian for regularization term using finite differences
void calculate_regularization_jacobian_numerical(
    std::vector<float>& jac_reg_row,
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    float noise_floor,
    float alpha,
    float lambda);

// Mixed Jacobian computation: Analytical for data term, numerical for regularization
void objective_function_3d_mixed_jacobian(
    std::vector<float>& residuals,
    std::vector<float>* jacobian,
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    float noise_floor,
    float alpha,
    float lambda);

class Peak3DNoRegCostFunction : public ldwcostfunction {
public:
    const std::vector<float>* data = nullptr;
    const std::vector<int>* x_coords = nullptr;
    const std::vector<int>* y_coords = nullptr;
    const std::vector<int>* z_coords = nullptr;
    int n_params = 0;

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobian) const override;
};

class SinglePeak3DCostFunction : public ldwcostfunction
{
public:
    const std::vector<float>* data = nullptr;
    const std::vector<int>* x_coords = nullptr;
    const std::vector<int>* y_coords = nullptr;
    const std::vector<int>* z_coords = nullptr;
    int n_params = kPeakParamCount;
    float alpha = 1.0f;
    float lambda = 0.0f;
    float noise_floor = 0.0f;

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobian) const override;
};

class Peak3DCostFunction : public ldwcostfunction {
public:
    const std::vector<float>* data = nullptr;
    const std::vector<int>* x_coords = nullptr;
    const std::vector<int>* y_coords = nullptr;
    const std::vector<int>* z_coords = nullptr;
    float noise_floor = 0.0f;
    float alpha = 1.0f;
    float lambda = 0.0f;
    int n_params = 0;  // Total number of parameters (n_peaks * 10)

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobian) const override;
};

#endif
