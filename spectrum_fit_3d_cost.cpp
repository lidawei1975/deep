#include "spectrum_fit_3d_cost.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

// ============================================================================
// Constants for 3D peak fitting (cost helpers)
// ============================================================================

static constexpr float FwhhFacctor = 3.0f;

int cube_flat_index(const SparseCubeIndex3D& cube, int x, int y, int z)
{
    return (x - cube.min_x) * cube.ny * cube.nz + (y - cube.min_y) * cube.nz + (z - cube.min_z);
}

SparseCubeIndex3D build_sparse_cube_index(const std::vector<int>& x_coords,
                                          const std::vector<int>& y_coords,
                                          const std::vector<int>& z_coords)
{
    SparseCubeIndex3D cube;
    if (x_coords.empty()) {
        return cube;
    }

    cube.min_x = *std::min_element(x_coords.begin(), x_coords.end());
    cube.min_y = *std::min_element(y_coords.begin(), y_coords.end());
    cube.min_z = *std::min_element(z_coords.begin(), z_coords.end());
    cube.max_x = *std::max_element(x_coords.begin(), x_coords.end());
    cube.max_y = *std::max_element(y_coords.begin(), y_coords.end());
    cube.max_z = *std::max_element(z_coords.begin(), z_coords.end());

    cube.nx = cube.max_x - cube.min_x + 1;
    cube.ny = cube.max_y - cube.min_y + 1;
    cube.nz = cube.max_z - cube.min_z + 1;

    cube.grid_to_sparse_idx.assign(cube.nx * cube.ny * cube.nz, -1);
    for (size_t i = 0; i < x_coords.size(); ++i) {
        cube.grid_to_sparse_idx[cube_flat_index(cube, x_coords[i], y_coords[i], z_coords[i])] = (int)i;
    }

    return cube;
}

float eval_axis_voigt_value(float coord, float center, float fwhh, float lfrac, float sqrt2ln2)
{
    float f = std::max(fwhh, 1e-6f);
    float sig = f / (2.0f * sqrt2ln2);
    float gam = f / 2.0f;
    float d = coord - center;
    float d2 = d * d;
    float G = std::exp(-d2 / (2.0f * sig * sig));
    float L = (gam * gam) / (d2 + gam * gam);
    float lf = std::min(1.0f, std::max(0.0f, lfrac));
    return (1.0f - lf) * G + lf * L;
}

void eval_axis_voigt_with_derivatives(
    float coord,
    float center,
    float fwhh,
    float lfrac,
    float sqrt2ln2,
    float &V,
    float &dV_dcenter,
    float &dV_dfwhh,
    float &dV_dlfrac)
{
    float f = std::max(fwhh, 1e-6f);
    float sig = f / (2.0f * sqrt2ln2);
    float gam = f / 2.0f;

    float d = coord - center;
    float d2 = d * d;

    float sig2 = sig * sig;
    float gam2 = gam * gam;

    float G = std::exp(-d2 / (2.0f * sig2));
    float L = gam2 / (d2 + gam2);

    float lf = std::min(1.0f, std::max(0.0f, lfrac));

    V = (1.0f - lf) * G + lf * L;
    dV_dcenter = (1.0f - lf) * G * (d / sig2) + lf * (L * L * (2.0f * d / gam2));

    float dsig_dfwhh = 1.0f / (2.0f * sqrt2ln2);
    float dG_dfwhh = G * (d2 / (sig * sig2)) * dsig_dfwhh;
    float dL_dfwhh = (gam * d2) / ((d2 + gam2) * (d2 + gam2));
    dV_dfwhh = (1.0f - lf) * dG_dfwhh + lf * dL_dfwhh;

    dV_dlfrac = L - G;
}

void accumulate_peak_model_sparse(
    std::vector<float>& model,
    const Peak3D& peak,
    const SparseCubeIndex3D& cube,
    float sqrt2ln2,
    float support_fwhh_factor)
{
    if (cube.max_x < cube.min_x || cube.max_y < cube.min_y || cube.max_z < cube.min_z) {
        return;
    }

    float fwhhx = std::max(peak.x_width, 1e-6f);
    float fwhhy = std::max(peak.y_width, 1e-6f);
    float fwhhz = std::max(peak.z_width, 1e-6f);

    int x_lo = cube.min_x;
    int x_hi = cube.max_x;
    int y_lo = cube.min_y;
    int y_hi = cube.max_y;
    int z_lo = cube.min_z;
    int z_hi = cube.max_z;

    if (support_fwhh_factor > 0.0f) {
        x_lo = std::max(cube.min_x, (int)std::floor(peak.x_center - support_fwhh_factor * fwhhx));
        x_hi = std::min(cube.max_x, (int)std::ceil(peak.x_center + support_fwhh_factor * fwhhx));
        y_lo = std::max(cube.min_y, (int)std::floor(peak.y_center - support_fwhh_factor * fwhhy));
        y_hi = std::min(cube.max_y, (int)std::ceil(peak.y_center + support_fwhh_factor * fwhhy));
        z_lo = std::max(cube.min_z, (int)std::floor(peak.z_center - support_fwhh_factor * fwhhz));
        z_hi = std::min(cube.max_z, (int)std::ceil(peak.z_center + support_fwhh_factor * fwhhz));
    }

    if (x_lo > x_hi || y_lo > y_hi || z_lo > z_hi) {
        return;
    }

    std::vector<float> vx((size_t)(x_hi - x_lo + 1));
    std::vector<float> vy((size_t)(y_hi - y_lo + 1));
    std::vector<float> vz((size_t)(z_hi - z_lo + 1));

    for (int x = x_lo; x <= x_hi; ++x) {
        vx[(size_t)(x - x_lo)] = eval_axis_voigt_value((float)x, peak.x_center, peak.x_width, peak.x_lfrac, sqrt2ln2);
    }
    for (int y = y_lo; y <= y_hi; ++y) {
        vy[(size_t)(y - y_lo)] = eval_axis_voigt_value((float)y, peak.y_center, peak.y_width, peak.y_lfrac, sqrt2ln2);
    }
    for (int z = z_lo; z <= z_hi; ++z) {
        vz[(size_t)(z - z_lo)] = eval_axis_voigt_value((float)z, peak.z_center, peak.z_width, peak.z_lfrac, sqrt2ln2);
    }

    for (int x = x_lo; x <= x_hi; ++x) {
        float v_x = vx[(size_t)(x - x_lo)];
        for (int y = y_lo; y <= y_hi; ++y) {
            float v_xy = v_x * vy[(size_t)(y - y_lo)];
            for (int z = z_lo; z <= z_hi; ++z) {
                int sparse_idx = cube.grid_to_sparse_idx[cube_flat_index(cube, x, y, z)];
                if (sparse_idx < 0) {
                    continue;
                }
                model[(size_t)sparse_idx] += peak.amplitude * v_xy * vz[(size_t)(z - z_lo)];
            }
        }
    }
}

void calculate_total_3d_model_sparse_with_support(std::vector<float>& model,
                                                   const std::vector<Peak3D>& peaks,
                                                   const std::vector<int>& x_coords,
                                                   const std::vector<int>& y_coords,
                                                   const std::vector<int>& z_coords,
                                                   float support_fwhh_factor)
{
    model.assign(x_coords.size(), 0.0f);

    if (x_coords.empty() || peaks.empty()) {
        return;
    }

    SparseCubeIndex3D cube = build_sparse_cube_index(x_coords, y_coords, z_coords);
    float ln2 = std::log(2.0f);
    float sqrt2ln2 = std::sqrt(2.0f * ln2);

    for (const auto& peak : peaks) {
        accumulate_peak_model_sparse(model, peak, cube, sqrt2ln2, support_fwhh_factor);
    }
}

float calculate_3d_peak_model(float x, float y, float z, const Peak3D& peak)
{
    float ln2 = std::log(2.0f);
    float sqrt2ln2 = std::sqrt(2.0f * ln2);

    auto eval_axis_value = [sqrt2ln2](float coord, float center, float fwhh, float lfrac) {
        float f = std::max(fwhh, 1e-6f);
        float sig = f / (2.0f * sqrt2ln2);
        float gam = f / 2.0f;
        float d = coord - center;
        float d2 = d * d;
        float G = std::exp(-d2 / (2.0f * sig * sig));
        float L = (gam * gam) / (d2 + gam * gam);
        float lf = std::min(1.0f, std::max(0.0f, lfrac));
        return (1.0f - lf) * G + lf * L;
    };

    float Vx = eval_axis_value(x, peak.x_center, peak.x_width, peak.x_lfrac);
    float Vy = eval_axis_value(y, peak.y_center, peak.y_width, peak.y_lfrac);
    float Vz = eval_axis_value(z, peak.z_center, peak.z_width, peak.z_lfrac);

    return peak.amplitude * Vx * Vy * Vz;
}

void calculate_total_3d_model_sparse(std::vector<float>& model,
                                     const std::vector<Peak3D>& peaks,
                                     const std::vector<int>& x_coords,
                                     const std::vector<int>& y_coords,
                                     const std::vector<int>& z_coords)
{
    calculate_total_3d_model_sparse_with_support(model, peaks, x_coords, y_coords, z_coords, FwhhFacctor);
}

void calculate_data_jacobian(
    std::vector<float>& jacobian,
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    const std::vector<float>& model,
    const std::vector<float>& residuals,
    float alpha)
{
    size_t N = partition_data.size();
    size_t M = peaks.size();

    float ln2 = std::log(2.0f);
    float sqrt2ln2 = std::sqrt(2.0f * ln2);

    jacobian.assign((N + 1) * kPeakParamCount * M, 0.0f);

    if (N == 0 || M == 0) {
        return;
    }

    SparseCubeIndex3D cube = build_sparse_cube_index(x_coords, y_coords, z_coords);

    for (size_t p = 0; p < M; ++p) {
        const Peak3D& peak = peaks[p];
        size_t col_offset = p * kPeakParamCount;

        float fwhhx = std::max(peak.x_width, 1e-6f);
        float fwhhy = std::max(peak.y_width, 1e-6f);
        float fwhhz = std::max(peak.z_width, 1e-6f);

        int x_lo = std::max(cube.min_x, (int)std::floor(peak.x_center - FwhhFacctor * fwhhx));
        int x_hi = std::min(cube.max_x, (int)std::ceil(peak.x_center + FwhhFacctor * fwhhx));
        int y_lo = std::max(cube.min_y, (int)std::floor(peak.y_center - FwhhFacctor * fwhhy));
        int y_hi = std::min(cube.max_y, (int)std::ceil(peak.y_center + FwhhFacctor * fwhhy));
        int z_lo = std::max(cube.min_z, (int)std::floor(peak.z_center - FwhhFacctor * fwhhz));
        int z_hi = std::min(cube.max_z, (int)std::ceil(peak.z_center + FwhhFacctor * fwhhz));

        if (x_lo > x_hi || y_lo > y_hi || z_lo > z_hi) {
            continue;
        }

        std::vector<float> vx((size_t)(x_hi - x_lo + 1));
        std::vector<float> dvx_dc(vx.size());
        std::vector<float> dvx_df(vx.size());
        std::vector<float> dvx_dl(vx.size());

        std::vector<float> vy((size_t)(y_hi - y_lo + 1));
        std::vector<float> dvy_dc(vy.size());
        std::vector<float> dvy_df(vy.size());
        std::vector<float> dvy_dl(vy.size());

        std::vector<float> vz((size_t)(z_hi - z_lo + 1));
        std::vector<float> dvz_dc(vz.size());
        std::vector<float> dvz_df(vz.size());
        std::vector<float> dvz_dl(vz.size());

        for (int x = x_lo; x <= x_hi; ++x) {
            size_t idx = (size_t)(x - x_lo);
            eval_axis_voigt_with_derivatives((float)x, peak.x_center, peak.x_width, peak.x_lfrac, sqrt2ln2,
                                             vx[idx], dvx_dc[idx], dvx_df[idx], dvx_dl[idx]);
        }
        for (int y = y_lo; y <= y_hi; ++y) {
            size_t idx = (size_t)(y - y_lo);
            eval_axis_voigt_with_derivatives((float)y, peak.y_center, peak.y_width, peak.y_lfrac, sqrt2ln2,
                                             vy[idx], dvy_dc[idx], dvy_df[idx], dvy_dl[idx]);
        }
        for (int z = z_lo; z <= z_hi; ++z) {
            size_t idx = (size_t)(z - z_lo);
            eval_axis_voigt_with_derivatives((float)z, peak.z_center, peak.z_width, peak.z_lfrac, sqrt2ln2,
                                             vz[idx], dvz_dc[idx], dvz_df[idx], dvz_dl[idx]);
        }

        for (int x = x_lo; x <= x_hi; ++x) {
            size_t x_idx = (size_t)(x - x_lo);
            float v_x = vx[x_idx];
            float dv_x_center = dvx_dc[x_idx];
            float dv_x_width = dvx_df[x_idx];
            float dv_x_lfrac = dvx_dl[x_idx];
            for (int y = y_lo; y <= y_hi; ++y) {
                size_t y_idx = (size_t)(y - y_lo);
                float v_y = vy[y_idx];
                float dv_y_center = dvy_dc[y_idx];
                float dv_y_width = dvy_df[y_idx];
                float dv_y_lfrac = dvy_dl[y_idx];
                for (int z = z_lo; z <= z_hi; ++z) {
                    int sparse_idx = cube.grid_to_sparse_idx[cube_flat_index(cube, x, y, z)];
                    if (sparse_idx < 0) {
                        continue;
                    }
                    size_t z_idx = (size_t)(z - z_lo);
                    float v_z = vz[z_idx];
                    float dv_z_center = dvz_dc[z_idx];
                    float dv_z_width = dvz_df[z_idx];
                    float dv_z_lfrac = dvz_dl[z_idx];

                    float w_i = (residuals[sparse_idx] > 0) ? alpha : 1.0f;
                    float sqrt_w = std::sqrt(w_i);

                    size_t row_idx = (size_t)sparse_idx * (kPeakParamCount * M);
                    jacobian[row_idx + col_offset + 0] = sqrt_w * (v_x * v_y * v_z);
                    jacobian[row_idx + col_offset + 1] = sqrt_w * peak.amplitude * (v_y * v_z) * dv_x_center;
                    jacobian[row_idx + col_offset + 2] = sqrt_w * peak.amplitude * (v_x * v_z) * dv_y_center;
                    jacobian[row_idx + col_offset + 3] = sqrt_w * peak.amplitude * (v_x * v_y) * dv_z_center;
                    jacobian[row_idx + col_offset + 4] = sqrt_w * peak.amplitude * (v_y * v_z) * dv_x_width;
                    jacobian[row_idx + col_offset + 5] = sqrt_w * peak.amplitude * (v_x * v_z) * dv_y_width;
                    jacobian[row_idx + col_offset + 6] = sqrt_w * peak.amplitude * (v_x * v_y) * dv_z_width;
                    jacobian[row_idx + col_offset + 7] = sqrt_w * peak.amplitude * (v_y * v_z) * dv_x_lfrac;
                    jacobian[row_idx + col_offset + 8] = sqrt_w * peak.amplitude * (v_x * v_z) * dv_y_lfrac;
                    jacobian[row_idx + col_offset + 9] = sqrt_w * peak.amplitude * (v_x * v_y) * dv_z_lfrac;
                }
            }
        }
    }
}

void calculate_regularization_jacobian_numerical(
    std::vector<float>& jac_reg_row,
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    float noise_floor,
    float alpha,
    float lambda)
{
    if (lambda <= 0.0f) {
        jac_reg_row.assign(kPeakParamCount * peaks.size(), 0.0f);
        return;
    }

    size_t M = peaks.size();
    jac_reg_row.assign(kPeakParamCount * M, 0.0f);

    std::vector<float> r_baseline = objective_function_3d(peaks, partition_data, x_coords, y_coords, z_coords,
                                                          noise_floor, alpha, lambda);
    float r_conc_baseline = r_baseline.back();

    float eps = 1e-5f;

    for (size_t p = 0; p < M; p++) {
        for (int param = 0; param < kPeakParamCount; param++) {
            std::vector<Peak3D> peaks_perturbed = peaks;

            switch (param) {
                case 0:  peaks_perturbed[p].amplitude += eps; break;
                case 1:  peaks_perturbed[p].x_center += eps; break;
                case 2:  peaks_perturbed[p].y_center += eps; break;
                case 3:  peaks_perturbed[p].z_center += eps; break;
                case 4:  peaks_perturbed[p].x_width += eps; break;
                case 5:  peaks_perturbed[p].y_width += eps; break;
                case 6:  peaks_perturbed[p].z_width += eps; break;
                case 7:  peaks_perturbed[p].x_lfrac += eps; break;
                case 8:  peaks_perturbed[p].y_lfrac += eps; break;
                case 9:  peaks_perturbed[p].z_lfrac += eps; break;
            }

            std::vector<float> r_perturbed = objective_function_3d(peaks_perturbed, partition_data, x_coords, y_coords, z_coords,
                                                                   noise_floor, alpha, lambda);
            float r_conc_perturbed = r_perturbed.back();

            size_t col_idx = p * kPeakParamCount + param;
            jac_reg_row[col_idx] = (r_conc_perturbed - r_conc_baseline) / eps;
        }
    }
}

std::vector<float> objective_function_3d(
    const std::vector<Peak3D>& peaks,
    const std::vector<float>& partition_data,
    const std::vector<int>& x_coords,
    const std::vector<int>& y_coords,
    const std::vector<int>& z_coords,
    float noise_floor,
    float alpha,
    float lambda)
{
    std::vector<float> model;
    calculate_total_3d_model_sparse(model, peaks, x_coords, y_coords, z_coords);

    std::vector<float> residuals(partition_data.size());
    float max_observed = *std::max_element(partition_data.begin(), partition_data.end());

    for (size_t idx = 0; idx < partition_data.size(); idx++) {
        residuals[idx] = model[idx] - partition_data[idx];
    }

    std::vector<float> weights(partition_data.size());
    for (size_t idx = 0; idx < residuals.size(); idx++) {
        weights[idx] = (residuals[idx] > 0) ? alpha : 1.0f;
    }

    std::vector<float> r_data(partition_data.size());
    for (size_t idx = 0; idx < residuals.size(); idx++) {
        r_data[idx] = std::sqrt(weights[idx]) * residuals[idx];
    }

    float r_conc = 0.0f;

    if (lambda > 0.0f) {
        float threshold = std::max(0.01f * max_observed * 6.0f, noise_floor);
        float q = 1.6f;
        float k = 100.0f;

        std::vector<float> sq_err(partition_data.size());
        for (size_t idx = 0; idx < residuals.size(); idx++) {
            sq_err[idx] = residuals[idx] * residuals[idx];
        }

        std::vector<float> w_mask(partition_data.size());
        for (size_t idx = 0; idx < residuals.size(); idx++) {
            float exponent = -k * (std::abs(residuals[idx]) - threshold);
            w_mask[idx] = 1.0f / (1.0f + std::exp(exponent));
        }

        float S = 0.0f;
        std::vector<float> s_masked(partition_data.size());
        for (size_t idx = 0; idx < partition_data.size(); idx++) {
            s_masked[idx] = sq_err[idx] * w_mask[idx];
            S += s_masked[idx];
        }

        if (S > 1e-16f) {
            std::vector<float> p(partition_data.size());
            for (size_t idx = 0; idx < partition_data.size(); idx++) {
                p[idx] = s_masked[idx] / S;
            }

            float N = (float)partition_data.size();
            float sum_p = 0.0f;
            for (size_t idx = 0; idx < partition_data.size(); idx++) {
                float grid_bias = -1.0f + 2.0f * (float)idx / (N - 1.0f);
                p[idx] = p[idx] * (1.0f + 1e-6f * grid_bias);
                sum_p += p[idx];
            }

            if (sum_p > 1e-16f) {
                for (size_t idx = 0; idx < p.size(); idx++) {
                    p[idx] /= sum_p;
                }
            }

            float Cq = 0.0f;
            for (float prob : p) {
                Cq += std::pow(prob, q);
            }

            float conc_penalty = 1.0f - Cq;
            r_conc = std::sqrt(lambda) * conc_penalty;
        }
    }

    std::vector<float> r_total(r_data.size() + 1);
    for (size_t idx = 0; idx < r_data.size(); idx++) {
        r_total[idx] = r_data[idx];
    }
    r_total[r_data.size()] = r_conc;

    return r_total;
}

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
    float lambda)
{
    size_t N = partition_data.size();
    size_t M = peaks.size();
    size_t n_params = kPeakParamCount * M;

    std::vector<float> r = objective_function_3d(peaks, partition_data, x_coords, y_coords, z_coords,
                                                 noise_floor, alpha, lambda);
    residuals = r;

    if (jacobian != nullptr) {
        jacobian->assign((N + 1) * n_params, 0.0f);

        std::vector<float> model;
        calculate_total_3d_model_sparse(model, peaks, x_coords, y_coords, z_coords);

        std::vector<float> residuals_data(N);
        for (size_t idx = 0; idx < N; idx++) {
            residuals_data[idx] = model[idx] - partition_data[idx];
        }

        calculate_data_jacobian(*jacobian, peaks, partition_data, x_coords, y_coords, z_coords,
                                model, residuals_data, alpha);

        std::vector<float> jac_reg_row;
        calculate_regularization_jacobian_numerical(jac_reg_row, peaks, partition_data, x_coords, y_coords, z_coords,
                                                    noise_floor, alpha, lambda);

        size_t reg_row_start = N * n_params;
        for (size_t col = 0; col < n_params; col++) {
            (*jacobian)[reg_row_start + col] = jac_reg_row[col];
        }
    }
}

bool Peak3DNoRegCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobian) const
{
    int n_peaks = n_params / kPeakParamCount;
    std::vector<Peak3D> peaks(n_peaks);

    for (int p = 0; p < n_peaks; p++) {
        peaks[p].amplitude = (float)*parameters[p * kPeakParamCount + 0];
        peaks[p].x_center = (float)*parameters[p * kPeakParamCount + 1];
        peaks[p].y_center = (float)*parameters[p * kPeakParamCount + 2];
        peaks[p].z_center = (float)*parameters[p * kPeakParamCount + 3];
        peaks[p].x_width = (float)*parameters[p * kPeakParamCount + 4];
        peaks[p].y_width = (float)*parameters[p * kPeakParamCount + 5];
        peaks[p].z_width = (float)*parameters[p * kPeakParamCount + 6];
        peaks[p].x_lfrac = (float)*parameters[p * kPeakParamCount + 7];
        peaks[p].y_lfrac = (float)*parameters[p * kPeakParamCount + 8];
        peaks[p].z_lfrac = (float)*parameters[p * kPeakParamCount + 9];
    }

    std::vector<float> model;
    calculate_total_3d_model_sparse(model, peaks, *x_coords, *y_coords, *z_coords);

    for (size_t idx = 0; idx < data->size(); idx++) {
        residuals[idx] = (double)(model[idx] - (*data)[idx]);
    }

    if (jacobian != nullptr) {
        size_t N = data->size();
        float ln2 = std::log(2.0f);
        float sqrt2ln2 = std::sqrt(2.0f * ln2);

        for (int c = 0; c < n_params; ++c) {
            if (jacobian[c] != nullptr) {
                std::fill(jacobian[c], jacobian[c] + N, 0.0);
            }
        }

        SparseCubeIndex3D cube = build_sparse_cube_index(*x_coords, *y_coords, *z_coords);

        for (int p = 0; p < n_peaks; ++p) {
            const Peak3D& peak = peaks[p];
            int col_offset = p * kPeakParamCount;

            float fwhhx = std::max(peak.x_width, 1e-6f);
            float fwhhy = std::max(peak.y_width, 1e-6f);
            float fwhhz = std::max(peak.z_width, 1e-6f);

            int x_lo = std::max(cube.min_x, (int)std::floor(peak.x_center - FwhhFacctor * fwhhx));
            int x_hi = std::min(cube.max_x, (int)std::ceil(peak.x_center + FwhhFacctor * fwhhx));
            int y_lo = std::max(cube.min_y, (int)std::floor(peak.y_center - FwhhFacctor * fwhhy));
            int y_hi = std::min(cube.max_y, (int)std::ceil(peak.y_center + FwhhFacctor * fwhhy));
            int z_lo = std::max(cube.min_z, (int)std::floor(peak.z_center - FwhhFacctor * fwhhz));
            int z_hi = std::min(cube.max_z, (int)std::ceil(peak.z_center + FwhhFacctor * fwhhz));

            if (x_lo > x_hi || y_lo > y_hi || z_lo > z_hi) {
                continue;
            }

            std::vector<float> vx((size_t)(x_hi - x_lo + 1));
            std::vector<float> dvx_dc(vx.size());
            std::vector<float> dvx_df(vx.size());
            std::vector<float> dvx_dl(vx.size());

            std::vector<float> vy((size_t)(y_hi - y_lo + 1));
            std::vector<float> dvy_dc(vy.size());
            std::vector<float> dvy_df(vy.size());
            std::vector<float> dvy_dl(vy.size());

            std::vector<float> vz((size_t)(z_hi - z_lo + 1));
            std::vector<float> dvz_dc(vz.size());
            std::vector<float> dvz_df(vz.size());
            std::vector<float> dvz_dl(vz.size());

            for (int x = x_lo; x <= x_hi; ++x) {
                size_t idx = (size_t)(x - x_lo);
                eval_axis_voigt_with_derivatives((float)x, peak.x_center, peak.x_width, peak.x_lfrac, sqrt2ln2,
                                                 vx[idx], dvx_dc[idx], dvx_df[idx], dvx_dl[idx]);
            }
            for (int y = y_lo; y <= y_hi; ++y) {
                size_t idx = (size_t)(y - y_lo);
                eval_axis_voigt_with_derivatives((float)y, peak.y_center, peak.y_width, peak.y_lfrac, sqrt2ln2,
                                                 vy[idx], dvy_dc[idx], dvy_df[idx], dvy_dl[idx]);
            }
            for (int z = z_lo; z <= z_hi; ++z) {
                size_t idx = (size_t)(z - z_lo);
                eval_axis_voigt_with_derivatives((float)z, peak.z_center, peak.z_width, peak.z_lfrac, sqrt2ln2,
                                                 vz[idx], dvz_dc[idx], dvz_df[idx], dvz_dl[idx]);
            }

            for (int x = x_lo; x <= x_hi; ++x) {
                size_t x_idx = (size_t)(x - x_lo);
                float v_x = vx[x_idx];
                float dv_x_center = dvx_dc[x_idx];
                float dv_x_width = dvx_df[x_idx];
                float dv_x_lfrac = dvx_dl[x_idx];
                for (int y = y_lo; y <= y_hi; ++y) {
                    size_t y_idx = (size_t)(y - y_lo);
                    float v_y = vy[y_idx];
                    float dv_y_center = dvy_dc[y_idx];
                    float dv_y_width = dvy_df[y_idx];
                    float dv_y_lfrac = dvy_dl[y_idx];
                    for (int z = z_lo; z <= z_hi; ++z) {
                        int sparse_idx = cube.grid_to_sparse_idx[cube_flat_index(cube, x, y, z)];
                        if (sparse_idx < 0) {
                            continue;
                        }
                        size_t z_idx = (size_t)(z - z_lo);
                        float v_z = vz[z_idx];
                        float dv_z_center = dvz_dc[z_idx];
                        float dv_z_width = dvz_df[z_idx];
                        float dv_z_lfrac = dvz_dl[z_idx];

                        if (jacobian[col_offset + 0] != nullptr) jacobian[col_offset + 0][sparse_idx] = (double)(v_x * v_y * v_z);
                        if (jacobian[col_offset + 1] != nullptr) jacobian[col_offset + 1][sparse_idx] = (double)(peak.amplitude * (v_y * v_z) * dv_x_center);
                        if (jacobian[col_offset + 2] != nullptr) jacobian[col_offset + 2][sparse_idx] = (double)(peak.amplitude * (v_x * v_z) * dv_y_center);
                        if (jacobian[col_offset + 3] != nullptr) jacobian[col_offset + 3][sparse_idx] = (double)(peak.amplitude * (v_x * v_y) * dv_z_center);
                        if (jacobian[col_offset + 4] != nullptr) jacobian[col_offset + 4][sparse_idx] = (double)(peak.amplitude * (v_y * v_z) * dv_x_width);
                        if (jacobian[col_offset + 5] != nullptr) jacobian[col_offset + 5][sparse_idx] = (double)(peak.amplitude * (v_x * v_z) * dv_y_width);
                        if (jacobian[col_offset + 6] != nullptr) jacobian[col_offset + 6][sparse_idx] = (double)(peak.amplitude * (v_x * v_y) * dv_z_width);
                        if (jacobian[col_offset + 7] != nullptr) jacobian[col_offset + 7][sparse_idx] = (double)(peak.amplitude * (v_y * v_z) * dv_x_lfrac);
                        if (jacobian[col_offset + 8] != nullptr) jacobian[col_offset + 8][sparse_idx] = (double)(peak.amplitude * (v_x * v_z) * dv_y_lfrac);
                        if (jacobian[col_offset + 9] != nullptr) jacobian[col_offset + 9][sparse_idx] = (double)(peak.amplitude * (v_x * v_y) * dv_z_lfrac);
                    }
                }
            }
        }
    }

    return true;
}

bool SinglePeak3DCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobian) const
{
    Peak3D peak;
    peak.amplitude = (float)*parameters[0];
    peak.x_center = (float)*parameters[1];
    peak.y_center = (float)*parameters[2];
    peak.z_center = (float)*parameters[3];
    peak.x_width = (float)*parameters[4];
    peak.y_width = (float)*parameters[5];
    peak.z_width = (float)*parameters[6];
    peak.x_lfrac = (float)*parameters[7];
    peak.y_lfrac = (float)*parameters[8];
    peak.z_lfrac = (float)*parameters[9];

    std::vector<float> res_vec;
    std::vector<float>* jac_ptr = (jacobian != nullptr) ? new std::vector<float>() : nullptr;

    objective_function_3d_mixed_jacobian(res_vec, jac_ptr, {peak}, *data, *x_coords, *y_coords, *z_coords,
                                         noise_floor, alpha, lambda);

    if (residuals) {
        for (size_t i = 0; i < res_vec.size(); i++) {
            residuals[i] = (double)res_vec[i];
        }
    }

    if (jacobian && jac_ptr) {
        size_t N_res = res_vec.size();
        size_t N_par = 10;
        for (size_t i = 0; i < N_res; i++) {
            for (size_t j = 0; j < N_par; j++) {
                jacobian[j][i] = (double)(*jac_ptr)[i * N_par + j];
            }
        }
        delete jac_ptr;
    }

    return true;
}

bool Peak3DCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobian) const
{
    std::vector<Peak3D> peaks;

    int n_peaks = n_params / kPeakParamCount;
    for (int p = 0; p < n_peaks; p++) {
        Peak3D peak;
        peak.amplitude = (float)*parameters[p * kPeakParamCount + 0];
        peak.x_center = (float)*parameters[p * kPeakParamCount + 1];
        peak.y_center = (float)*parameters[p * kPeakParamCount + 2];
        peak.z_center = (float)*parameters[p * kPeakParamCount + 3];
        peak.x_width = (float)*parameters[p * kPeakParamCount + 4];
        peak.y_width = (float)*parameters[p * kPeakParamCount + 5];
        peak.z_width = (float)*parameters[p * kPeakParamCount + 6];
        peak.x_lfrac = (float)*parameters[p * kPeakParamCount + 7];
        peak.y_lfrac = (float)*parameters[p * kPeakParamCount + 8];
        peak.z_lfrac = (float)*parameters[p * kPeakParamCount + 9];
        peaks.push_back(peak);
    }

    std::vector<float> res;
    std::vector<float>* jac_ptr = (jacobian != nullptr) ? new std::vector<float>() : nullptr;

    objective_function_3d_mixed_jacobian(res, jac_ptr, peaks, *data, *x_coords, *y_coords, *z_coords,
                                         noise_floor, alpha, lambda);

    if (residuals && !res.empty()) {
        std::copy(res.begin(), res.end(), residuals);
    }

    if (jacobian && jac_ptr) {
        int n_residuals = res.size();
        size_t n_params_local = jac_ptr->size() / n_residuals;

        for (int i = 0; i < n_residuals; i++) {
            for (size_t j = 0; j < n_params_local; j++) {
                if (jacobian[j] != nullptr) {
                    jacobian[j][i] = (double)(*jac_ptr)[i * n_params_local + j];
                }
            }
        }
        delete jac_ptr;
    }

    return true;
}
