#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>

constexpr int DATA_POINTS = 1000;
constexpr int TASKS = 8;
constexpr int MAX_ITERS = 100;
constexpr double EPS = 1e-8;
constexpr double LAMBDA_INIT = 1e-3;

using Data1D = std::vector<double>;
using Params = std::vector<double>; // [A, x0, sigma]

struct FitWorkspace {
    Params params = {1.0, 500.0, 40.0}; // Initial guess
    std::vector<double> residuals = std::vector<double>(DATA_POINTS);
    std::vector<std::vector<double>> J = std::vector<std::vector<double>>(DATA_POINTS, std::vector<double>(3));
};

// Generate noisy Gaussian data
Data1D generate_data(double A, double x0, double sigma, std::mt19937& rng) {
    std::normal_distribution<> noise(0.0, 1.0);
    Data1D data(DATA_POINTS);
    for (int i = 0; i < DATA_POINTS; ++i) {
        double x = i;
        data[i] = A * std::exp(-std::pow(x - x0, 2) / (2 * sigma * sigma)) + noise(rng);
    }
    return data;
}

// Compute Gaussian value
double gaussian(double x, const Params& p) {
    double A = p[0], x0 = p[1], sigma = p[2];
    double dx = x - x0;
    return A * std::exp(-(dx * dx) / (2 * sigma * sigma));
}

// Simple LM fit for Gaussian
void fit_gaussian_lm(const Data1D& y_data, FitWorkspace& ws) {
    Params& p = ws.params;
    auto& residuals = ws.residuals;
    auto& J = ws.J;

    double lambda = LAMBDA_INIT;

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        double cost = 0.0;

        // Compute residuals and Jacobian
        for (int i = 0; i < DATA_POINTS; ++i) {
            double x = i;
            double A = p[0], x0 = p[1], sigma = p[2];
            double dx = x - x0;
            double sigma2 = sigma * sigma;
            double exp_term = std::exp(-dx * dx / (2 * sigma2));
            double y_fit = A * exp_term;
            double error = y_fit - y_data[i];
            residuals[i] = error;
            cost += error * error;

            // Jacobian w.r.t A, x0, sigma
            J[i][0] = exp_term;
            J[i][1] = A * exp_term * dx / sigma2;
            J[i][2] = A * exp_term * dx * dx / (sigma * sigma2);
        }

        // Build approximate Hessian and gradient
        double H[3][3] = {}, g[3] = {};

        for (int i = 0; i < DATA_POINTS; ++i) {
            for (int j = 0; j < 3; ++j) {
                g[j] += J[i][j] * residuals[i];
                for (int k = 0; k < 3; ++k) {
                    H[j][k] += J[i][j] * J[i][k];
                }
            }
        }

        // Add damping term (Levenberg modification)
        for (int j = 0; j < 3; ++j) {
            H[j][j] *= (1.0 + lambda);
        }

        // Solve H * delta = -g using Gaussian elimination (since H is 3x3)
        double delta[3];
        double A_mat[3][4]; // augmented matrix

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                A_mat[i][j] = H[i][j];
            A_mat[i][3] = -g[i];
        }

        // Gaussian elimination (basic)
        for (int i = 0; i < 3; ++i) {
            // Normalize row
            double factor = A_mat[i][i];
            for (int j = i; j < 4; ++j)
                A_mat[i][j] /= factor;
            // Eliminate below
            for (int k = i + 1; k < 3; ++k) {
                double f = A_mat[k][i];
                for (int j = i; j < 4; ++j)
                    A_mat[k][j] -= f * A_mat[i][j];
            }
        }

        // Back-substitution
        for (int i = 2; i >= 0; --i) {
            delta[i] = A_mat[i][3];
            for (int j = i + 1; j < 3; ++j)
                delta[i] -= A_mat[i][j] * delta[j];
        }

        // Update parameters
        Params new_p = {p[0] + delta[0], p[1] + delta[1], p[2] + delta[2]};

        // Compute new cost to check if update helped
        double new_cost = 0.0;
        for (int i = 0; i < DATA_POINTS; ++i) {
            double x = i;
            double dx = x - new_p[1];
            double sigma2 = new_p[2] * new_p[2];
            double y_fit = new_p[0] * std::exp(-dx * dx / (2 * sigma2));
            double err = y_fit - y_data[i];
            new_cost += err * err;
        }

        if (new_cost < cost) {
            // Accept update
            p = new_p;
            lambda *= 0.7;
            if (std::abs(cost - new_cost) < EPS) break; // Converged
        } else {
            // Reject and increase lambda
            lambda *= 2.0;
        }
    }
}

int main() {
    std::vector<Data1D> datasets(TASKS);
    std::vector<FitWorkspace> workspaces(TASKS);
    std::vector<Params> fitted_params(TASKS);

    std::mt19937 rng(42);
    for (int i = 0; i < TASKS; ++i) {
        double A = 100.0 + i * 10;
        double x0 = 30.0 + i * 2;
        double sigma = 5.0 + 0.5 * i;
        datasets[i] = generate_data(A, x0, sigma, rng);
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < TASKS; ++i) {
        fit_gaussian_lm(datasets[i], workspaces[i]);
        fitted_params[i] = workspaces[i].params;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Parallel LM fitting done in " << elapsed.count() << " seconds\n";
    for (int i = 0; i < TASKS; ++i) {
        const auto& p = fitted_params[i];
        std::cout << "Fit " << i << ": A=" << p[0] << ", x0=" << p[1] << ", sigma=" << p[2] << "\n";
    }

    return 0;
}
