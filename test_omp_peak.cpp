#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>

#include "lmminimizer_n.h"
#include "cost_functors.h"

constexpr int GRID_SIZE = 100;
constexpr int TOTAL_SIZE = GRID_SIZE * GRID_SIZE;
constexpr int TASKS = 4;

using Surface1D = std::vector<double>;

// Generate a 1D flattened surface with 1 or 2 Gaussian peaks
Surface1D generate_gaussian_surface(std::mt19937& rng) {
    Surface1D grid(TOTAL_SIZE, 0.0);
    std::uniform_real_distribution<> dist_pos(0, GRID_SIZE - 1);
    std::uniform_real_distribution<> dist_amp(50.0, 200.0);
    std::uniform_real_distribution<> dist_sigma(5.0, 20.0);
    std::uniform_int_distribution<> dist_peaks(1, 2);

    int num_peaks = dist_peaks(rng);
    for (int p = 0; p < num_peaks; ++p) {
        double x0 = dist_pos(rng);
        double y0 = dist_pos(rng);
        double A = dist_amp(rng);
        double sigma = dist_sigma(rng);

        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                double dx = x - x0;
                double dy = y - y0;
                grid[y * GRID_SIZE + x] += A * std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            }
        }
    }

    return grid;
}

void process_surface(const Surface1D& surface, double **par, double **new_par, double **jaco, double *ypre)
    // Simulate some heavy computation
    // This is where you would call your cost function or optimization routine
    // For demonstration, we just sum the surface values
{   
    auto start = std::chrono::high_resolution_clock::now();
    // volatile double sum = 0;
    // for (size_t i = 0; i < surface.size(); ++i) {
    //     sum += std::sin(surface[i]) * std::cos(surface[i]);
    // }
    // if (sum == 123456.0) std::cout << "Unlikely!\n"; // prevent optimization


    par[0][0]=10;
    par[1][0]=GRID_SIZE/2;
    par[2][0]=3;
    par[3][0]=4;
    par[4][0]=GRID_SIZE/2;
    par[5][0]=2;
    par[6][0]=1;


    int xdim=GRID_SIZE;
    int ydim=GRID_SIZE;

    mycostfunction_voigt cost_function(xdim,ydim,surface.data());

    class levmarq minimizer;

    
    minimizer.solve(par,new_par, surface.size(), NULL /**weight*/,jaco, ypre, cost_function);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for surface processing: " << elapsed.count() << " seconds\n";

}

int main() {
    std::cout << "OpenMP Gaussian Surface Test\n";
    std::cout << "Each task reads a 100x100 grid with Gaussian peaks\n";

    int max_threads = omp_get_max_threads();

    // Pre-generate surfaces
    std::vector<Surface1D> surfaces(TASKS);
    std::mt19937 rng(42); // fixed seed

    for (int i = 0; i < TASKS; ++i) {
        surfaces[i] = generate_gaussian_surface(rng);
    }

    int test_number[2] = {4,1};

    for (int i = 0; i < 2; ++i)
    {

        int num_threads = test_number[i];

        omp_set_num_threads(num_threads);

        

        /**
         * Pre allocate memory for the cost function of each thread
         * par is 7 by 1
         * new_par is 7 by 1
         * jaco is 7 by n_datapoint
         * y_pre is n_datapoint
        */
        std::vector<double **> par(TASKS);
        std::vector<double **> new_par(TASKS);
        std::vector<double **> jaco(TASKS);
        std::vector<double *> ypre(TASKS);

        for (int i = 0; i < TASKS; ++i)
        {
            par[i] = new double*[7];
            new_par[i] = new double*[7];
            jaco[i] = new double*[7];
            ypre[i] = new double[GRID_SIZE * GRID_SIZE];

            for (int j = 0; j < 7; ++j) {
                par[i][j] = new double[1];
                new_par[i][j] = new double[1];
                jaco[i][j] = new double[GRID_SIZE * GRID_SIZE];
            }
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for 
        for (int i = 0; i < TASKS; ++i) {
            Surface1D surface = generate_gaussian_surface(rng);
            // const Surface1D& surface = surfaces[i];
            double **current_par = par[i];
            double **current_new_par = new_par[i];
            double **current_jaco = jaco[i];
            double *current_ypre = ypre[i];
            process_surface(surface, current_par, current_new_par, current_jaco, current_ypre); 
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Threads: " << num_threads
                  << ", Time: " << elapsed.count() << " seconds\n";
    }

    return 0;
}