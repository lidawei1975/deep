#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

void compute_heavy_task(int iterations) {
    double sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum += std::sin(i) * std::cos(i); // Trig operations are CPU-intensive
    }
    // Prevent compiler optimization
    volatile double sink = sum;
}

int main() {
    const int tasks = 16; // number of tasks (can be more or less)
    const int iterations_per_task = 10'000'000; // adjust for desired workload

    std::cout << "OpenMP Parallel Scaling Test\n";

    int max_threads = omp_get_max_threads();

    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
        omp_set_num_threads(num_threads);

        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < tasks; ++i) {
            compute_heavy_task(iterations_per_task);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Threads: " << num_threads
                  << ", Time: " << elapsed.count() << " seconds\n";
    }

    return 0;
}