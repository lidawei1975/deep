#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <omp.h>

constexpr size_t CHUNK_SIZE_MB = 1000; // per task
constexpr size_t CHUNK_SIZE_BYTES = CHUNK_SIZE_MB * 1024 * 1024;
constexpr size_t DOUBLES_PER_CHUNK = CHUNK_SIZE_BYTES / sizeof(double);

void memory_intensive_task(std::vector<double>& data) {
    volatile double sum = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += std::sqrt(data[i]); // memory access + compute
    }
    // Prevent optimization
    if (sum == 0.123456) std::cout << "Impossible!\n";
}

int main() {
    const int tasks = 8;
    std::cout << "OpenMP Memory Bandwidth Scaling Test\n";
    std::cout << "Each task reads " << CHUNK_SIZE_MB << " MB of data\n";

    int max_threads = omp_get_max_threads();

    // Prepare memory blocks for each task
    std::vector<std::vector<double>> memory_blocks(tasks);
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(1.0, 100.0);

    for (auto& block : memory_blocks) {
        block.resize(DOUBLES_PER_CHUNK);
        for (auto& val : block) {
            val = dist(rng); // Fill with random doubles
        }
    }

    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
        omp_set_num_threads(num_threads);

        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (int i = 0; i < tasks; ++i) {
            memory_intensive_task(memory_blocks[i]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Threads: " << num_threads
                  << ", Time: " << elapsed.count() << " seconds\n";
    }

    return 0;
}