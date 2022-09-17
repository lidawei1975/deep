#include <deque>
#include <cmath>

#ifndef LDW_MATH
#define LDW_MATH

namespace ldw_math
{
    void sortArr(std::vector<double> &arr, std::vector<int> &ndx);
    double calcualte_median(std::vector<double> scores);
    int calcualte_median_int(std::vector<int> scores);
    std::vector<std::deque<int> > bread_first(int *neighbor, int n, int n_min_size=2);
    bool linreg(const std::vector<double> x, const std::vector<double> y, double &a, double &b, double &r);
    bool random_sampling_consensus(std::vector<double> x, std::vector<double> y, int max_round, double error_cutoff, int init_number, int n_inline_cutoff,double &a_best,double &b_best,double &r_best);
}

#endif