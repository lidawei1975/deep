#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>	
#include <Eigen/SparseQR>


#include "spectrum_io_1d.h"

class spectrum_baseline_1d:public spectrum_io_1d
{
private:
    /**
     * To speed up the calculation, we only calculate the baseline for the  max_calculation_length points.
     * If input spectrum is longer than this, we will do data striding and interpolation after the baseline calculation.
    */
    int max_calculation_length; 
protected:

    std::vector<float> baseline;  //read or calculated baseline

    double a0,b0;
    std::vector<int> segment_type;

    bool calculate_baseline(const std::vector<int> &s,const std::vector<double> &y,std::vector<float> &x,double noise, int flag);
    bool calculate_baseline_solver_1(double a,double b,std::vector<int> &s,std::vector<double> &y,Eigen::VectorXd &solution);
    bool calculate_baseline_solver_2(double a,double b,std::vector<int> &s,std::vector<double> &y,Eigen::VectorXd &solution);

public:

    spectrum_baseline_1d(); //default max_calculation_length=4096
    spectrum_baseline_1d(int); //set max_calculation_length=n
    ~spectrum_baseline_1d();

    /**
     * @brief main function to calculate the baseline.
     * a,b are parameters for the baseline.
     * method: 0 (normal) 1 (sparse) matrix based algorithm
     * fname_baseline: output file name for the baseline.
    */
    bool work(double a0_,double b0_,int n_water,int method,std::string fname_baseline);

    /**
     * @brief read baseline from file instead of calculating it.
     * fname: input file name for the baseline.
    */
    bool read_baseline(std::string fname);

    /**
     * Get a read-only reference to the baseline.
    */
    const std::vector<float> &get_baseline() const {return baseline;};

};