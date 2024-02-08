#include <vector>
#include <string>
#include "json/json.h"
#include "spectrum_io_1d.h"

#ifndef LDW_PARA
#define LDW_PARA
#define SMALL 1e-10
#define PI 3.14159265358979323846
#define M_SQRT_PI 1.772453850905516
#define M_SQRT_2PI 2.506628274631000
#define M_1_SQRT_PI 0.564189583547756
#endif

#ifndef DB_MATCH_GLOBAL_1D_H
#define DB_MATCH_GLOBAL_1D_H

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


struct Functor_shared_static_variables
{
    static double pos_restrain_strength;
    static double relative_position_restrain_strength;
};


struct Voigt_functor_shape_base: public Functor_shared_static_variables, public ceres::CostFunction
{
    std::vector<double> height;
    std::vector<double> original_ppm;

    std::vector<double> original_sigma;
    std::vector<double> original_gamma; 

    std::vector<double> spectrum_tofit;
    std::vector<double> spectrum_tofit_ppm;
    std::vector<double> spectrum_weight;

    /**
     * The upper limit of the spectrum. Fitted spectrum can not be larger than it at any point
    */
    std::vector<double> spectrum_max; 

    int n_datapoint; //for convenience = spectrum_tofit.size()
    int np; //for convenience = original_ppm.size()

    void voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const;
    void voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const;

    int nresiduals;

    Voigt_functor_shape_base();
    ~Voigt_functor_shape_base();

    inline std::vector<int> *parameter_block_sizes() { return mutable_parameter_block_sizes(); };
    inline void set_n_residuals(int n) { set_num_residuals(n); };
};


/**
 * @brief peak fitting functor for ceres, fit npeak*3+1 paramters: peak postional, sigma, gamma
 * height (concentration) rescale factor of all peaks.
 * This one has weight for each data point.
*/
struct Voigt_functor_shape_w: public Voigt_functor_shape_base
{
   
    Voigt_functor_shape_w(std::vector<double> height,  std::vector<double> ppm,  std::vector<double> sigma,  std::vector<double> gamma,
                          std::vector<double> spectrum_, std::vector<double> spectrum_ppm_, std::vector<double> spectrum_weight_,std::vector<double> v_total_);
    ~Voigt_functor_shape_w();


    bool operator()(double const* const* parameters, double* residuals) const; //numerical jacobian
    bool Evaluate(double const *const *, double *, double **) const; //analytical jacobian

};

/**
 * @brief peak fitting functor for ceres, fit npeak*3+2 paramters: peak postional, sigma, gamma. 
 * height (concentration) rescale factor of all peaks.
 * width rescale factor of all peaks.
 * This one has weight for each data point.
*/
struct Voigt_functor_shape_width: public Voigt_functor_shape_base
{
    Voigt_functor_shape_width(std::vector<double> height,  std::vector<double> ppm,  std::vector<double> sigma,  std::vector<double> gamma,
                          std::vector<double> spectrum_, std::vector<double> spectrum_ppm_, std::vector<double> spectrum_weight_,std::vector<double> v_total_);
    ~Voigt_functor_shape_width();

    bool operator()(double const* const* parameters, double* residuals) const; //numerical jacobian
    bool Evaluate(double const *const *, double *, double **) const; //analytical jacobian

};

/**
 * @brief peak fitting functor for ceres, positional restraint and relative position restraint parts
*/
struct pos_restrain_functor: public Functor_shared_static_variables
{
    std::vector<int> absolute_position_restrain_peaks;
    std::vector<double> absolute_position_restrain_position;
    std::vector<int> relative_position_restrain_peaks_1;
    std::vector<int> relative_position_restrain_peaks_2;
    std::vector<double> relative_position_restrain_distance;


    pos_restrain_functor(std::vector<int>,std::vector<double>,std::vector<int>,std::vector<int>,std::vector<double>);
    bool operator()(double const* const* parameters, double* residuals) const;
};


/**
 * @brief static shared data for all matched compounds and class db_match_global_1d
*/
struct shared_data_global_1d
{
    static int n_verbose_global_1d;
    static double ref_correction_global_1d;
    static bool b_allow_overal_width_rescale_factor;
    static double sigma_gamma_upper_bound;
    static double sigma_gamma_lower_bound;
    static double match_width_factor_upper_bound;
    static double match_width_factor_lower_bound;
}; 


class one_matched_compound : public shared_data_global_1d
{
protected:

    bool b_analytical_jacobian; // if true, use analytical jacobian, otherwise use numerical jacobian

    Json::Value self_json; // json object for this matched compound

    /**
     * @brief varibles for matched compound
     * loaded in load_matched_compound() from json file
     */
    int npeak;
    std::string base_name;
    /**
     * These are from database (spin simulation, followed by deconvolution to Voigt peaks).
     * database_amplitude is the height of the peak while database_intensity is the total area of the peak (concentration)
    */
    std::vector<double> database_ppm,database_intensity,database_sigma,database_gamma,database_amplitude;
    std::vector<int> match_type;
    std::vector<int> match;
    std::vector<int> peak_groups;
    std::vector<int> peak_group_normal_peaks;
    double v_fitted;

    /**
     * @brief varibles for matched compound that will be optimized.
     * Not all of them are used in optimization, depending on the optimization mode
     */
    std::vector<double> match_ppm;
    std::vector<double> match_sigma,match_gamma;
    std::vector<double> peak_distortion_factor;
    double match_height_factor,match_width_factor;

    double upper_bound_match_height_factor;

    /**
     * @brief Initial value before optimization
     */
    std::vector<double> original_ppm;
    
    /**
     * @brief some not changed varibles
     * read them in load_matched_compound() from json file
     * and write them as is in save_matched_compound() to json file
     */
    std::vector<double> match_confident_level;
    double match_height_factor2;
    double total_mean_confident_level;

    double stop1,step1,begin1; // ppm range of the spectrum. From class db_match_global_1d
    int nspectrum_size; // number of points in the spectrum. From class db_match_global_1d
    int water_index_start,water_index_stop; // index of water region. From class db_match_global_1d
    

    bool voigt_convolution_of_one_peak(int peak_index,int &start,int &stop,std::vector<double> &v,const double cutoff);

    /**
     * Varibles for peak postional restraints.
    */
    std::vector<int> absolute_position_restrain_peaks;
    std::vector<double> absolute_position_restrain_position;
    std::vector<int> relative_position_restrain_peaks_1;
    std::vector<int> relative_position_restrain_peaks_2;
    std::vector<double> relative_position_restrain_distance;
    int n_relative_position_restrain_peaks;

public:

    int compound_index; //for debug. From class db_match_global_1d
    int loop; //for debug. From class db_match_global_1d

    one_matched_compound();
    ~one_matched_compound();
    bool set_ppm_range(double,double,double,int,int,int);
    bool load_matched_compound(Json::Value, double);
    bool save_matched_compound(Json::Value &root);

    /**
     * generate a voigt convolution of the whole matched compound
    */
    bool voigt_convolution(std::vector<double> &v,std::vector<int> &v_index, double cutoff=0.01);

    /**
     * optimize height and width_rescale_factor of matched compound
     * Non-weighted and Weighted versions
    */
    bool optimization(std::vector<double> &v, const std::vector<int> &v_index);
    bool optimization_weighted(std::vector<double> &v, const std::vector<int> &v_index, const std::vector<double> &weights); // optimize height and width_rescale_factor of matched compound, with different weights at different ppm
    
    /**
     * optimize height and width_rescale_factor of matched compound and peak position of each peak
     * Non-weighted and Weighted versions
    */
    bool optimization_v2(std::vector<double> &v, const std::vector<int> &v_index); // optimize peak postional, height and width_rescale_factor of matched compound
    bool optimization_v2_weighted(std::vector<double> &v, const std::vector<int> &v_index,const std::vector<double> &weights); // optimize peak postional, height and width_rescale_factor of matched compound. Weighted version
    
    /**
     * optimize height and width_rescale_factor of matched compound and peak position; dostoration factor of each peak
     * Weighted versions
    */
    bool optimization_v3_weighted(std::vector<double> &v, const std::vector<int> &v_index,const std::vector<double> &weights); // optimize peak postional, height and width_rescale_factor of matched compound. Weighted version
    
    /**
     * optimize height of matched compound and peak position, sigma and gamma of each peak (3*npeak+1 parameters)
     * Weighted versions
    */
    bool compound_optimization(std::vector<double> &v_total, std::vector<double> &v, const std::vector<int> &v_index,const std::vector<double> &weights); // optimize peak postional, height and width_rescale_factor of matched compound. Weighted version
    
    /**
     * apply width_rescale_factor to sigma and gamma of each peak
    */
    bool apply_width_rescale_factor();

    bool print();
    
    const double get_v_fitted() const {return v_fitted;};
    const std::string get_base_name() const {return base_name;};

};

class db_match_global_1d : public spectrum_io_1d, public shared_data_global_1d
{
private:
    std::vector<one_matched_compound> matched_compounds;
    double water_width;
    double db_limit;
    int npeak; // number of peaks in the experiment
    int water_index_start,water_index_stop; // index of water region. 

    // from all matched compounds. Same size as spectrum_io_1d::spect
    std::vector<double> simulated_spectrum; 
    
public:
    db_match_global_1d(double,double); //water_width,db_limit
    ~db_match_global_1d();
    bool ref_correction_spectrum(double ref_correction);
    bool deal_with_water();
    bool load_matched_compounds(std::string,double);
    bool save_matched_compounds(std::string);
    bool save_simulated_spectrum(std::string,bool = false);


    bool optimization(int rmax = 10,bool b_weight = false);
};

#endif