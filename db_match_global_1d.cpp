
// #include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <set>

#include <time.h>
#include <sys/time.h>



#include "commandline.h"
#include "hungary.h"

#include "db_match_global_1d.h"


double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

//will link to the C function in libcerf
extern "C"  
{
    double voigt(double x, double sigma, double gamma);
    void re_im_w_of_z(double x, double y, double *r, double *i); // re_im_w_of_z
};

/**
 * Program static vaiables
*/
double Functor_shared_static_variables::pos_restrain_strength=1.0;
double Functor_shared_static_variables::relative_position_restrain_strength=1000.0;


int shared_data_global_1d::n_verbose_global_1d = 0;
bool shared_data_global_1d::b_allow_overal_width_rescale_factor = false;
double shared_data_global_1d::ref_correction_global_1d = 0.0;
double shared_data_global_1d::sigma_gamma_lower_bound = 0.8;
double shared_data_global_1d::sigma_gamma_upper_bound = 1.25;
double shared_data_global_1d::match_width_factor_upper_bound = 6.0;
double shared_data_global_1d::match_width_factor_lower_bound = 0.25;

/**
 * Base class for all functors. 
*/
Voigt_functor_shape_base::Voigt_functor_shape_base()
{

};

Voigt_functor_shape_base::~Voigt_functor_shape_base()
{

};


void Voigt_functor_shape_base::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;
    double sigma3 = sigma * sigma2;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);

    double t1 = z_i * l - z_r * v;
    double t2 = z_r * l + z_i * v;

    *r_x0 = t1 / (sigma2 * M_SQRT_PI);
    *r_gamma = (t2 - M_1_SQRT_PI) / (sigma2 * M_SQRT_PI);
    *r_sigma = -v / M_SQRT_2PI / sigma2 - t1 * x0 / sigma3 / M_SQRT_PI - (t2 - M_1_SQRT_PI) * gamma / sigma3 / M_SQRT_PI;
    return;
};

void Voigt_functor_shape_base::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);
    return;
};

/**
 * voigt functor weighted with shape fitting
 * All input paramters are for DB peaks, not for fitting
 * 
*/ 
Voigt_functor_shape_w::Voigt_functor_shape_w(
    std::vector<double> height_,
    std::vector<double> ppm_,
    std::vector<double> sigma_,
    std::vector<double> gamma_,
    std::vector<double> spectrum_,
    std::vector<double> spectrum_ppm_,
    std::vector<double> spectrum_weight_,
    std::vector<double> v_total_)
{
    height = height_;
    original_ppm=ppm_;
    original_sigma=sigma_;
    original_gamma=gamma_;
    spectrum_tofit=spectrum_;
    spectrum_tofit_ppm=spectrum_ppm_;
    spectrum_weight=spectrum_weight_;
    spectrum_max=v_total_;

    n_datapoint = spectrum_tofit.size();
    np = original_ppm.size();

    return;
};

Voigt_functor_shape_w::~Voigt_functor_shape_w()
{
};


/**
 * This is the key function of the functor. It will be called by ceres to calcuate the residuals and jacobians when using analytical derivatives
*/
bool Voigt_functor_shape_w::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double aa=xx[0][0]; //aa is overall height factor
    double vvx, r_x, r_sigmax, r_gammax;
    // voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = -spectrum_tofit[i];
            /**
             * initialize jaco[0] to 0. Because jaco[0][0:n_datapoint] is for aa, which is the sum of contributions from all the peaks
             * other jaco do not need to be initialized because they will be calculated below, each element is from one peak only.
            */
            jaco[0][i] =0 ;
        }
        /**
         * Loop through all the peaks in the compound. np is the number of peaks in the compound
        */
        for (int m = 0; m < np; m++)
        {
            double a = aa*height[m];
            double x0 = xx[1][m];
            double sigmax = fabs(xx[2][m]);
            double gammax = fabs(xx[3][m]);
            for (int i = 0; i < n_datapoint; i++)
            {
                voigt_helper(spectrum_tofit_ppm[i] - x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
                residual[i] += a * vvx;

                /**
                 * jaco[0] is for xx[0] (overall height) size is n_datapoint*1
                 * jaco[1] is for xx[1] (peak centers), size is n_datapoint*np, 
                 *  organized as 
                 * jaco[1]0]=d_spectrum_tofit[0]/d_peak_center[0], jaco[1][1]=d_spectrum_tofit[0]/d_peak_center[1], etc
                 * jaco[1][npeak+0] = d_spectrum_tofit[1]/d_peak_center[0], jaco[1][npeak+1] = d_spectrum_tofit[1]/d_peak_center[1], etc
                 * 
                 * jaco[2] is for xx[2] (peak sigma), size is n_datapoint*np
                 * jaco[3] is for xx[3] (peak gamma), size is n_datapoint*np
                */
                jaco[0][i] += vvx * height[m];       // with respect to aa. Sum of contributions from all the peaks
                jaco[1][i * np + m] = -a * r_x;     // with respect to peak center
                jaco[2][i * np + m] = a * r_sigmax; // with respect to sigmax
                jaco[3][i * np + m] = a * r_gammax; // with respect to gammax
            }
        }
        /**
         * apply weights to the residual and jacobi
        */
        for(int i=0;i<n_datapoint;i++)
        {
            residual[i] = residual[i]*spectrum_weight[i];
            jaco[0][i] = jaco[0][i] * spectrum_weight[i];
            for (int m = 0; m < np; m++)
            {
                jaco[1][i * np + m] = jaco[1][i * np + m] * spectrum_weight[i];
                jaco[2][i * np + m] = jaco[2][i * np + m] * spectrum_weight[i];
                jaco[3][i * np + m] = jaco[3][i * np + m] * spectrum_weight[i];
            }
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = -spectrum_tofit[i];
        }
        for (int m = 0; m < np; m++)
        {
            /** NOTE: voigt function is not normalized, a*height[i] is acutally the total area of the peak
             * need to devided by voigt(0,sigma,gamma) to get the height of the peak
             * parameters[1][i] is the fitted ppm of the i-th peak
             * parameters[2][i] is the fitted sigma
             * parameters[3][i] is the fitted gamma
            */ 
            double a=aa*height[m]; 
            double x0 = xx[1][m];
            double sigmax = fabs(xx[2][m]);
            double gammax = fabs(xx[3][m]);
            for (int i = 0; i < n_datapoint; i++)
            {
                voigt_helper(spectrum_tofit_ppm[i] - x0, sigmax, gammax, &vvx);
                residual[i] += a * vvx;
            }
        }
        /**
         * apply weights to the residual
        */
        for(int i=0;i<n_datapoint;i++)
        {
            residual[i] = residual[i]*spectrum_weight[i];
        }
    }
    return true;
};

   

/**
 * @brief operator() is the main function of the functor for numerical derivative. It will be called by ceres to calcuate the residuals
*/
bool Voigt_functor_shape_w::operator()(double const* const* parameters, double* residuals) const
{
    int n = original_ppm.size();

    /**
     * a is the fitting parameter: height factor
    */
    double a = parameters[0][0];

    for(int j=0;j<spectrum_tofit.size();j++)
    {
        residuals[j] = 0;
    }

    std::vector<double> predicted_spectrum(spectrum_tofit.size(),0.0);

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<spectrum_tofit.size();j++)
        {
            /**
             * NOTE: voigt function is not normalized, a*height[i] is acutally the total area of the peak
             * need to devided by voigt(0,sigma,gamma) to get the height of the peak
             * parameters[1][i] is the fitted ppm of the i-th peak
             * parameters[2][i] is the fitted sigma
             * parameters[3][i] is the fitted gamma
             * 
             * voigt function call will treat negative sigma or gamma as positive.
            */
            predicted_spectrum[j] += a*height[i]*voigt(parameters[1][i]-spectrum_tofit_ppm[j],parameters[2][i],parameters[3][i]);
        }
    }

    /**
     * spectral fitting part residuals = (spectrum_tofit - residuals)*spectrum_weight
    */
    for(int j=0;j<spectrum_tofit.size();j++)
    {
        residuals[j] = (predicted_spectrum[j]-spectrum_tofit[j])*spectrum_weight[j];
    }

    return true;
}


/**
 * @brief pos_restrain_functor
*/
pos_restrain_functor::pos_restrain_functor(
    std::vector<int> absolute_position_restrain_peaks_,
    std::vector<double> absolute_position_restrain_position_,
    std::vector<int> relative_position_restrain_peaks_1_,
    std::vector<int> relative_position_restrain_peaks_2_,
    std::vector<double> relative_position_restrain_distance_)
{
    absolute_position_restrain_peaks = absolute_position_restrain_peaks_;
    absolute_position_restrain_position = absolute_position_restrain_position_;
    relative_position_restrain_peaks_1 = relative_position_restrain_peaks_1_;
    relative_position_restrain_peaks_2 = relative_position_restrain_peaks_2_;
    relative_position_restrain_distance = relative_position_restrain_distance_;
    return;
};

bool pos_restrain_functor::operator()(double const* const* parameters, double* residuals) const
{
    int n = absolute_position_restrain_peaks.size();
     /**
     * postional restrain part residuals
    */
    for(int j=0;j<n;j++)
    {
        int n=absolute_position_restrain_peaks[j];
        residuals[j] = pos_restrain_strength*(parameters[0][n]-absolute_position_restrain_position[j]);
    }

    /**
     * relative positioanl restrain part residuals
    */
    for(int j=0;j<relative_position_restrain_distance.size();j++)
    {
        int j1 = relative_position_restrain_peaks_1[j];
        int j2 = relative_position_restrain_peaks_2[j];
        residuals[n+j] = relative_position_restrain_strength*(fabs(parameters[0][j1]-parameters[0][j2])-relative_position_restrain_distance[j]);
    }

    return true;
};


/**
 * voigt functor weighted with shape fitting
 * All input paramters are for DB peaks, not for fitting
 * 
*/ 
Voigt_functor_shape_width_w::Voigt_functor_shape_width_w(
    std::vector<double> height_,
    std::vector<double> ppm_,
    std::vector<double> sigma_,
    std::vector<double> gamma_,
    std::vector<double> spectrum_,
    std::vector<double> spectrum_ppm_,
    std::vector<double> spectrum_weight_,
    std::vector<double> v_total_)
{
    height = height_;
    original_ppm=ppm_;
    original_sigma=sigma_;
    original_gamma=gamma_;
    spectrum_tofit=spectrum_;
    spectrum_tofit_ppm=spectrum_ppm_;
    spectrum_weight=spectrum_weight_;
    spectrum_max=v_total_;

    n_datapoint = spectrum_tofit.size();
    np = original_ppm.size();

    return;
};

Voigt_functor_shape_width_w::~Voigt_functor_shape_width_w()
{
};


/**
 * This is the key function of the functor. It will be called by ceres to calcuate the residuals and jacobians when using analytical derivatives
*/
bool Voigt_functor_shape_width_w::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double aa=xx[0][0]; //aa is overall height factor
    double ww=xx[4][0]; //ww is overall width factor
    double vvx, r_x, r_sigmax, r_gammax;
    // voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = -spectrum_tofit[i];
            /**
             * initialize jaco[0] to 0. Because jaco[0][0:n_datapoint] is for aa, which is the sum of contributions from all the peaks
             * other jaco do not need to be initialized because they will be calculated below, each element is from one peak only.
            */
            jaco[0][i] = 0 ;
            jaco[4][i] = 0 ;
        }
        /**
         * Loop through all the peaks in the compound. np is the number of peaks in the compound
        */
        for (int m = 0; m < np; m++)
        {
            double a = aa*height[m];
            double x0 = xx[1][m];
            double sigmax = fabs(xx[2][m]);
            double gammax = fabs(xx[3][m]);
            for (int i = 0; i < n_datapoint; i++)
            {
                voigt_helper(spectrum_tofit_ppm[i] - x0, sigmax*ww, gammax*ww, &vvx, &r_x, &r_sigmax, &r_gammax);
                residual[i] += a * vvx;

                /**
                 * jaco[0] is for xx[0] (overall height) size is n_datapoint*1
                 * jaco[1] is for xx[1] (peak centers), size is n_datapoint*np, 
                 *  organized as 
                 * jaco[1]0]=d_spectrum_tofit[0]/d_peak_center[0], jaco[1][1]=d_spectrum_tofit[0]/d_peak_center[1], etc
                 * jaco[1][npeak+0] = d_spectrum_tofit[1]/d_peak_center[0], jaco[1][npeak+1] = d_spectrum_tofit[1]/d_peak_center[1], etc
                 * 
                 * jaco[2] is for xx[2] (peak sigma), size is n_datapoint*np
                 * jaco[3] is for xx[3] (peak gamma), size is n_datapoint*np
                */
                jaco[0][i] += vvx * height[m];       // with respect to aa. Sum of contributions from all the peaks
                jaco[1][i * np + m] = -a * r_x;     // with respect to peak center
                jaco[2][i * np + m] = a * r_sigmax * ww; // with respect to sigmax
                jaco[3][i * np + m] = a * r_gammax * ww; // with respect to gammax
                jaco[4][i] += a * (r_gammax*gammax+r_sigmax*sigmax); // with respect to ww
            }
        }
        /**
         * apply weights to the residual and jacobi
        */
        for(int i=0;i<n_datapoint;i++)
        {
            residual[i] = residual[i]*spectrum_weight[i];
            jaco[0][i] = jaco[0][i] * spectrum_weight[i];
            jaco[4][i] = jaco[4][i] * spectrum_weight[i];
            for (int m = 0; m < np; m++)
            {
                jaco[1][i * np + m] = jaco[1][i * np + m] * spectrum_weight[i];
                jaco[2][i * np + m] = jaco[2][i * np + m] * spectrum_weight[i];
                jaco[3][i * np + m] = jaco[3][i * np + m] * spectrum_weight[i];
            }
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = -spectrum_tofit[i];
        }
        for (int m = 0; m < np; m++)
        {
            /** NOTE: voigt function is not normalized, a*height[i] is acutally the total area of the peak
             * need to devided by voigt(0,sigma,gamma) to get the height of the peak
             * parameters[1][i] is the fitted ppm of the i-th peak
             * parameters[2][i] is the fitted sigma
             * parameters[3][i] is the fitted gamma
            */ 
            double a=aa*height[m]; 
            double x0 = xx[1][m];
            double sigmax = fabs(xx[2][m])*ww;
            double gammax = fabs(xx[3][m])*ww;
            for (int i = 0; i < n_datapoint; i++)
            {
                voigt_helper(spectrum_tofit_ppm[i] - x0, sigmax, gammax, &vvx);
                residual[i] += a * vvx;
            }
        }
        /**
         * apply weights to the residual
        */
        for(int i=0;i<n_datapoint;i++)
        {
            residual[i] = residual[i]*spectrum_weight[i];
        }
    }
    return true;
};

   

/**
 * @brief operator() is the main function of the functor for numerical derivative. It will be called by ceres to calcuate the residuals
*/
bool Voigt_functor_shape_width_w::operator()(double const* const* parameters, double* residuals) const
{
    int n = original_ppm.size();

    /**
     * a is the fitting parameter: height factor
    */
    double a = parameters[0][0];

    /**
     * w is the fitting parameter: width factor
    */
    double w = parameters[4][0];

    for(int j=0;j<spectrum_tofit.size();j++)
    {
        residuals[j] = 0;
    }

    std::vector<double> predicted_spectrum(spectrum_tofit.size(),0.0);

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<spectrum_tofit.size();j++)
        {
            /**
             * NOTE: voigt function is not normalized, a*height[i] is acutally the total area of the peak
             * need to devided by voigt(0,sigma,gamma) to get the height of the peak
             * parameters[1][i] is the fitted ppm of the i-th peak
             * parameters[2][i] is the fitted sigma
             * parameters[3][i] is the fitted gamma
             * 
             * voigt function call will treat negative sigma or gamma as positive.
            */
            predicted_spectrum[j] += a*height[i]*voigt(parameters[1][i]-spectrum_tofit_ppm[j],parameters[2][i]*w,parameters[3][i]*w);
        }
    }

    /**
     * spectral fitting part residuals = (spectrum_tofit - residuals)*spectrum_weight
    */
    for(int j=0;j<spectrum_tofit.size();j++)
    {
        residuals[j] = (predicted_spectrum[j]-spectrum_tofit[j])*spectrum_weight[j];
    }

    return true;
}

//one_matched_compound
//constructor and destructor
one_matched_compound::one_matched_compound()
{
    b_analytical_jacobian=true;
    match_height_factor = 1.0;
    match_width_factor = 1.0;
};

one_matched_compound::~one_matched_compound()
{
};

bool one_matched_compound::set_ppm_range(double stop1_,double step1_,double begin1_,int n_,int w1_,int w2_)
{
    stop1 = stop1_;
    step1 = step1_;
    begin1 = begin1_;
    nspectrum_size = n_;
    water_index_start = w1_;
    water_index_stop = w2_;
    return true;
}

bool one_matched_compound::load_matched_compound(Json::Value x,double scale)
{
    /**
     * Save a copy of the json object
    */
    self_json = x;

    /**
     * If we have width-factor in webapp, then scale the width of peaks in the database
     * This is in addtional to the width_scale in command line (passed as varible scale in this function call)
    */
    if(x.isMember("webapp") && x["webapp"].isMember("width-factor"))
    {
        scale = scale * x["webapp"]["width-factor"].asDouble();
    }
    
    base_name = x["base_name"].asString();
    npeak = x["database_ppm"].size(); 
    
    database_ppm.resize(npeak);
    database_intensity.resize(npeak);
    database_amplitude.resize(npeak);
    database_sigma.resize(npeak);
    database_gamma.resize(npeak);
    original_ppm.resize(npeak);
    match_ppm.resize(npeak);
    match_sigma.resize(npeak);
    match_gamma.resize(npeak);
    peak_distortion_factor.resize(npeak,1.0); //1.0 means no distortion
    match_type.resize(npeak);
    match.resize(npeak);
    match_confident_level.resize(npeak);

    for(int i=0;i<npeak;i++){
        database_ppm[i] = x["database_ppm"][i].asDouble();
        /**
         * Important, it is called database_intensity, but it is actually the amplitude of the peak in json file!!!
        */
        database_amplitude[i] = x["database_amplitude"][i].asDouble();
        database_intensity[i] = x["database_intensity"][i].asDouble();
        database_sigma[i] = x["database_sigma"][i].asDouble();
        database_gamma[i] = x["database_gamma"][i].asDouble();
        match_ppm[i] = x["match_ppm"][i].asDouble();
        match_type[i] = x["match_type"][i].asInt();
        match[i] = x["match"][i].asInt();
        match_confident_level[i] = x["match_confident_level"][i].asDouble();
        original_ppm[i] = x["original_ppm"][i].asDouble();
        match_sigma[i] = x["match_sigma"][i].asDouble() * scale;
        match_gamma[i] = x["match_gamma"][i].asDouble() * scale;
    }

    match_height_factor = 1.0;
    match_width_factor = 1.0;
   
    /**
     * Peak volume is database_intensity * v_fitted. In other words, v_fitted scale database to fit the experiment.
    */
    v_fitted = x["v_fitted"].asDouble();
    total_mean_confident_level = x["confident_level"].asDouble();


    //load in peak group information from "pms_group"
    for(int i=0;i<x["pms_group"].size();i++){
        peak_groups.push_back(x["pms_group"][i]["n_peaks"].asInt());
        peak_group_normal_peaks.push_back(x["pms_group"][i]["n_normal"].asInt());
    }

    /**
     * Example: 
     * peak_groups  = {2,3}, means 1st peak group has 2 peaks, 2nd peak group has 3 peaks
     * So,
     * peak_group_boundaries = {0,2,5}
    */
    std::vector<int> peak_group_boundaries;
    peak_group_boundaries.push_back(0);
    int n = peak_groups.size();
    for(int i=0;i<n;i++)
    {
        peak_group_boundaries.push_back(peak_group_boundaries[i]+peak_groups[i]);
    }

    /**
     * relative positioanl restrain is only applied to the neighboring peaks in the same peak group
     * We precalcualte the neighboring peaks in the same peak group here and their distance in ppm
    */
    for(int i=0;i<peak_group_boundaries.size()-1;i++)
    {
        int b = peak_group_boundaries[i];
        int e = peak_group_boundaries[i+1];
        for(int j1=b;j1<e-1;j1++)
        {   
            int j2=j1+1; 
            /**
             * relative position restrain is only applied to the neighboring peaks in the same peak group, that is, between j1 and j2 here
            */
            relative_position_restrain_peaks_1.push_back(j1);
            relative_position_restrain_peaks_2.push_back(j2);
            relative_position_restrain_distance.push_back(fabs(original_ppm[j1]-original_ppm[j2]));
        }
        absolute_position_restrain_peaks.push_back(b);
        absolute_position_restrain_position.push_back(original_ppm[b]);
    }

    n_relative_position_restrain_peaks = relative_position_restrain_distance.size();


    /**
     * For verification. Get total database_intensity for each peak group and print them out
    */
    if(n_verbose_global_1d>1)
    {
        std::cout<<"For compound "<<base_name<<", total intensity of each peak group: "<<std::endl;
        for(int i=0;i<peak_group_boundaries.size()-1;i++)
        {
            int b = peak_group_boundaries[i];
            int e = peak_group_boundaries[i+1];
            double total_intensity=0.0;
            for(int j=b;j<e;j++)
            {
                total_intensity += database_intensity[j];
            }
            std::cout<<"Peak group "<<i<<" total intensity: "<<total_intensity*170<<std::endl;
        }
        std::cout<<std::endl;
    }

   
    return true;
}


/**
 * Save matched compound to json file, after fitting. 
 * We may also read this file to run fit again. So we need to be compatible with the file format of load_matched_compound
*/
bool one_matched_compound::save_matched_compound(Json::Value &root)
{
    /**
     * Step1, copy all the information from self_json to root
    */
    root=self_json; 

    /**
     * Step 2, update values changed by fitting: match_ppm, match_sigma, match_gamma, and v_fitted
    */
    for(int i=0;i<npeak;i++)
    {
        root["match_ppm"][i] = match_ppm[i];
        root["match_sigma"][i] = match_sigma[i];
        root["match_gamma"][i] = match_gamma[i];
        /**
         * Update match_amplitude
         * Remember database_intensity will not change by fitting but 
         * change of match_sigma and match_gamma will effetively change match_amplitude.
        */
        root["match_amplitude"][i] = database_intensity[i]*voigt(0,match_sigma[i],match_gamma[i]);
    }
    root["v_fitted"] = v_fitted;

    return true;
};

/**
 * @brief Generate a voigt convolution of one peak. 
 * 
 * @param peak_index index of the peak in the matched compound
 * @param start first index (inclusion) of the convoluted spectrum. ppm of this index is begin1 + step1*start
 * @param stop last index (exclusion) of the convoluted spectrum. ppm of this index is begin1 + step1*stop
 * @param v spectrum after convolution. size is stop-start
 * @param cutoff cutoff for the convolution. If v_current<cutoff*v_center, then stop convolution
 * @return true 
 */

bool one_matched_compound::voigt_convolution_of_one_peak(int peak_index, int &start,int &stop,std::vector<double> &v,const double cutoff)
{
    double peak_center=match_ppm[peak_index];

    //find nearest data points of the peak center. Keep in mind that step1 is a negative number. One example: stop=10, step1=-0.1 and begin1=0.
    int ic = int((peak_center - begin1)/step1+0.5);
    double ic_ppm = begin1 + step1*ic;

    double v_current = voigt(peak_center-ic_ppm,match_sigma[peak_index]*match_width_factor,match_gamma[peak_index]*match_width_factor);
    double v_center=v_current;
    int i1 = ic;
    while(v_current>cutoff*v_center)
    {
        v.push_back(v_current*v_fitted*database_intensity[peak_index]); 
        i1 = i1 - 1;
        double i1_ppm = begin1 + step1*i1;
        v_current = voigt(peak_center-i1_ppm,match_sigma[peak_index]*match_width_factor,match_gamma[peak_index]*match_width_factor);
    }
    start = i1+1; //start is the first index of the spectrum. We need to add 1 because last v_current at i1 is less than 0.01, so it is not included in v

    //now do the same thing for the right side. 
    std::reverse(v.begin(),v.end()); //First flip v
    int i2 = ic+1; //start from the next point because the current point is already in v
    double i2_ppm = begin1 + step1*i2;
    v_current = voigt(peak_center-i2_ppm,match_sigma[peak_index]*match_width_factor,match_gamma[peak_index]*match_width_factor);
    
    while(v_current>cutoff*v_center)
    {
        v.push_back(v_current*v_fitted*database_intensity[peak_index]); 
        i2 = i2 + 1;
        i2_ppm = begin1 + step1*i2;
        v_current = voigt(peak_center-i2_ppm,match_sigma[peak_index]*match_width_factor,match_gamma[peak_index]*match_width_factor);
    }
    //stop is the last index of the spectrum. We do not need to add 1 because last v_current at i2 is less than 0.01, so it is not included in v. stop is exclusive
    stop = i2; 

    return true;
};


/**
 * @brief Generate a voigt convolution of one compound, from all the peaks in the compound
 * @param cutoff cutoff for the convolution. If v_current<cutoff*v_center, then stop convolution for each peak.
 * @param v spectrum after convolution. 
 * @param v_index index of the spectrum. ppm of this index is begin1 + step1*v_index
 * v and v_index have the same size.
*/
bool one_matched_compound::voigt_convolution(std::vector<double> &v, std::vector<int> &v_index,double cutoff)
{
    double small=std::numeric_limits<double>::min();
    std::vector<std::vector<double>> v_all;
    std::vector<int> tstarts,tstops;
    int min_start=100000000;
    int max_stop=0;
    for(int i=0;i<npeak;i++)
    {
        std::vector<double> v_new;
        int start,stop;
        /**
         * @brief voigt_convolution_of_one_peak will return v_new, start and stop
         * v_new is the spectrum of one peak
         * start is the first index of the spectrum and stop is the last index of the spectrum (exclusive)
        */
        voigt_convolution_of_one_peak(i,start,stop,v_new,cutoff);
        v_all.push_back(v_new);
        tstarts.push_back(start);
        tstops.push_back(stop);

        if(start<min_start) min_start=start;
        if(stop>max_stop) max_stop=stop;
    }

    //now we need to merge all the v_new
    int n = max_stop - min_start;
    std::vector<double> vtemp(n,0.0);
    for(int i=0;i<npeak;i++)
    {
        int start = tstarts[i];
        int stop = tstops[i];
        //copy v_all[i] to vtemp[start-min_start:stop-min_start]. skip water region [water_index_start:water_index_stop]
        for(int j=start;j<stop;j++)
        {
            if(j<water_index_start || j>=water_index_stop) //exclude water region
            {
                vtemp[j-min_start] += v_all[i][j-start];
            }
        }
    }

    /** Now copy vtemp to v, and copy corresponding index to v_index.
     * In this process, skip all the points that are less than small (effectively 0.0)
    */ 
    for(int i=0;i<n;i++)
    {
        if(vtemp[i]>small)
        {
            v.push_back(vtemp[i]);
            v_index.push_back(i+min_start);
        }
    }

    return true;
}

/**
 * @brief Optimize peak postions, sigma, gamma of each peak (npeak*3 fitting parameters) and match_height_factor (1 fitting parameter) to match theoretical spectrum with the v
 * Weighted version by v_weight
 * @param v_total spectral maximal. Fitted spectrum can't be larger than this
 * @param v target spectrum
 * @param v_index index of the spectrum. ppm of this index is begin1 + step1*v_index
 * @return true 
 */
bool one_matched_compound::compound_optimization(std::vector<double> &v_total, std::vector<double> &v, const std::vector<int> &v_index, const std::vector<double> &v_weight)
{
    //convert v_index to v_ppm, using begin1 and step1
    std::vector<double> v_ppm(v_index.size());
    for(int i=0;i<v_index.size();i++)
    {
        v_ppm[i] = begin1 + step1*v_index[i];
    }

    //scale down the v to avoid numerical problem, using match_height_factor
    for(int i=0;i<v.size();i++)
    {
        v[i] = v[i]/v_fitted;
        v_total[i] = v_total[i]/v_fitted;
    }

    /**
     * This is an important step. We will fit each peak invividually, using the same v_ppm and v_weight,
     * with postional restrain and sigma/gamma bounds to estimate the upper bound of match_height_factors.
     * The min of these upper bounds will be used as the upper bound of match_height_factor in the final fitting.
    */
    if(loop==-1)
    {
        upper_bound_match_height_factor = std::numeric_limits<double>::max();
        for(int i=0;i<npeak;i++)
        {
            /**
             * if original_ppm[i] is in the (expanded) water region, then skip it
            */
            if(original_ppm[i]>=4.2 && original_ppm[i]<5.2)
            {
                continue;
            }

            ceres::Problem problem_one_peak;

            std::vector<double> database_intensity_one_peak(1,database_intensity[i]);
            std::vector<double> original_ppm_one_peak(1,original_ppm[i]);
            std::vector<double> database_sigma_one_peak(1,database_sigma[i]);
            std::vector<double> database_gamma_one_peak(1,database_gamma[i]);
            
            /**
             * We replace v with v_total,because this step is for upper bound estimation. If we are using v, we may underestimate the upper bound by a lot for weak peaks which
             * is overlapping with bigger peaks.
            */
            Voigt_functor_shape_w *cost_function = new Voigt_functor_shape_w(database_intensity_one_peak, original_ppm_one_peak,database_sigma_one_peak,database_gamma_one_peak,v_total,v_ppm,v_weight,v_total);
            cost_function->set_n_residuals(v.size());
            cost_function->parameter_block_sizes()->push_back(1); //match_height_factor
            cost_function->parameter_block_sizes()->push_back(1); //match_ppm
            cost_function->parameter_block_sizes()->push_back(1); //match_sigma
            cost_function->parameter_block_sizes()->push_back(1); //match_gamma

            /**
             * We don't want to change match_ppm, match_sigma, match_gamma, so we define copies of them here
            */
            double match_ppm_one_peak = match_ppm[i];
            double match_sigma_one_peak = match_sigma[i];
            double match_gamma_one_peak = match_gamma[i];

            problem_one_peak.AddResidualBlock(cost_function, NULL, &match_height_factor, &match_ppm_one_peak, &match_sigma_one_peak, &match_gamma_one_peak);

            
            /**
             * Set up upper and lower bounds for sigma and gamma. Apply 6.0 and 0.25 becasue we allow very wide peaks and very narrow peak in this step only. 
             * And match_height_factor can not be negative
            */
            problem_one_peak.SetParameterLowerBound(&match_sigma_one_peak,0,database_sigma[i]*sigma_gamma_lower_bound*0.25);
            problem_one_peak.SetParameterLowerBound(&match_gamma_one_peak,0,database_gamma[i]*sigma_gamma_lower_bound*0.25);
            problem_one_peak.SetParameterUpperBound(&match_sigma_one_peak,0,database_sigma[i]*sigma_gamma_upper_bound*6.0);
            problem_one_peak.SetParameterUpperBound(&match_gamma_one_peak,0,database_gamma[i]*sigma_gamma_upper_bound*6.0);
            problem_one_peak.SetParameterLowerBound(&match_height_factor,0,0.0);

            /**
             * Set up upper and lower bounds for match_ppm_one_peak. Peak can only move within 0.001 ppm
            */
            problem_one_peak.SetParameterLowerBound(&match_ppm_one_peak,0,original_ppm[i]-0.001);
            problem_one_peak.SetParameterUpperBound(&match_ppm_one_peak,0,original_ppm[i]+0.001);

            ceres::Solver::Options options;
            options.max_num_iterations = 250;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary summary_one_peak;
            ceres::Solve(options, &problem_one_peak, &summary_one_peak);

            if(n_verbose_global_1d>0)
            {
                std::cout<<"For peak "<<i<<" at "<<match_ppm_one_peak<<", upper bound of match_height_factor is "<<match_height_factor<<" ";
            }
            if(match_height_factor<upper_bound_match_height_factor)
            {
                upper_bound_match_height_factor = match_height_factor;
            }
            if(n_verbose_global_1d>0)
            {
                std::cout<<"Min is "<<upper_bound_match_height_factor<<std::endl;
            }
        }
        /**
         * upper_bound_match_height_factor is scaled up to account for possible error in spin simulation.
        */
        upper_bound_match_height_factor*=2.00;
    }
       


    if(b_allow_overal_width_rescale_factor) //uisng match_width_factor to rescale sigma and gamma gloablly: one for all peaks
    {
        ceres::Problem problem;
        /**
         * Define cost function for spectral fitting
         */
        Voigt_functor_shape_width_w *cost_function = new Voigt_functor_shape_width_w(database_intensity, original_ppm, database_sigma, database_gamma, v, v_ppm, v_weight, v_total);

        /**
         * Set number of residuals
         */
        cost_function->set_n_residuals(v.size());

        /**
         * Set parameter block size
         * 1 for match_height_factor
         * npeak for match_ppm
         * npeak for match_sigma
         * npeak for match_gamma
         */
        cost_function->parameter_block_sizes()->push_back(1);
        cost_function->parameter_block_sizes()->push_back(match_ppm.size());
        cost_function->parameter_block_sizes()->push_back(match_sigma.size());
        cost_function->parameter_block_sizes()->push_back(match_gamma.size());
        cost_function->parameter_block_sizes()->push_back(1); //match_width_factor


        problem.AddResidualBlock(cost_function, NULL, &match_height_factor, match_ppm.data(), match_sigma.data(), match_gamma.data(), &match_width_factor);

         /**
         * Define a cost function for postional restrain
        */
        ceres::DynamicNumericDiffCostFunction<pos_restrain_functor> *cost_function_pos = new ceres::DynamicNumericDiffCostFunction<pos_restrain_functor>(
            new pos_restrain_functor(absolute_position_restrain_peaks,absolute_position_restrain_position,relative_position_restrain_peaks_1,relative_position_restrain_peaks_2,relative_position_restrain_distance));
        cost_function_pos->AddParameterBlock(match_ppm.size());
        cost_function_pos->SetNumResiduals(absolute_position_restrain_peaks.size()+n_relative_position_restrain_peaks);

        /**
         * Add residual block for postional restrain. Fittting parameters are match_ppm (npeak)
        */
        ceres::ResidualBlockId r_id_pos=problem.AddResidualBlock(cost_function_pos, NULL, match_ppm.data());


        /**
         * Add sigma and gamma bounds. Defined as global varibles (static member)
        */
        for (std::size_t i = 0; i < match_sigma.size(); ++i) {
            problem.SetParameterLowerBound(match_sigma.data(),i,database_sigma[i]*sigma_gamma_lower_bound);
            problem.SetParameterLowerBound(match_gamma.data(),i,database_gamma[i]*sigma_gamma_lower_bound);
            problem.SetParameterUpperBound(match_sigma.data(),i,database_sigma[i]*sigma_gamma_upper_bound);
            problem.SetParameterUpperBound(match_gamma.data(),i,database_gamma[i]*sigma_gamma_upper_bound);
        }

        /**
         * Constrain match_width_factor to be beween 0.25 and 4.0. Prevent too narrow or too wide peaks.
         * Remember this is on top of sigma and gamma bounds above.
        */
        problem.SetParameterLowerBound(&match_width_factor,0,match_width_factor_lower_bound);
        problem.SetParameterUpperBound(&match_width_factor,0,match_width_factor_upper_bound);


        /**
         * Constrain match_height_factor to be positive (concentration can not be negative) and less than upper_bound_match_height_factor estimated above
        */
        problem.SetParameterLowerBound(&match_height_factor,0,0.0);
        // problem.SetParameterUpperBound(&match_height_factor,0,upper_bound_match_height_factor);


        ceres::Solver::Options options;
        options.max_num_iterations = 250;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        /**
         * Make sure match_sigma and match_gamma is positive for subsequent calculation. They are taken abs in the cost function
        */
        for(int i=0;i<match_sigma.size();i++)
        {
            match_sigma[i] = fabs(match_sigma[i]);
            match_gamma[i] = fabs(match_gamma[i]);
        }  

    }
    else //no gloabl peak width factor
    {
        /**
         * Now we do the final fitting, with upper_bound_match_height_factor as the upper bound of match_height_factor
        */
        ceres::Problem problem;
        ceres::ResidualBlockId r_id;
        if(b_analytical_jacobian==true)
        {
            /**
             * Define cost function for spectral fitting
            */
            Voigt_functor_shape_w *cost_function = new Voigt_functor_shape_w(database_intensity, original_ppm,database_sigma,database_gamma,v, v_ppm,v_weight,v_total);

            /**
             * Set number of residuals
            */
            cost_function->set_n_residuals(v.size());

            /**
             * Set parameter block size
             * 1 for match_height_factor
             * npeak for match_ppm
             * npeak for match_sigma
             * npeak for match_gamma
            */
            cost_function->parameter_block_sizes()->push_back(1);
            cost_function->parameter_block_sizes()->push_back(match_ppm.size());
            cost_function->parameter_block_sizes()->push_back(match_sigma.size());
            cost_function->parameter_block_sizes()->push_back(match_gamma.size());
            
            r_id=problem.AddResidualBlock(cost_function, NULL, &match_height_factor, match_ppm.data(), match_sigma.data(), match_gamma.data());
        }
        
        else //numerical cost function
        {
            /**
             * Define cost function for spectral fitting
            */
            ceres::DynamicNumericDiffCostFunction<Voigt_functor_shape_w> *cost_function = new ceres::DynamicNumericDiffCostFunction<Voigt_functor_shape_w>(
                new Voigt_functor_shape_w(database_intensity, original_ppm,database_sigma,database_gamma,v, v_ppm,v_weight,v_total));


            /**
             * Set number of fitting paramters. This is required for DynamicNumericDiffCostFunction
            */
            cost_function->AddParameterBlock(1);
            cost_function->AddParameterBlock(match_ppm.size());            
            cost_function->AddParameterBlock(match_sigma.size()); 
            cost_function->AddParameterBlock(match_gamma.size()); 
            cost_function->SetNumResiduals(v.size()); 

            /**
             * Add residual block for spectral fitting. Fittting parameters are match_height_factor(1), match_ppm (npeak), match_sigma (npeak) and match_gamma (npeak)
            */
            r_id=problem.AddResidualBlock(cost_function, NULL, &match_height_factor,match_ppm.data(),match_sigma.data(),match_gamma.data());
        }
        

        /**
         * Define a cost function for postional restrain
        */
        ceres::DynamicNumericDiffCostFunction<pos_restrain_functor> *cost_function_pos = new ceres::DynamicNumericDiffCostFunction<pos_restrain_functor>(
            new pos_restrain_functor(absolute_position_restrain_peaks,absolute_position_restrain_position,relative_position_restrain_peaks_1,relative_position_restrain_peaks_2,relative_position_restrain_distance));
        cost_function_pos->AddParameterBlock(match_ppm.size());
        cost_function_pos->SetNumResiduals(absolute_position_restrain_peaks.size()+n_relative_position_restrain_peaks);

        /**
         * Add residual block for postional restrain. Fittting parameters are match_ppm (npeak)
        */
        ceres::ResidualBlockId r_id_pos=problem.AddResidualBlock(cost_function_pos, NULL, match_ppm.data());


        /**
         * Add sigma and gamma bounds. Defined as global varibles (static member)
        */
        for (std::size_t i = 0; i < match_sigma.size(); ++i) {
            problem.SetParameterLowerBound(match_sigma.data(),i,database_sigma[i]*sigma_gamma_lower_bound);
            problem.SetParameterLowerBound(match_gamma.data(),i,database_gamma[i]*sigma_gamma_lower_bound);
            problem.SetParameterUpperBound(match_sigma.data(),i,database_sigma[i]*sigma_gamma_upper_bound);
            problem.SetParameterUpperBound(match_gamma.data(),i,database_gamma[i]*sigma_gamma_upper_bound);
        }
        /**
         * Constrain match_height_factor to be positive (concentration can not be negative) and less than upper_bound_match_height_factor estimated above
        */
        problem.SetParameterLowerBound(&match_height_factor,0,0.0);
        // problem.SetParameterUpperBound(&match_height_factor,0,upper_bound_match_height_factor);


        ceres::Solver::Options options;
        options.max_num_iterations = 250;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        /**
         * Make sure match_sigma and match_gamma is positive for subsequent calculation. They are taken abs in the cost function
        */
        for(int i=0;i<match_sigma.size();i++)
        {
            match_sigma[i] = fabs(match_sigma[i]);
            match_gamma[i] = fabs(match_gamma[i]);
        }

    
    #ifdef DEBUG
        std::vector< ceres::ResidualBlockId > to_eval;
        to_eval.push_back( r_id );
        
        ceres::Problem::EvaluateOptions evaluate_options;
        evaluate_options.residual_blocks = to_eval;
        double total_cost = 0.0;
        std::vector<double> evaluated_residuals;
        problem.Evaluate(evaluate_options, &total_cost, &evaluated_residuals, nullptr, nullptr);


        //scale back v, v_total and evaluated_residuals
        for(int i=0;i<v.size();i++)
        {
            v_total[i] = v_total[i]*v_fitted;
            v[i] = v[i]*v_fitted;
            evaluated_residuals[i] = evaluated_residuals[i]*v_fitted;
        }


        /**
         * print v_ppm and v for debug, to a file named: "v_ppm_${compound_index}_${loop}.txt"
         */
        std::ofstream fout;
        fout.open("v_ppm_" + std::to_string(compound_index) + "_" + std::to_string(loop) + ".txt");
        for (int i = 0; i < v_ppm.size(); i++)
        {
            fout << v_ppm[i] << " " << v[i] << " " << v[i] - evaluated_residuals[i]/v_weight[i] << std::endl;
        }
        fout.close();

    #endif
    }

    /**
     * Update v_fitted and upper_bound_match_height_factor using match_height_factor then reset match_height_factor to 1.0
     * By doing this, we only need to keep track of one parameter v_fitted for each compound
    */
    v_fitted = v_fitted*match_height_factor;
    // upper_bound_match_height_factor /= match_height_factor;  //upper_bound_match_height_factor is not use any more? Maybe no need to update it?
    match_height_factor = 1.0;

    return true;
}

// class db_match_1d
// constructor
db_match_global_1d::db_match_global_1d(double c1, double c2)
{
    water_width = c1;
    db_limit = c2;
    water_index_start = 0;
    water_index_stop = 0;
};

// destructor
db_match_global_1d::~db_match_global_1d(){
    //
};

/**
 * @brief reference correction of ppm
*/
bool db_match_global_1d::ref_correction_spectrum(double t)
{
    /**
     * Need to to define water region. This is a static varible shared by all compounds and this class
    */
    ref_correction_global_1d=t;


    stop1 = stop1 + t;
    begin1 = begin1 + t;

    return true;
}

/**
 * @brief set point range of water region from ppm. Water is usually at 4.7 ppm (before reference correction)
*/
bool db_match_global_1d::deal_with_water()
{
    /**
     * Convert 4.7 ppm into index, using begin1 and step1
    */
    int water_center_index = int((4.7 - begin1 + ref_correction_global_1d)/step1+0.5);

    /**
     * Convert from ppm to index
    */
    int water_width_int = water_width/fabs(step1);

    water_index_start = water_center_index-water_width_int;
    water_index_stop  = water_center_index+water_width_int;

    if(n_verbose_global_1d>0)
    {
        std::cout<<"Water region is centered at "<<water_center_index<< " and from "<<begin1+step1*water_index_start<<" to "<<begin1+step1*water_index_stop<<std::endl;
    }

    return true;
}

/**
 * load matched compounds from json file, generated by db_match_1d (main_1d.cpp)
 * @par infname: input json file name
 * @par scale: scale factor for peak width, because the width of peaks in the database is not always consistent with experimental peaks
 */
bool db_match_global_1d::load_matched_compounds(std::string filename,double scale)
{
    std::ifstream infile;
    infile.open(filename);
    if (!infile.is_open())
    {
        std::cout << "Error: cannot open file " << filename << std::endl;
        return false;
    }

    Json::Value root;
    infile >> root;

    /**
     * root is an array of compounds.
     * each compound is an object with many fields: base_name, database_gamma, etc.
     * most of the field is an array of numbers
     * the field "base_name" is a string
    */

    //loop over all compounds
    for(int i=0;i<root.size();i++)
    {
        class one_matched_compound one_compound;
        one_compound.load_matched_compound(root[i],scale);  
        one_compound.set_ppm_range(stop1,step1,begin1,xdim,water_index_start,water_index_stop);
        one_compound.compound_index = i; //for debug
        matched_compounds.emplace_back(one_compound);
    }

    /**
     * return false if matched_compounds is empty
    */
    if(matched_compounds.size()==0)
    {
        return false;
    }


    //run convolution of each compound and store the results in simulated_spectrum 
    simulated_spectrum.clear();
    simulated_spectrum.resize(spect.size(),0);
    for(int i=0;i<matched_compounds.size();i++)
    {
        std::vector<int> v_index;
        std::vector<double> v;
        matched_compounds[i].voigt_convolution(v,v_index);
        for(int k=0;k<v.size();k++)
        {
            simulated_spectrum[v_index[k]] += v[k];
        }
    }

    return true;
};

/**
 * @brief save matched compounds to json file
*/
bool db_match_global_1d::save_matched_compounds(std::string filename)
{
    Json::Value root = Json::arrayValue;
    for(int i=0;i<matched_compounds.size();i++)
    {
        matched_compounds[i].save_matched_compound(root[i]);
    }

    std::ofstream outfile;
    outfile.open(filename);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << filename << std::endl;
        return false;
    }
    outfile << root;
    outfile.close();

    return true;
}

/**
 * @brief Optimize peak positions, match_height_factor and match_height_factor of each compound
 * to minimize the difference between simulated_spectrum and spect
 * Using a Gaussian-mixure model to reduce the size of the problem
 * @return true 
 */

bool db_match_global_1d::optimization(int rmax,bool b_weighted)
{

    double time_start = get_wall_time();
    for(int loop=0;loop<=rmax;loop++)
    {
        /**
         * for debug, print something for each compound
         * At loop 0, we haven't done anything, so we skip it
         * At loop loop, we are printing result after loop loop-1 opt.
        */
        if(n_verbose_global_1d>1 && loop>0)
        {
            std::cout<<"After loop "<<loop-1<<std::endl;
            for(int i=0;i<matched_compounds.size();i++)
            {
                std::cout << i << " " << matched_compounds[i].get_v_fitted();
                std::cout << std::endl;
            }
            std::cout<<std::endl<<std::endl;
        }


        std::vector<std::vector<double>> v_all,v_all_for_opt;
        std::vector<std::vector<int>> v_index_all,v_index_all_for_opt;
        simulated_spectrum.clear();
        simulated_spectrum.resize(spect.size(),0); //total simulated spectrum, including all compounds
        for(int i=0;i<matched_compounds.size();i++)
        {
            std::vector<double> v,v_for_opt;
            std::vector<int> v_index,v_index_for_opt;
            matched_compounds[i].voigt_convolution(v,v_index,0.002); //larger vonvolution width for deconvolute experiments into predicted compounds
            matched_compounds[i].voigt_convolution(v_for_opt,v_index_for_opt,0.02); //smaller vonvolution width for fast optimization of each compound
            v_all.push_back(v);
            v_index_all.push_back(v_index);
            v_all_for_opt.push_back(v_for_opt);
            v_index_all_for_opt.push_back(v_index_for_opt);
            for(int k=0;k<v.size();k++)
            {
                simulated_spectrum[v_index[k]] += v[k];
            }
        }

       
#ifdef DEBUG
        /**
         * @brief for debug, print simulated_spectrum to a file named: "simulated_spectrum_${loop-1}.txt"
         * loop-1 because we haven't done anything at loop 0
         * and at loop=1, we are printing result after loop 0 opt.
        */
        if( loop>0)
        {
            std::string filename="simulated_spectrum_";
            filename=filename+std::to_string(loop-1);
            filename=filename+".txt";
            std::ofstream outfile(filename);
            for(int i=0;i<simulated_spectrum.size();i++)
            {
                outfile<<simulated_spectrum[i]<<std::endl;
            }
            outfile.close();
        }
#endif

        /**
         * @brief check if we have converged
         * When loop == rmax, we have done rmax optimization, so we break after generated simulated_spectrum (no more optimization run)
        */
        if(loop==rmax)
        {
            break;
        }


        //now we have simulated_spectrum and v_all
        //for each compound, we deconvolve the simulated_spectrum to each compounds (Gaussian mixture model)

        for(int i=0;i<matched_compounds.size();i++)
        {
            std::cout<<"Work on compound "<<i<<" ("<<matched_compounds[i].get_base_name()<<") out of "<<matched_compounds.size()<<std::endl;
            std::vector<double> v_deconvolved;
            std::vector<double> v_weight;
            std::vector<double> v_total; //experimental spectral 
            for(int j=0;j<v_all_for_opt[i].size();j++)
            {
                int k=v_index_all_for_opt[i][j];
                double v1=v_all_for_opt[i][j]; //v1 is the intensity compound i at index k
                double v2=simulated_spectrum[k]; //v2 is the total intensity at index k
                double v3=std::max(spect[k],0.0f); //v3 is the intensity of experimental spectrum at index k
                if(v2>0)
                {
                    double r=v1/v2; 
                    double r2=v1/v3;
                    /**
                     * r and r2 must be from 0 to 1
                    */
                    if(r>1.0) r=1.0;
                    if(r<0.0) r=0.0;
                    if(r2>1.0) r2=1.0;
                    if(r2<0.0) r2=0.0;
                    v_deconvolved.push_back(r*v3);
                    v_weight.push_back(std::min(r,r2));
                }
                else
                {
                    v_deconvolved.push_back(0.0);
                    v_weight.push_back(0.00001);
                }
                v_total.push_back(v3);
            }

            /**
            * @brief now we have v_deconvolved, which is the v_deconvolved spectrum of compound i
            * we need to optimize the peak positions, peak shape (sigma,gamma) match_height_factor 
            */
            matched_compounds[i].loop = loop; // for debug

            if (b_weighted)
            {
                matched_compounds[i].compound_optimization(v_total,v_deconvolved, v_index_all_for_opt[i], v_weight);
            }
            else
            {
                // to do: add a non-weighted version of optimization_v4 and call it here
                matched_compounds[i].compound_optimization(v_total,v_deconvolved, v_index_all_for_opt[i], v_weight);
            }

            std::cout<<"Finished optimization of compound "<<i<<" ("<<matched_compounds[i].get_base_name()<<") out of "<<matched_compounds.size()<<" at loop "<<loop<<" out of "<< rmax <<", Wall time is "<<get_wall_time()-time_start<<std::endl;
        }
        std::cout<<"Finished optimization of all compounds at loop "<<loop<<" out of "<< rmax <<", Wall time is "<<get_wall_time()-time_start<<std::endl;
    }



    for (int i = 0; i < matched_compounds.size(); i++)
    {
        if(n_verbose_global_1d>0)
        {
            matched_compounds[i].print();   
        }
        matched_compounds[i].apply_width_rescale_factor();
    }

    return true;
};

/**
 * print some basic information
 */
bool one_matched_compound::print()
{
    std::cout<<"Matched compound "<<compound_index<<" ("<<base_name<<") peak width factor is "<<match_width_factor<<" and fitted V is "<<v_fitted<<std::endl;
    return true;
}

/**
 * apply match_width_factor to sigma and gamma of each peak
 */
bool one_matched_compound::apply_width_rescale_factor()
{
    for(int i=0;i<npeak;i++)
    {
        match_sigma[i] = match_sigma[i]*match_width_factor;
        match_gamma[i] = match_gamma[i]*match_width_factor;
    }
    match_width_factor = 1.0;
    return true;
}

    /**
     * @brief Currently only support ft1 format.
     * Will write ft1 if the header is set (we have read in a spectrum in ft1 format)
     * Otherwise, will write a txt file, like the input file (generated by totxt command of Topspin)
     * @param fname output file name
     * @return true
     */
    bool db_match_global_1d::save_simulated_spectrum(std::string fname, bool b_json)
{

    if(b_header==true) //we read in a spectrum in ft1 format
    {
        FILE *fp;
        fp = fopen(fname.c_str(),"w");
        if(fp==NULL)
        {
            std::cout << "Error: cannot open file " << fname << std::endl;
        }
        fwrite(&header,512,sizeof(float),fp);

        /**
         * NMRpipe use float32, so we need to convert simulated_spectrum to float32
        */
        std::vector<float> simulated_spectrum_float32(simulated_spectrum.size());
        for(int i=0;i<simulated_spectrum.size();i++)
        {
            simulated_spectrum_float32[i] = simulated_spectrum[i];
        }

        fwrite(simulated_spectrum_float32.data(),simulated_spectrum_float32.size(),sizeof(float),fp);
        fclose(fp);
    }
    else
    {
        std::cout<<"Can't write ft1 file because header is not available."<<std::endl;
    }

    if(b_json==true)
    {
        Json::Value s = Json::arrayValue;
        for(int i=0;i<simulated_spectrum.size();i++)
        {
            s[i]=simulated_spectrum[i];
        }
        Json::Value root;
        root["simulated_spectrum"]=s;

        std::ofstream outfile("simulated_spectrum.json");
        outfile << root;
        outfile.close();
    }

    return true;
}