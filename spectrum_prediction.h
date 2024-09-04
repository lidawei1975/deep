#include <vector>
#include <string>

#include "contour.h"


struct compound
{   
    /**
     * Compound name and database origin.
     * origin will be used as a key to find the compound in the database.
    */
    std::string namne,origin,origin_1d;

    /**
     * Peak paramters will be saved in individual vectors.
     * We will use  a*exp_a,exp_x and exp_y to generate simulated spectrum.
     * a: database peak intensity.
     * exp_a: experimental peak height
     * exp_v: experimental peak volume.
     * exp_x: experimental peak position in direct dimension.
     * exp_y: experimental peak position in indirect dimension.
     * x: database peak position in direct dimension.
     * y: database peak position in indirect dimension.
    */
    std::vector<double> exp_a,exp_v,exp_x,exp_y;
    /**
     * Peak shape parameters.
    */
    std::vector<double> sigma_x,gamma_x;

    /**
     * Peak group information.
     * Same peak group means peaks from the same proton atom.
     * [3,4,8] means 0-3, 3-4, 4-8 are from the same proton atom 
     * in the above exp_a,exp_v,exp_x,exp_y,sigma_x,gamma_x variables.
    */
    std::vector<int> peak_group;


    /**
     * Peak parameters that are the same for all peaks from the same proton atom.
     * That is, below varibles have the same size as peak_group.size().
    */
    std::vector<double> a,x,y,sigma_y,gamma_y;
    
    /**
     * Flag to indicate whether we need to simulate the spectrum.
    */
    int simulate_spectrum_flag;
};

class spectrum_prediction
{
private:
    /**
     * Experimental spectrum information, we need to predict matched compound spectrum based on this information.
     */
    double b0;
    double begin,step,stop; //direct dimension ppm information. begin>stop and step is negative.
    double begin_c,step_c,stop_c; //indirect dimension ppm information. begin_c>stop_c and step_c is negative.
    int x_dim,y_dim; //direct and indirect dimension data points.
    double log_detection_limit,noise_level; //log detection limit and noise level.
    int n_acquisition,n_zf; //number of acquisition and zero filling. n_acquisition*n_zf should be equal to x_dim.
    double spectral_width,spectral_width_c; //spectral width in direct and indirect dimension.



    /**
     * Compounds.
    */
    std::vector<compound> compounds;

    /**
     * Simulated spectrum. Shared by all compounds because we only need to simulate one compound at a time.
     * @param simulated_spectrum column major simulated spectrum.
    */
    std::vector<std::vector<double>> simulated_spectrum; 

    /**
     * This is the sum of all simulated spectrum.
    */
    std::vector<std::vector<double>> sum_of_simulated_spectrum;

    /**
     * Define a group of rectangles to label the signal region.
     * [x1,y1] is the left bottom corner, [x2,y2] is the right top corner.
    */
    std::vector<int> signal_region_x1,signal_region_y1,signal_region_x2,signal_region_y2;

    /**
     * Json root to save all the spectral contour information.
    */
    Json::Value root;
    /**
     * Raw data array to save the contour data
    */
    std::vector<float> contour_data;


    void simulate_spectrum_of_one_compound(int compound_index);
    void calcualte_contour(Json::Value &,std::vector<float> &);

public:
    spectrum_prediction();
    ~spectrum_prediction();

    void load_query_json(std::string query_file);
    void simu_and_contour_one_by_one();
    void save_simulated_spectrum(std::string output_file);
    void save_simulated_spectrum_binary(std::string output_file);
    void save_sum_of_simulated_spectrum(std::string output_file);
};