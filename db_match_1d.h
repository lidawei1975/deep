#include <vector>
#include "spectrum_io_1d.h"
#include "db_1d.h"

#ifndef DB_MATCH_1D_H
#define DB_MATCH_1D_H
enum MatchType
{
    MatchType_normal,
    MatchType_covered,
    MatchType_bdl,
    MatchType_skipped,
    MatchType_water,
    MatchType_unknown
};

//static member variables. They are shared by all instances of pattern_match_1d and db_match_1d
struct match_1d_static_varibles
{
public:
    static int n_verbose_match_1d; //verbose level

    /**
     * @brief experimental peak list, used in both pattern_match_1d and db_match_1d
     * 
     */
    static std::vector<double> ppm, intensity_original;
    static std::vector<double> volume,confident_level;
    static std::vector<int> backgroud_flag; //backgroud_flag of experimental peak list. 0: not background, 1: background

    /**
     * @brief derived varible for db matching
     * 
     */
    static std::vector<double> effective_width; //effective peak width, defined as volume/height

    /**
     * @brief addtional exp peak information that is currently not used.
     * Thye will be copied to the output peak list with assignment information.
     */
    static std::vector<double> sigmax, gammax,pos,delta_intensity,peak_integral,peak_width;
    static std::vector<int> nround;
    static std::vector<std::string> user_comments; // user comments (peak name in experiment peak list)

    /**
     * @brief experimental spectrum information.
     */
    static std::vector<float> spectrum; //exp data
    static double stop,step,begin; //ppm range of exp data
    static double log_detection_limit; //log of detection limit
    static double max_value_of_exp_spectrum; //maximum value of exp spectrum

    /**
     * @brief below are the variables used to track which compound matches the experimental peak
     * At this time, only MatchType_normal is tracked.
     * They all have same size as ppm, intensity, etc.
     * They are vector of vector. The first index is the index of the peak in the experimental peak list.
     * The second is corresponding compound index 
     * In matching_compound_peak_index, it means peak index within the compound.
     * matching_compound_index[3]=[2,4] and matching_compound_peak_index[3]=[1,3] means
     * the 3rd peak in the experimental peak list
     * is matched 1th peak of 2nd compound and 3rd peak of 4th compound.
     * matching_compound_height[3] = {1.0,0.8} means the height of two the matched db peaks are 100% and 80% of the experimental peak height
     */
    static std::vector<std::vector<int>> matching_compound_index; //compound index. 
    static std::vector<std::vector<int>> matching_compound_peak_index; //compound index.
    static std::vector<std::vector<double>> matching_compound_height; //height of matched db peaks.
    


    /**
     * const variables
    */
    static const double cost_peak_pattern_locations;
    static const double cost_peak_pattern_height;
    static const double cost_overall_shift;
    static const double cost_reduce_high_peak;
    static const double peak_match_scale;
    static const double additional_cost_bdl,additional_cost_covered;

    /**
     * @brief Maximal number of splitted peaks in the DB list (of each proton) to be used in peak matching.
    */
    static const int n_max_peak_number;
};

class pattern_match_1d : public match_1d_static_varibles
{
private:

    /**
     * intensity of the experimental peak list, this is a copy of intensity_original (static member variable shared by all instances of pattern_match_1d and db_match_1d)
     * In try_alternative_match, intensity is modified to simulate the effect of other compounds
    */
    std::vector<double> intensity; 


    bool b_fake; //if true, assign all peaks to MatchType_water
    int peak_group_index;

    std::vector<double> database_peak_height,database_peak_ppm;

    double cutoff_pattern;
    double cutoff;

    double lb,ub; //lower and upper bound of peak heights to be matched. relative to the highest data point in the experimental spectrum

    std::vector<double> database_peak_ppm_original,database_peak_height_original; //original database peak list before any filtering. save a copy.
    std::vector<int> database_peak_removed; //same size as database_peak_ppm_original. 1: removed, 0: not removed
    std::vector<int> mapping_2_original; //mapping from database peak index to database_peak_ppm_original index
    double max_value_of_amplitude2; //maximum value of amplitude2 (from all database peak list of db_match_1d)

    std::vector<std::vector<int>> matching_candidates; // matching candidates[database peak index][all possible exp peak index]
    std::vector<std::vector<MatchType>> matching_candidates_type;  //type: normal, covered, bdl
    std::vector<std::vector<double>> matching_candidates_ppm; //ppm of matching candidates
    std::vector<std::vector<double>> matching_candidates_ppm_diff; //ppm difference of matching candidates wrt database peak
    std::vector<std::vector<double>> matching_candidates_height_ratio;  //height ratio of matching candidates wrt database peak
    std::vector<std::vector<double>> matching_candidates_height; //height of matching candidates
    
    std::vector<std::vector<int>> assignment; // assignment[assignment index][peak index in experimental peak list]
    std::vector<std::vector<MatchType>> assignment_type;
    std::vector<std::array<double,2>> assignment_cost;
    std::vector<double> assignment_min_intensity;

    double allowed_intensity_upper_bound; //upper bound of intensity for the compound, estibalished by this PMS, using the highest of all possible assignments

    std::vector<double> cost_at_intensity;

    std::vector<int> possible_assigned_peak; //save size as ppm. 1: assigned in any possible assignment, 0: not assigned

    //index of best at current intensity_check
    double intensity_check;
    int best_assignment_index;

    int number_normal; //number of normal peaks in best assignment
    std::vector<double> peak_confident_level_of_assignment; //confident level of each assiged exoerimental peak
    double mean_confident_level; //mean confident level of all assigned experimental peaks

    //sort_ndx index of sorted peaks according to intensity. 0: highest intensity, 1: second highest intensity, ...
    // Origianl peak list is sorted by ppm in acending order
    //sort_ndx_inversed is the inverse of sort_ndx (i.e. sort_ndx[sort_ndx_inversed[i]] = i)
    //sort_ndx_inversed is used to restore the original order of peaks in function get_best_assignment and in debug1.txt file
    std::vector<int> sort_ndx;
    std::vector<int> sort_ndx_inversed;

    //recursive function
    std::vector<std::vector<int>> assignment_wrt_candidate; //assignment_wrt_candidate[assignment index][all possible exp peak index, WRT matching_candidates]
    std::vector<std::vector<double>> assignment_covered_height_ratio; //assignment_covered_height_ratio[assignment index][exp spectral height/db peak height ratio at each exp peak index]. 0 if not covered
    std::vector<std::vector<double>> assignment_covered_ppm; //assignment_covered_ppm[assignment index][position in ppm of max height]. 0 if not covered
    void combinations(const int current_peak,std::vector<int> accum,const long unsigned int bdl_position);

    double get_max_spectral_data(double left, double right, int &max_position); //return the maximum spectral data between left and right. max_position is the position of the maximum spectral data


    std::vector<double> collision_intensity_factor; //peak intensity need to be reduced, caused by peak matching clasher (multiple compounds match one peak)
    

public:

    int db_index; //index of the database item in the database list

    pattern_match_1d();
    ~pattern_match_1d();
    bool init(int i, double x_, double y_,
                            std::vector<double> pattern_height_,
                            std::vector<double> pattern_,
                            double lb_, double ub_,
                            double max_value_of_amplitude2_);
    bool run_fake(); //assign all peaks to MatchType_water
    bool run();
    void calculate_cost(std::ofstream &);

    double calculate_cost_at_intensity(double,std::ofstream &,int);
    bool calculate_cost_at_intensity_with_details(std::ofstream &,double intensity_check_,std::vector<double> &costs_sorted, std::vector<int> &ndx_inversed);
    bool get_key_intensity(std::vector<double> &,std::vector<double> &);
    bool get_best_assignment(std::vector<int> &,std::vector<MatchType> &, std::vector<double> &, std::vector<double> &);
    bool get_best_assignment_v2(std::vector<int> &,std::vector<MatchType> &, std::vector<double> &, std::vector<double> &,int);
    int get_number_of_assignment();
    int get_number_normal();
    bool reset_number_normal();
    int get_number_of_peaks();
    double get_mean_confident_level();
    void set_collision_intensity_factor(std::vector<double>); //set collision_intensity_factor from db_match_1d
    void clear_collision_intensity_factor(); //set collision_intensity_factor to all 0


    /**
     * Get a reference to one assignment
    */
    const std::vector<int> &get_assignment(int) const;

    /**
     * Get a readonly reference to possible_assigned_peak
    */
    const std::vector<int> &get_possible_assigned_peak() const;

};


class db_match_1d : public spectrum_io_1d, public match_1d_static_varibles, public CDatabase_item
{
private:

    double reference_correction; //reference correction (+this value to all peaks and spe)
    
    double cutoff;
    double cutoff_pattern;
    double water_width;
    
    std::vector<double> ppm2, amplitude2, sigmax2, gammax2, vol2; //database peak list. vol2 is analytical peak volume
    double max_value_of_amplitude2; //maximum value of amplitude2

    /**
     * ppm per point in db
     * For simulated spectra, it is calculated according to experimental ppm range and number of points (database match experimental spectrum)
    */
    double step2; 
    /**
     * One peak group corresponds to one proton (one PMs)
     * npeak_group is the number of peak groups == pms.size()
     * peak_stop[i] is the stop index of the i-th peak group in the database peak list defined above
    */
    int npeak_group;
    std::vector<int> peak_stop;
    std::vector<int> proton_index_of_each_group; //proton index of each group, load from spin simulation file
    std::vector<double> pka_shift; //pka caused larger ppm cutoff of each group, load from pka_shift file. Same size as npeak_group

    /**
     * Total volume (from all splitting) of each proton (peak group)
    */
    std::vector<double> total_volume;

    std::vector<int> group_id; //group id of special peaks in the db (pKa is near 7.4)
    std::vector<double> delta_ppm; //delta_ppm of special peaks in the db (pKa is near 7.4)

    double v_fitted; //fitted value (concentration of matched compound) using peak height ratio
    std::vector<int> final_best_assignment;
    std::vector<MatchType> final_best_assignment_type;
    std::vector<double> final_best_assignment_height_ratio;
    std::vector<double> final_best_assignment_ppm;
    std::vector<double> matched_ppm_with_group_adjust; //ppm of matched peaks, after group based adjustment.
    std::vector<double> matched_ppm; //ppm of matched peaks, without group based adjustment.

    std::vector<pattern_match_1d> pms; //pattern_match_1d objects

    std::vector<int> pms_update_flag; //1: need update, 0: not need update

    /**
     * The final cost of the best assignment for this compound
    */
    double min_cost; 

    /**
     * Varibels for clash detection, re-assignment, etc
    */
    std::vector<int> clash_index_in_experiment; //experimental peak index with clash
    std::vector<int> clash_index_in_assignment; //index of the assignment with clash
    std::vector<int> clash_index_in_pms;        //index of the pms with clash
    std::vector<int> clash_index_in_peaks_of_pms; //index of the peak in the pms with clash
    std::vector<std::vector<int>> clash_index_in_each_pms_groups; //clash_index_in_each_pms_groups[2]={3,4} means pms[2]'s peak 3 and 4 have clahs with other compounds
    std::vector<std::vector<int>> exp_index_in_each_pms_groups; //exp_index_in_each_pms_groups[2]={13,24} means pms[2]'s peak 3 and 4 have clahs with other compounds, on exp peak index 13 and 24

    std::vector<double> collision_intensity_factor; ////peak intensity need to be reduced, caused by peak matching clasher (multiple compounds match one peak)

    

    /**
     * @param nstride_database_spectrum number of stride in the simulated database spectrum to adjust PPP
    */
    int nstride_database_spectrum;

    bool peak_reading_pipe(std::string infname);
    bool peak_reading_database(std::string infname); //read in database peaks


    bool print_result(double inten_check,double cost);    
    bool fit_volume(double);
    bool calculate_matched_ppm();

    bool combinatory_optimization_solver(
                                        std::vector<pattern_match_1d *> const p_pms,
                                        std::vector<std::vector<int>> const indices_of_all_pms,
                                        std::vector<std::vector<double>> const costs_of_all_pms,
                                        std::vector<int> &best_assignment,
                                        double &best_cost);

    bool clash_detection(const std::vector<int> current_assignment_of_each_pms_unsorted, const std::vector<pattern_match_1d *> &p_pms) const;

public:

    int db_index; //index of the database item in the database list (selected subset,not full database list)

    double get_median_experimental_peak_width(); //return the median peak width of the experimental peak list
    double get_median_database_peak_width(); //return the median peak width of the database peak list

    db_match_1d(double,double,double); //cutoff, cutoff_pattern,water_width
    ~db_match_1d();

    /**
     * @brief read in experimental peak list
     * 
     * @param infname file name of the experimental peak list
     * @param ref_correction  reference correction (+this value to all peaks)
     * @param detection_limit 
     * @return true 
     */
    bool peak_reading(std::string infname,double ref_correction,double detection_limit);

    /**
     * @brief write experimental peak list with assignment information
     * (Which peak is assigned to which compound)
     * @param outfname 
     * @return true 
     */
    bool peak_writing(std::string outfname,std::vector<int>);

    bool load_pka_shift();

    /**
     * 1. Load individual simulated peak pos,height and assignment from a file, such as bmse000047_1.txt
     * 2. Simulate a spectrum, using b0,ndata and nzf and window function
     * 3. Run Deep Picker 1D to get peak list (with assignment to each protons). Peaks belong to same proton form a group.
    */
    bool simulate_database(double b0,double spectral_width,int ndata,int nzf,std::string apodization_method, double db_height_cutoff,double R2);

    bool run_match(double lb,double ub,bool b_keep_bdl=false,bool b_keep_covered=true);
    bool save_summary_json(Json::Value &root);
    bool save_summary(std::ofstream &);
    bool copy_and_ref_correction_spectrum(double ref_correction);

    bool try_alternative_match(bool b_keep_bdl=false,bool b_keep_covered=true);
    bool print_matching_compound_index(std::ofstream &);

    bool update_matching_compound_index();

    bool has_clash_and_is_easy();

    /**
     * Readonly reference to min_cost
    */
    double get_min_cost() const;

    bool clear_matching_compound_index();
    bool set_matching_compound_index();
    bool set_pms_update_flag_false();
};

#endif