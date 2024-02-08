// #include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <set>

#include "json/json.h"
#include "commandline.h"
#include "hungary.h"
#include "db_match_1d.h"

#include "DeepConfig.h"

/** these are static member varibles of class match_1d_static_varibles
 * Please see the header file for more details
 */
// peak rel
std::vector<double> match_1d_static_varibles::ppm, match_1d_static_varibles::intensity_original;
std::vector<double> match_1d_static_varibles::volume, match_1d_static_varibles::effective_width, match_1d_static_varibles::confident_level;
std::vector<int> match_1d_static_varibles::backgroud_flag;

// spectrum related
std::vector<float> match_1d_static_varibles::spectrum;
double match_1d_static_varibles::max_value_of_exp_spectrum;
double match_1d_static_varibles::stop, match_1d_static_varibles::step, match_1d_static_varibles::begin;
std::vector<std::vector<int>> match_1d_static_varibles::matching_compound_index;
std::vector<std::vector<int>> match_1d_static_varibles::matching_compound_peak_index;
std::vector<std::vector<double>> match_1d_static_varibles::matching_compound_height;
double match_1d_static_varibles::log_detection_limit;

// other
int match_1d_static_varibles::n_verbose_match_1d = 0;

// addtional exp peak information that is currently not used. but will be copied to the output peak list with assignment information.
std::vector<double> match_1d_static_varibles::sigmax, match_1d_static_varibles::gammax;
std::vector<double> match_1d_static_varibles::pos, match_1d_static_varibles::delta_intensity, match_1d_static_varibles::peak_integral, match_1d_static_varibles::peak_width;
std::vector<int> match_1d_static_varibles::nround;
std::vector<std::string> match_1d_static_varibles::user_comments;

/**
 * cost_peak_pattern_locations: cost for peak pattern locations matching. sum of the difference of peak locations * this number
 * cost_peak_pattern_height: cost for peak pattern height matching. RMSD of peak heights * this number
 * cost_overall_shift: cost for overall shift. overal shift of each peak group * this number. When cutoff is not 0.02, actual cost will be adjusted accordingly.
 * cost_reduce_high_peak: cost for reducing high peaks. natural-log(intensity) * this number. Notice the negative sign when applied in the code.
 * cost_reduce_high_peak * log(2) = 0.1: double the height of experimental peak will reduce cost by 0.1
 * Worst peak pattern match has a cost of 0.002 (ppm) * 15 = 0.03
 * Worst peak overall shift has a cost of 0.02 (ppm) *1 = 0.02
 * When peak height 5 times high, it reduce the cost by log(5)*0.02 = 0.032
 * Peak height off by 4 times will cost 0.005 * 4 = 0.02
 *
 * peak_match_scale =100: scale all above up, so that intensity mis-match part is less important
 * peak_match_scale also applied to additional_cost_covered and additional_cost_bdl, because they are also related  to peak pattern matching part.
 *
 */

const double match_1d_static_varibles::cost_peak_pattern_locations = 15.0;
const double match_1d_static_varibles::cost_peak_pattern_height = 0.005;
const double match_1d_static_varibles::cost_overall_shift = 0.02;
const double match_1d_static_varibles::cost_reduce_high_peak = 0.15;
const double match_1d_static_varibles::peak_match_scale = 40.0;

const double match_1d_static_varibles::additional_cost_bdl = 2.0;
const double match_1d_static_varibles::additional_cost_covered = 0.2;

const int match_1d_static_varibles::n_max_peak_number = 10;

int main(int argc, char **argv)
{
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1:normal)");

    args.push_back("-f");
    args2.push_back("arguments_db_1d.txt");
    args3.push_back("read arguments from file (arguments.txt)");

    /**
     * @brief input file name for experimental peaks (from DP and VF)
     */
    args.push_back("-in");
    args2.push_back("experiment.tab");
    args3.push_back("input peaks file name");

    args.push_back("-peak_out");
    args2.push_back("experiment_assignment.tab");
    args3.push_back("Output peaks with assignment information file name");

    args.push_back("-spe_in");
    args2.push_back("test.ft1");
    args3.push_back("input spectral file name (peaks.tab)");

    /**
     * Read a output from previous run of this program.
     * This file contains the result of the matching of the experimental peaks to the database peaks
     * of a group of database items.
     */
    args.push_back("-global-opt");
    args2.push_back("yes");
    args3.push_back("Run final global optimization while considering all database items simultaneously");

    args.push_back("-db-all");
    args2.push_back("All.list");
    args3.push_back("List of all database items");

    args.push_back("-db");
    args2.push_back("database.list");
    args3.push_back("database item ID list (subset of All.list)");

    args.push_back("-folder");
    args2.push_back("database-500");
    args3.push_back("database files location (database-500)");

    args.push_back("-pka_folder");
    args2.push_back("pka_shift");
    args3.push_back("pka shift files location (pka_shift)");

    args.push_back("-out");
    args2.push_back("result.txt result.json");
    args3.push_back("output file names (result.txt result.json)");

    args.push_back("-cutoff_h");
    args2.push_back("0.02");
    args3.push_back("matching mean CS difference cutoff");

    args.push_back("-cutoff_h2");
    args2.push_back("0.003");
    args3.push_back("matching mean CS difference cutoff");

    args.push_back("-ref");
    args2.push_back("0.03");
    args3.push_back("Reference correction for experiments (ppm)");

    args.push_back("-keep_bdl");
    args2.push_back("no");
    args3.push_back("Keep bdl compounds in the output (no)");

    args.push_back("-keep_covered");
    args2.push_back("yes");
    args3.push_back("Keep fully covered compounds in the output (yes)");

    args.push_back("-b0");
    args2.push_back("850.0");
    args3.push_back("B0 field strength (850.0), required for spin simulation");

    args.push_back("-ndata");
    args2.push_back("131072");
    args3.push_back("Number of data points (131072), required for spin simulation");

    args.push_back("-nzf");
    args2.push_back("2");
    args3.push_back("Times of zero filling (2), required for spin simulation");

    args.push_back("-apod");
    args2.push_back("kaiser 0.5 0.896 3.684");
    args3.push_back("apodization function name and parameters, separated by space(s), required for spin simulation");

    args.push_back("-spectral_width");
    args2.push_back("15.9800");
    args3.push_back("spectral width in ppm (15.9800), required for spin simulation");

    args.push_back("-r2");
    args2.push_back("3.0");
    args3.push_back("R2 relaxation rate (3.0), required for spin simulation");

    args.push_back("-detection_limit");
    args2.push_back("0");
    args3.push_back("Lowest peak height that can be detected (0: estimate from input peak list)");

    args.push_back("-water");
    args2.push_back("0.5");
    args3.push_back("width of water region in ppm (4.7 +- this number)");

    args.push_back("-db_height_cutoff");
    args2.push_back("0.1");
    args3.push_back("skip database peaks with height lower than this cutoff (0: no cutoff)");

    args.push_back("-width-factor");
    args2.push_back("1.0");
    args3.push_back("factor to multiply the width of database peaks, will not use but pass to the output file");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    match_1d_static_varibles::n_verbose_match_1d = std::stoi(cmdline.query("-v"));
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    std::string infname = cmdline.query("-in");
    std::string outfname_string = cmdline.query("-out");
    std::string spe_infname = cmdline.query("-spe_in");
    std::string peak_outfname = cmdline.query("-peak_out");

    // separate outfname_string into a vector of strings, using space as a delimiter
    std::vector<std::string> outfnames;
    std::stringstream ss(outfname_string);
    std::string outfname;
    while (ss >> outfname)
    {
        outfnames.push_back(outfname);
    }

    double cutoff_h = atof(cmdline.query("-cutoff_h").c_str());
    double cutoff_h2 = atof(cmdline.query("-cutoff_h2").c_str());
    double ref = atof(cmdline.query("-ref").c_str());
    double detection_limit = atof(cmdline.query("-detection_limit").c_str());
    double water_width = std::stod(cmdline.query("-water"));
    double db_height_cutoff = std::stod(cmdline.query("-db_height_cutoff"));
    bool b_keep_bdl = (cmdline.query("-keep_bdl") == "yes") ? true : false;
    bool b_keep_covered = (cmdline.query("-keep_covered") == "yes") ? true : false;

    double b0 = atof(cmdline.query("-b0").c_str());
    double spectral_width = atof(cmdline.query("-spectral_width").c_str());
    int ndata = atoi(cmdline.query("-ndata").c_str());
    int nzf = atoi(cmdline.query("-nzf").c_str());
    std::string apodization_string = cmdline.query("-apod");

    double r2 = atof(cmdline.query("-r2").c_str());

    if (b_keep_bdl == true)
    {
        b_keep_covered = true; // if b_keep_bdl is true, b_keep_covered must be true. It doesn't make sense because bdl peaks are always covered.
    }

    /**
     * Start code to run -db-id or -db
     */
    std::string folder = cmdline.query("-folder");
    std::string pka_folder = cmdline.query("-pka_folder");
    std::string db_all = cmdline.query("-db-all");
    std::vector<db_match_1d> db_match_list;

    /**
     * @brief read db_all and save it to db_match_list
     */
    std::string line;
    std::ifstream db_all_file(db_all);
    while (getline(db_all_file, line))
    {
        db_match_1d d(cutoff_h, cutoff_h2, water_width);
        d.process_input(line);
        db_match_list.push_back(d);
    }
    db_all_file.close();

    /**
     * @brief read db_id and save it to db_match_list
     */
    std::vector<std::string> db_ids;
    std::vector<double> lbs, ubs;
    std::string db_id;

    /**
     * Read from -db, which is a file with predifined list of db_ids (subset of db_all)
     */
    std::string db = cmdline.query("-db");
    std::ifstream db_file(db);
    while (getline(db_file, db_id))
    {
        /**
         * Seperate db_id into fields, seperated by space(s)
         */
        std::stringstream ss(db_id);
        std::string field;
        std::vector<std::string> fields;
        while (ss >> field)
        {
            fields.push_back(field);
        }

        /**
         * Skip empty lines
         */
        if (fields.size() == 0)
        {
            continue;
        }

        /**
         * Remove .tab from fields[0] if fields[0] ends with .tab
        */
        if (fields[0].length()>4 &&  fields[0].substr(fields[0].size() - 4, 4) == ".tab")
        {
            fields[0] = fields[0].substr(0, fields[0].size() - 4);
        }

        /**
         * 3 fields: database item id, lower bound, upper bound
         */
        if (fields.size() == 3)
        {
            db_ids.push_back(fields[0]);
            lbs.push_back(atof(fields[1].c_str()));
            ubs.push_back(atof(fields[2].c_str()));
        }
        /**
         * All others, suppose first field is database item id, lower bound is 0.0, upper bound is 1.0
         */
        else
        {
            db_ids.push_back(fields[0]);
            lbs.push_back(0.0);
            ubs.push_back(1.0);
        }
    }

    /**
     * db_match_list contains all the database items in db_all
     * db_ids contains the database items that we need to query
     * We need to find the index of the database item in db_match_list
     */
    std::vector<int> ndx_db; // vector to save the index of the matched db_id item in db_match_list
    ndx_db.clear();
    for (int m = 0; m < db_ids.size(); m++)
    {
        db_id = db_ids[m];
        /**
         * each line in db_file is a database file id
         * User provided. we need to find the index of the database item in db_match_list
         */
        for (int i = 0; i < db_match_list.size(); i++)
        {
            /**
             * @brief search db_file_id in db_match_list, which is all db items in db_all
             * We then save the index of the matched db item to ndx_db
             */
            if (db_match_list[i].search_by_id_exact(db_id) == true)
            {
                ndx_db.push_back(i);
                break;
            }
        }
    }

    /**
     * We need to remvoe duplicate items in ndx_db
     * Because each database item may crash if it run twice.
     * Remove duplicate corresponding db_ids, lbs and ubs too.
     */
    std::vector<int> duplication_flag(ndx_db.size(), -1); // -1 means not duplicated, otherwise it is the index of the duplicated item
    for (int i = 0; i < ndx_db.size(); i++)
    {
        if (duplication_flag[i] >= 0)
        {
            continue;
        }
        for (int j = i + 1; j < ndx_db.size(); j++)
        {
            if (duplication_flag[j] >= 0)
            {
                continue;
            }
            if (ndx_db[i] == ndx_db[j])
            {
                duplication_flag[j] = i;
            }
        }
    }

    /**
     * @brief remove duplicated items in ndx_db,db_ids,lbs,ubs, using duplication_flag
     */
    for (int i = ndx_db.size() - 1; i >= 0; i--)
    {
        if (duplication_flag[i] >= 0)
        {
            ndx_db.erase(ndx_db.begin() + i);
            db_ids.erase(db_ids.begin() + i);
            lbs.erase(lbs.begin() + i);
            ubs.erase(ubs.begin() + i);
        }
    }

    if (match_1d_static_varibles::n_verbose_match_1d >= 0)
    {
        std::cout << "Need to query " << ndx_db.size() << " database items." << std::endl;
    }

    std::vector<int> int_matched; // vector to save b_matched. b_matched is a vector of bool. int_matched is a vector of int.
    for (int i0 = 0; i0 < ndx_db.size(); i0++)
    {
        int i = ndx_db[i0];
        if (match_1d_static_varibles::n_verbose_match_1d >= 0)
        {
            std::cout << "db ID is: " << db_match_list[i].get_id() << std::endl;
        }
        if (i0 == 0)
        { /**
           * @brief read spectrum from infname and copy it to d.spectrum. Apply reference correction to d.spectrum.
           * @brief read peaks from infname and save them then apply reference correction
           * We only need to do this for the first database file because they are saved to static variables.
           * ans shared by all the db_match_1d objects and pattern_match_1d objects.
           * Please line 20 of this file for all the static variables.
           */
            db_match_list[i].init(5.5, 3.5, 0.0);
            db_match_list[i].read_spectrum(spe_infname);
            db_match_list[i].copy_and_ref_correction_spectrum(ref); // set up verbose level for peak fitter (to run on-the-fly DB peak simulation)
            db_match_list[i].peak_reading(infname, ref, detection_limit);
        }
        db_match_list[i].db_index = i0;
        db_match_list[i].folder = folder; // folder is the folder to load all the database files (.tab or .txt, depending on options)
        db_match_list[i].pka_folder = pka_folder;

        /**
         * @param b_load: true if database is loaded successfully, or simulated successfully
         */
        bool b_load;
        b_load = db_match_list[i].simulate_database(b0, spectral_width, ndata, nzf, apodization_string, db_height_cutoff, r2);
        db_match_list[i].load_pka_shift();

        /**
         * Load pka shift from database file.
         */

        if (b_load == false)
        {
            continue; // empty database or error, skip it
        }

        /**
         * @brief run_match() is the main function to run the matching
         * It returns true if the database item is matched
         * or false if the database item is not matched
         * When b_keep_bdl is true and b_keep_covered are both true, all database items will be matched.
         */
        db_match_list[i].set_pms_update_flag_false();
        bool b_matched = db_match_list[i].run_match(lbs[i0], ubs[i0], b_keep_bdl, b_keep_covered);
        int_matched.push_back(int(b_matched));


        /**
         * Seperated output files for each database item
        */
        std::ofstream fout(db_ids[i0] + "_result.json"); // db_ids[i0] is same as db_match_list[i].get_id(); 
        if(b_matched == true)
        {
            Json::Value root;
            db_match_list[i].save_summary_json(root);
            fout << root;
        }
        else
        {
            Json::Value root;
            root["base_name"] = db_match_list[i].get_id();   
            root["v_fitted"] = 0.0; //This is a flag to indicate that the database item is not matched
        }
        fout.close();

        if (match_1d_static_varibles::n_verbose_match_1d >= 0)
        {
            std::cout << i0 << " (out of " << ndx_db.size() << ") is done." << std::endl;
        }
    }

    /**
     * Run global optimization for selected easy database items
    */
    bool b_global_opt = cmdline.query("-global-opt") == "yes" || cmdline.query("-global-opt") == "Yes" || cmdline.query("-global-opt") == "Y" || cmdline.query("-global-opt") == "y";
    if (b_global_opt == true)
    {
        std::vector<int> ndx_updated;
        /**
         * Step 1, Find experimental peaks that have been mapped to multiple database items
         * db_match_list[*].matching_compound_index has the same size as experimenal peak list (ppm, intensity, etc)
         * db_match_list[3] =[], means not matched to any database item
         * db_match_list[3] =[0,2], means matched to database item 0 and 2 (index is WRT ndx_db, not the original db_match_list)
         */
        for (int i0 = 0; i0 < ndx_db.size(); i0++)
        {
            int i = ndx_db[i0];
            if(i0==0)
            {
                /**
                 * Work on shared static variables. Need to only need to do this once.
                 */
                db_match_list[i].clear_matching_compound_index();
            }

            /**
             * Skip the database item that is not matched
             */
            if (int_matched[i0] == 1)
            {
                db_match_list[i].update_matching_compound_index();
            } 
        }

        std::ofstream fout_debug("clash_debug.txt");
        db_match_list[0].print_matching_compound_index(fout_debug);
        fout_debug.close();

        /**
         * Step 2, find all db items that have clashing peaks and are easy to fix
        */
        for (int i0 = 0; i0 < ndx_db.size(); i0++)
        {
            int i = ndx_db[i0];
            if (int_matched[i0] == 1 && db_match_list[i].has_clash_and_is_easy() == true)
            {
                ndx_updated.push_back(i0);    
            }
        }
        
        /**
         * Write a webapp summary josn file for all updated database items
         * so that the web server know which database items have been updated 
        */
        std::ofstream fout_json("webserver-updated.json");
        Json::Value root2;
        for (int i1 = 0; i1 < ndx_updated.size(); i1++)
        {
            root2[i1]=db_ids[ndx_updated[i1]];
        }
        fout_json << root2;
        fout_json.close();

        /**
         * Step 3, greedy algorithm, try to adjust matching of some easy database items, one at a time, instead of global.
        */
        for (int i1 = 0; i1 < ndx_updated.size(); i1++)
        {
            int i = ndx_db[ndx_updated[i1]];

            double current_min_cost = db_match_list[i].get_min_cost();
            db_match_1d d = db_match_list[i];
            d.try_alternative_match();
            double new_min_cost = d.get_min_cost();
            db_match_list[i] = d;
            /**
             * print out the new matching, even if it is not updated because web server want to know.
             */
            std::ofstream fout(db_ids[ndx_updated[i1]] + "_result2.json");
            Json::Value root;
            db_match_list[i].save_summary_json(root);
            fout << root;
            fout.close();
        }
    }


    /**
     * @brief get median of all median database peak width of all database items
     * median_db_peak_width
     */
    std::vector<double> database_peak_width;
    for (int i0 = 0; i0 < ndx_db.size(); i0++)
    {
        if (int_matched[i0] == 0)
        {
            continue; // skip the database item that is not matched
        }
        int i = ndx_db[i0];
        database_peak_width.push_back(db_match_list[i].get_median_database_peak_width());
    }
    std::sort(database_peak_width.begin(), database_peak_width.end());
    double median_database_peak_width = 0.001;
    if (database_peak_width.size() > 0)
    {
        median_database_peak_width = database_peak_width[database_peak_width.size() / 2];
    }
    double median_experimental_peak_width = db_match_list[0].get_median_experimental_peak_width();

    /**
     * Write a webapp summary josn file
    */
    std::ofstream fout_json("webserver-result.json");
    Json::Value root2;
    root2["cutoff_h"] = cutoff_h;
    root2["cutoff_h2"] = cutoff_h2;
    root2["ref"] = ref;
    root2["water_width"] = water_width;
    root2["median_database_peak_width"] = median_database_peak_width;
    root2["median_experimental_peak_width"] = median_experimental_peak_width;
    fout_json << root2;
    fout_json.close();

    /**
     * @brief write the results to output files
     * The user can specify multiple output files. The output files must be .txt.
     * Files with other types will be ignored.
     * Only first one will be used.
     */

    std::ofstream fout;               // output file stream for .txt file
    for (int j = 0; j < outfnames.size(); j++)
    {
        // if outfnames[j] ends with .txt, write to it
        if ((!fout.is_open()) && outfnames[j].substr(outfnames[j].size() - 4, 4) == ".txt")
        {
            fout.open(outfnames[j]);
        }
    }
    if (fout.is_open())
    {
        int counter = 0;
        for (int i0 = 0; i0 < ndx_db.size(); i0++)
        {
            // if int_matched[i0]==0, skip this database item
            if (int_matched[i0] == 0)
            {
                continue;
            }
            db_match_list[ndx_db[i0]].save_summary(fout);//save_summary is a function of db_match_1d to write in txt format
            fout << std::endl;
        }
        fout.close();
    }


    return 0;
}