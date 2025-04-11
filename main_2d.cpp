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
#include "db_match_2d.h"

#include "DeepConfig.h"

/** these are static member varibles of class match_static_varibles
 * Please see the header file for more details
 * At this time, all will be read from spectral file, except n_verbose_db_match_2d and n_acquisition
 * n_zf will be set to x_dim/n_acquisition
 */
int match_static_varibles::n_verbose_db_match_2d; // verbose level

/**
 * Experimental data
 * @param ppm: peak proton chemical shift
 * @param intensity: peak intensity
 * @param ppm_c: peak carbon chemical shift
*/
std::vector<double> match_static_varibles::ppm, match_static_varibles::intensity, match_static_varibles::sigmax, match_static_varibles::gammax;
std::vector<double> match_static_varibles::ppm_c, match_static_varibles::sigmax_c, match_static_varibles::gammax_c; // additional exp data for carbon dimension
std::vector<double> match_static_varibles::effective_width; // effective width and 
std::vector<double>   match_static_varibles::volume; //volume of each peak

/**
 * For mutiplets analysis
*/
std::vector<int> match_static_varibles::doublets_matrix; 
std::vector<std::deque<int>> match_static_varibles::mutiplets;
std::vector<int> match_static_varibles::mutiplet_type;
std::vector<double> match_static_varibles::mutiplet_maxj;
std::vector<int> match_static_varibles::exp_peak_type; 


std::vector<float> match_static_varibles::spectrum;                                                                 // exp data
double match_static_varibles::stop, match_static_varibles::step, match_static_varibles::begin;                      // ppm range of exp data along proton dimension
double match_static_varibles::stop_c, match_static_varibles::step_c, match_static_varibles::begin_c;                // ppm range of exp data along carbon dimension
double match_static_varibles::log_detection_limit;                                                                  // log of detection limit
double match_static_varibles::spectral_width_h, match_static_varibles::spectral_width_c; 
int match_static_varibles::x_dim, match_static_varibles::y_dim; // dimension of the spectrum
int match_static_varibles::x_dim_original; // dimension of the spectrum before zero filling
int match_static_varibles::n_acquisition;                        // number of acquisitions along proton dimension
int match_static_varibles::n_zf;                                 // number of zero filling along proton dimension
float match_static_varibles::b0;                               // magnetic field strength
double match_static_varibles::noise_level_shared;             // noise level. This is shared by all the db_match objects and pattern_match objects.

const double match_static_varibles::cutoff_c_scale_pattern_match = 10.0; //  pattern match cutoff along C is 10 times of that along H

/**
 * Weights for matching
*/
const double match_static_varibles::cost_peak_pattern_locations=10.0;
const double match_static_varibles::cost_peak_pattern_height=0.005;
const double match_static_varibles::cost_overall_shift=5;
const double match_static_varibles::cost_reduce_high_peak=0.15;
const double match_static_varibles::peak_match_scale=40.0; 

const double match_static_varibles::additional_cost_bdl = 6.0;
const double match_static_varibles::additional_cost_covered = 3.0;

int main(int argc, char **argv)
{
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
    std::cout<<"This is a 2D database query program."<<std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1:normal)");

    args.push_back("-f");
    args2.push_back("arguments_db_match_2d.txt");
    args3.push_back("read arguments from file");

    args.push_back("-in");
    args2.push_back("peaks.dat");
    args3.push_back("input experimental peak file name (peaks.dat)");

    args.push_back("-sigmay");
    args2.push_back("1.0");
    args3.push_back("Peak width(Gaussian sigma) along indirect dimension");

    args.push_back("-spe_in");
    args2.push_back("input.ft2");
    args3.push_back("input spectral file name");

    args.push_back("-n_acquisition");
    args2.push_back("0");
    args3.push_back("number of acquisition along proton dimension");

    /**
     * @brief folder and db are used to read database files
     * db is a file containing the name of each database file (one compound per file)
     * folder is the location of database files. We need both of them to read database files.
     */
    args.push_back("-db");
    args2.push_back("colmarm_850/all.list");
    args3.push_back("database item list");

    args.push_back("-folder");
    args2.push_back("colmarm_850");
    args3.push_back("database files location");

    /**
     * Folder to read possible CS instabilities: some proton peaks are very senseitive to environment
     * one compound per file
    */
    args.push_back("-folder_diff");
    args2.push_back("database2d_diff");
    args3.push_back("database files location for CS instabilities");

    /**
     * The program will use both experimental database
     * and a predicted database(only J values are predicted)
    */
    args.push_back("-db_pre");
    args2.push_back("database_2d_red/sdf.list");
    args3.push_back("predicted database item list");

    args.push_back("-folder_pre");
    args2.push_back("database_2d_red");
    args3.push_back("predicted database files location");

    args.push_back("-out");
    args2.push_back("query.json");
    args3.push_back("output file name");

    args.push_back("-cutoff_h");
    args2.push_back("0.03");
    args3.push_back("matching proton chemical shift cutoff");

    args.push_back("-cutoff_Methylene");
    args2.push_back("0.01");
    args3.push_back("matching proton relative chemical shift cutoff for methylene");

    args.push_back("-cutoff_h2");
    args2.push_back("0.008");
    args3.push_back("matching proton chemical shift cutoff for J-split peaks");

    args.push_back("-cutoff_c");
    args2.push_back("0.4");
    args3.push_back("matching carbon chemical shift cutoff");

    args.push_back("-ratio");
    args2.push_back("0.6");
    args3.push_back("matching ratio cutoff");

    args.push_back("-ref");
    args2.push_back("0.0 0.0");
    args3.push_back("reference correction for proton and carbon");



    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    match_static_varibles::n_verbose_db_match_2d = atoi(cmdline.query("-v").c_str());


    std::string infname = cmdline.query("-in");
    std::string spe_infname = cmdline.query("-spe_in");
    std::string dbfname = cmdline.query("-db");
    std::string outfname = cmdline.query("-out");
    

    /**
     * cutoff_h is proton chemical shift cutoff
     * cutoff_Methylene is proton relative chemical shift cutoff for methylene group's 2 protons
     * cutoff_pattern is proton chemical shift cutoff for J-split peaks originating from the same Proton
     * cutoff_c is carbon chemical shift cutoff
     * We don't define cutoff_c_pattern because it is predefined as 10 times of cutoff_pattern
    */
    double cutoff_h = atof(cmdline.query("-cutoff_h").c_str());
    double cutoff_Methylene = atof(cmdline.query("-cutoff_Methylene").c_str());
    double cutoff_pattern = atof(cmdline.query("-cutoff_h2").c_str());
    double cutoff_c = atof(cmdline.query("-cutoff_c").c_str());
    double ratio = atof(cmdline.query("-ratio").c_str());

    double ref_h=0.0;
    double ref_c=0.0;
    std::string ref = cmdline.query("-ref");
    /**
     * ref is a string of two doubles separated by space(s)
    */
    std::istringstream iss(ref);
    iss>>ref_h>>ref_c;

    int n_acquisition = atoi(cmdline.query("-n_acquisition").c_str());

    std::string folder = cmdline.query("-folder");
    std::string db = cmdline.query("-db");
    std::string db_pre = cmdline.query("-db_pre");
    std::string folder_pre = cmdline.query("-folder_pre");
    std::string folder_diff = cmdline.query("-folder_diff");

    /**
     * @brief These two will be written to outfname
     * 
     */
    Json::Value compounds=Json::arrayValue;
    Json::Value mytrace=Json::arrayValue;
    Json::Value exp_information=Json::objectValue;

    /**
     * @brief vector of db_match
     * Each db_match represents one database item
     * saved in a file in folder named "folder"
     */
    std::vector<db_match> db_match_list;

    /**
     * @brief db_match_list_boundary is a int number
     * in db_match_lsit, [0, db_match_list_boundary) are experimental database items
     * and [db_match_list_boundary, db_match_list.size()) are predicted database items
     */
    int db_match_list_boundary;

    /**
     * @brief peak_match_compound is a vector of vector of pair of int
     * peak_match_compound[3]=[{0,1},{7,2},{12,0}] means peak 3 matches compounds 0 peak1, compound 7 peak2, and compound 12 peak0
     * Note 0,7,12 are the index of matched compounds only, not the index of all compounds.
     */
    std::vector<std::vector<std::pair<int,int>>> peak_match_compound; 

    /**
     * @brief read db and save it to db_match_list
     * Example line:
     * HMDB00002_j.cs HMDB00002 bmse000872 428 1,3_Diaminopropane
     * HMDB00002_j.cs is the name of a database file (one compound per file)
     * HMDB00002 is the 2D datasource ID
     * bmse000872 is the 1D datasource ID
     * 428 is the pubchem ID
     * 1,3_Diaminopropane is the name of the compound
     */
    std::string line;
    std::ifstream db_file(db);
    bool b_first = true;
    while (getline(db_file, line))
    {
        /**
         * @brief read one line from db and split it into 5 strings using space as delimiter
         * skip line starting with #
         */
        if (line[0] == '#')
        {
            continue;
        }
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (getline(iss, token, ' '))
        {
            tokens.push_back(token);
        }

        db_match d(cutoff_h, cutoff_c, cutoff_pattern,cutoff_Methylene,ratio);

        if(b_first)
        {
           /**
             * @brief read spectrum from spe_infname and copy it to d.spectrum.
             * We only need to do this for the first database file because they are saved to static variables.
             * ans shared by all the db_match objects and pattern_match objects.
             * 
             * number of acquisitions along proton dimension is set here. 
             * Important: 0 means read from spectral file, field name xdim_original
             */
            d.n_acquisition = n_acquisition;
            d.load_exp_data(infname, spe_infname, ref_h, ref_c);
            /**
             * At this time, we don't run fit, do not simulate spectrum at y direction, so we just set the peak shape to Gaussian with sigma
             * this number will be passed in the output as well
            */
            d.set_database_peak_sigma_c(atof(cmdline.query("-sigmay").c_str()));
            d.analysis_exp_data();
            // d.get_mutiplets_as_compounds(compounds);
            peak_match_compound.resize(match_static_varibles::ppm.size());
            b_first = false;
        }
        
        /**
         * Read J peaks from the database and simulate expected spectrum according to experimental spectrum (ZF, SW, ref, etc.)
        */
        d.read_db(folder + "/" + tokens[0]);
        /**
         * Copy tokens[0] to a new string and remove _j.cs from the end, then add .txt to the end
        */
        std::string diff_file_name = tokens[0].substr(0, tokens[0].find("_j.cs")) + ".txt";
        d.read_db_cs_sensitive(folder_diff + "/" + diff_file_name);
        d.set_addtional_information(tokens[1], tokens[2], tokens[3], tokens[4]);
        db_match_list.emplace_back(d);
    }
    db_file.close();
    db_match_list_boundary = db_match_list.size();

    /**
     * Read in predicted database
    */
    db_file.open(db_pre);
    while (getline(db_file, line))
    {
        if (line[0] == '#')
        {
            continue;
        }

        /**
         * Replace multiple spaces with one space
        */
        line.erase(std::unique(line.begin(), line.end(), [](char a, char b) { return a == ' ' && b == ' '; }), line.end());

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (getline(iss, token, ' '))
        {
            tokens.push_back(token);
        }

        db_match d(cutoff_h, cutoff_c, 0.0 ,cutoff_Methylene,ratio);

        if(b_first) //Need this in case there is no experimental data
        {
            d.n_acquisition = n_acquisition;
            d.load_exp_data(infname, spe_infname, ref_h, ref_c);
            d.analysis_exp_data();
            peak_match_compound.resize(match_static_varibles::ppm.size());
            b_first = false;
        }
        
        d.read_predicted_db_simple(folder_pre + "/" + tokens[0]);
        d.set_addtional_information(tokens[1], tokens[2], tokens[3], tokens[4]);
        db_match_list.emplace_back(d);
    }


    if (match_static_varibles::n_verbose_db_match_2d > 0)
    {
        std::cout << "Need to query " << db_match_list_boundary << " experimental database items and " << db_match_list.size() - db_match_list_boundary << " predicted database items." << std::endl;
    }




    std::vector<int> int_matched; // vector to save b_matched. b_matched is a vector of bool. int_matched is a vector of int.
    int matched_count = 0;        // number of matched compounds
    for (int i = 0; i < db_match_list.size(); i++)
    {
        db_match_list[i].db_index = i;
        bool b_matched = false;

        if(i<db_match_list_boundary)
        {
            b_matched = db_match_list[i].run_match();
        }
        else
        {
            b_matched = db_match_list[i].run_match_predicted();
        }

        int_matched.push_back(int(b_matched));
        if(match_static_varibles::n_verbose_db_match_2d>0) std::cout << i << " (out of " << db_match_list.size() << ") is done." << std::endl;

        /**
         * @brief if b_matched is true, save the result to a json object
         */
        if(b_matched)
        {
            Json::Value one_matched_compound;
            if(i<db_match_list_boundary)
            {
                db_match_list[i].save_result(one_matched_compound,matched_count,peak_match_compound);
            }
            else
            {
                db_match_list[i].save_result_predicted(one_matched_compound,matched_count,peak_match_compound);
            }
            compounds.append(one_matched_compound);
            matched_count++;
        }
    }

    /**
     * Generate a share vector for each peak of each compound. 
     * Then update unique property of each compound in the JSON array compounds
     * compounds[com_index]["peaks"][peak_index]["share"]: array of compound index, which share peaks with this peak of this compound
    */
    for(int i=0;i<peak_match_compound.size();i++)
    {
        for(int j1=0;j1<peak_match_compound[i].size();j1++)
        { 
            int compound_index=peak_match_compound[i][j1].first;
            int peak_index=peak_match_compound[i][j1].second;
            int c=0;
            for(int j2=0;j2<peak_match_compound[i].size();j2++)
            {
                if(j2==j1) continue;
                compounds[compound_index]["peaks"][peak_index]["share"][c]=peak_match_compound[i][j2].first;
                ++c;
            }
        }
    }
    /**
     * @var compounds_shared: same size as compounds
     * compounds_shared[2]=3 means 3 peaks of compound 2 is NOT unique
    */
    std::vector<int> compounds_shared(matched_count,0);
    for(int i=0;i<matched_count;i++)
    {
        for(int j=0;j<compounds[i]["peaks"].size();j++)
        {
            if(compounds[i]["peaks"][j]["share"].size()>0)
            {
                compounds_shared[i]++;
            }
        }
    }

    for(int i=0;i<matched_count;i++)
    {
        compounds[i]["unique"]= std::to_string(compounds_shared[i]) + "/" + compounds[i]["ngood"].asString();
    }


    /**
     * Save some shared information. 
    */
    if(db_match_list.size()>0)
    {
        db_match_list[0].save_peak_result(mytrace,peak_match_compound); //This function only needs to be called once.
        db_match_list[0].save_experiment_information(exp_information);
    }


    /**
     * @brief save the result as a json object to outfname
     */
    Json::Value root;
    root["compound"]=compounds;
    root["mytrace"]=mytrace; 
    root["exp_information"]=exp_information;

    std::ofstream fout(outfname);
    fout<<root;
    fout.close();

    return 0;
}