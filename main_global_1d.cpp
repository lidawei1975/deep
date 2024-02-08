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
#include "db_match_global_1d.h"





int main(int argc, char **argv)
{

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1: normal");

    args.push_back("-f");
    args2.push_back("arguments_global_1d.txt");
    args3.push_back("read arguments from file (arguments_global_1d.txt)");

    args.push_back("-in");
    args2.push_back("result.json");
    args3.push_back("Input peak based DB query result");

    args.push_back("-out");
    args2.push_back("opt_result.json");
    args3.push_back("File name for output optimized result in JSON format");

    args.push_back("-spe_in");
    args2.push_back("test.ft1");
    args3.push_back("input spectral file name (peaks.tab)");

    args.push_back("-spe_out");
    args2.push_back("simulated.ft1");
    args3.push_back("File name for simulated spectrum");

    args.push_back("-json_out");
    args2.push_back("yes");
    args3.push_back("Save simulated spectrum in JSON format too (yes)");

    args.push_back("-rmax");
    args2.push_back("10");
    args3.push_back("Maximum rounds of optimization (10)");

    args.push_back("-width_scale");
    args2.push_back("1.0");
    args3.push_back("Apply scale factor for peak width before optimization(1.0)");

    args.push_back("-weighted");
    args2.push_back("yes");
    args3.push_back("Use weighted squared difference in the optimization (yes)");

    args.push_back("-widen");
    args2.push_back("yes");
    args3.push_back("Allow overall widen of all peaks of one compound (yes)");


    args.push_back("-pos_restraint_strength");
    args2.push_back("1.0");
    args3.push_back("Strength of position restraint");

    args.push_back("-relative_pos_restraint_strength");
    args2.push_back("5000.0");
    args3.push_back("Strength of relative position restraint");
    
    args.push_back("-ref");
    args2.push_back("0.0");
    args3.push_back("Reference correction for experiment (ppm)");

    args.push_back("-water");
    args2.push_back("0.9");
    args3.push_back("width of water region in ppm (4.7 +- this number)");

    args.push_back("-db_height_cutoff");
    args2.push_back("0.01");
    args3.push_back("skip database peaks with height lower than this cutoff (0: no cutoff)");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    shared_data_global_1d::n_verbose_global_1d = atoi(cmdline.query("-v").c_str());

#ifdef DEBUG
    shared_data_global_1d::n_verbose_global_1d = 1;
    std::cout << "DEBUG mode, set verbose level to 1" << std::endl;
#endif

    std::string infname = cmdline.query("-in");
    std::string outfname = cmdline.query("-out");
    std::string spe_infname = cmdline.query("-spe_in");
    std::string spe_outfname = cmdline.query("-spe_out");

    bool b_json = cmdline.query("-json_out") == "yes" ? true : false;
    int rmax = atoi(cmdline.query("-rmax").c_str());

    double ref = atof(cmdline.query("-ref").c_str());
    double water_width=std::stod(cmdline.query("-water"));
    double db_height_cutoff=std::stod(cmdline.query("-db_height_cutoff"));

    Functor_shared_static_variables::pos_restrain_strength=std::stod(cmdline.query("-pos_restraint_strength"));
    Functor_shared_static_variables::relative_position_restrain_strength=std::stod(cmdline.query("-relative_pos_restraint_strength"));
    
    
    bool b_weighted = cmdline.query("-weighted") == "yes" || cmdline.query("-weighted") == "Yes" || cmdline.query("-weighted") == "Y" || cmdline.query("-weighted") == "y";
    shared_data_global_1d::b_allow_overal_width_rescale_factor = cmdline.query("-widen") == "yes" || cmdline.query("-widen") == "Yes" || cmdline.query("-widen") == "Y" || cmdline.query("-widen") == "y";

   
    class db_match_global_1d d(water_width,db_height_cutoff);

    /**
     * init(double user_, double user2_, double noise_) as defined in base class, First two are not used in this program
     * noise is set to 0, so that the noise level is estimated from the spectrum
    */
    d.init(5.5,3.5,0.0); 
    /**
     * read spectrum from file and estimate noise level
    */
    d.read_spectrum(spe_infname);
    /**
     * correct ppm of the spectrum
    */
    d.ref_correction_spectrum(ref);

    /**
     * define water region, to be excluded in the optimization
     * This function will convert ppm to index of the spectrum
    */
    d.deal_with_water(); 

    /**
     * load matched compounds from json file, generated by db_match_1d (main_1d.cpp)
     * @par infname: input json file name
     * @par width_scale: scale factor for peak width, because the width of peaks in the database is not always consistent with experimental peaks
    */
    double width_scale=std::stod(cmdline.query("-width_scale"));
    if(d.load_matched_compounds(infname,width_scale)==true)
    {
        /**
         * main optimization function
         **/       
        d.optimization(rmax,b_weighted);   

        /**
         * save optimized result to file, in json format
        */
        d.save_matched_compounds(outfname);

        /**
         * save simulated spectrum to file
         * @par spe_outfname: output spectrum file name
         * @par b_json: save simulated spectrum in json format too, name is simulated_spectrum.json
        */
        d.save_simulated_spectrum(spe_outfname,b_json);
    }
    
    return 0;
}