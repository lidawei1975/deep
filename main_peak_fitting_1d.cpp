
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include <time.h>
#include <sys/time.h>

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include "json/json.h"
#include "commandline.h"
#include "fid_1d.h"
#include "spectrum_fit_1d.h"




#include "DeepConfig.h"

int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
 
    struct timeval time;
    gettimeofday(&time,NULL);
    double time_1= (double)time.tv_sec + (double)time.tv_usec * .000001;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-v");
    args2.push_back("1");
    args3.push_back("verbose level (0: minimal, 1: normal, 2: details)");

    args.push_back("-f");
    args2.push_back("arguments_vf1d.txt");
    args3.push_back("Arguments file");

    args.push_back("-method");
    args2.push_back("voigt");
    args3.push_back("Peak shape: gaussian, lorentz or voigt");  

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("user defined noise level scale factor for peak picking");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("user defined noise level scale factor for peak fitting");

    args.push_back("-noise_level");
    args2.push_back("0.0");
    args3.push_back("Direct set noise level to this value, estimate from sepctrum if input is 0.0");

    args.push_back("-in");
    args2.push_back("test.ft1"); 
    args3.push_back("input file name (test.ft1)");

    args.push_back("-stride");
    args2.push_back("1");
    args3.push_back("stride factor for spectrum (3)");

    args.push_back("-peak_in");
    args2.push_back("peaks.tab");
    args3.push_back("Read peaks list from this file");

    args.push_back("-spectrum-begin");
    args2.push_back("100.0");
    args3.push_back("spectrum extraction begin in ppm (100.0)");

    args.push_back("-spectrum-end");
    args2.push_back("-10.0");
    args3.push_back("spectrum extraction end in ppm (-10.0)");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("Allow negative peaks");

    args.push_back("-doesy");
    args2.push_back("no");
    args3.push_back("doesy experiments for pseudo 2D spectra");

    args.push_back("-z_gradient");
    args2.push_back("z_gradient.txt");
    args3.push_back("z_gradient file for pseudo 2D spectra");

    args.push_back("-out");
    args2.push_back("fitted.tab");
    args3.push_back("output file name");
    
    args.push_back("-maxround");
    args2.push_back("50");
    args3.push_back("maximal rounds in iterative fitting process(50)");

    args.push_back("-combine");
    args2.push_back("0.01");
    args3.push_back("Combine peaks with relative fitting error < this value. Default is 0.01");

    args.push_back("-n_err");
    args2.push_back("0");
    args3.push_back("Round of MC based fitting error estimation. Only run when >=5");

    args.push_back("-zf");
    args2.push_back("1");
    args3.push_back("Time of zero filling. Skipped if -n_err is less than 2");

    args.push_back("-out_json");
    args2.push_back("yes");
    args3.push_back("Output fitted peaks and reconstructed spectrum in json format for further analysis");

    args.push_back("-out_json_fname");
    args2.push_back("recon.json");
    args3.push_back("Output json file name for fitted peaks and reconstructed spectrum");

    args.push_back("-individual");
    args2.push_back("yes");
    args3.push_back("Include individual profile of fitted peaks in json format");

    args.push_back("-recon");
    args2.push_back("yes");
    args3.push_back("Write a reconstructed spectrum file named as input file name + _recon.ft2 (only work when input is in ft2 format)");

    args.push_back("-remove");
    args2.push_back("no");
    args3.push_back("Remove failed peaks in the output file. Default is no");

    args.push_back("-folder");
    args2.push_back("./sim_diff");
    args3.push_back("save reconstructed and differential spectra files in this folder.");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname,outfname,peakfname;
    double user,user2;
    double max_width;
    int maxround;
    int fit_type_flag=2;
    bool b_out_json=false;
    bool b_individual_peaks=false;
    bool b_recon=false;
    int n_stride=1;
    bool b_negative=false;

    maxround=atoi(cmdline.query("-maxround").c_str());
    b_out_json=cmdline.query("-out_json")=="yes" || cmdline.query("-out_json")=="y";
    b_recon=cmdline.query("-recon")=="yes" || cmdline.query("-recon")=="y";
    b_individual_peaks=cmdline.query("-individual")=="yes" || cmdline.query("-individual")=="y";
    b_negative=cmdline.query("-negative")=="yes" || cmdline.query("-negative")=="y";
    

    shared_data_1d::n_verbose=atoi(cmdline.query("-v").c_str()); //set verbose level. Defined in spectrum_fit_1d.cpp
    shared_data_1d::b_dosy=cmdline.query("-doesy")=="yes" || cmdline.query("-doesy")=="y"; //set doesy flag
    shared_data_1d::b_remove_failed_peaks=cmdline.query("-remove")=="yes" || cmdline.query("-remove")=="y";


    double noise_level=stod(cmdline.query("-noise_level"));
    double to_near_cutoff=0.0000001;
    infname = cmdline.query("-in");
    peakfname = cmdline.query("-peak_in");
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());
    n_stride=atoi(cmdline.query("-stride").c_str());


    if(cmdline.query("-method") == "gaussian") fit_type_flag=1;
    else if(cmdline.query("-method") == "voigt") fit_type_flag=2;
    else if(cmdline.query("-method") == "lorentz") fit_type_flag=3;
    else if(cmdline.query("-method").substr(0,1)=="g" || cmdline.query("-method").substr(0,1)=="g") fit_type_flag=1;
    else if(cmdline.query("-method").substr(0,1)=="v" || cmdline.query("-method").substr(0,1)=="V") fit_type_flag=2;
    else if(cmdline.query("-method").substr(0,1)=="l" || cmdline.query("-method").substr(0,1)=="L") fit_type_flag=3;
    else
    {
        std::cout<<"Error: Peak shape not recognized. Use gaussian, lorentz or voigt"<<std::endl;
        return 0;
    }


    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        //for pseudo-2D
        std::cout<<"Input files are "<<infname<<std::endl;
         std::istringstream iss;
        iss.str(infname);
        std::string p;
        std::vector<std::string> file_names;

        while (iss >> p)
        {
            file_names.push_back(p);
        }


        class spectrum_fit_1d x;
        x.init(user,user2,noise_level);
        x.init_fit(fit_type_flag,maxround,to_near_cutoff);
        
        /**
         * Read z_gradient file for pseudo 2D spectra when b_dosy is true
        */
        if(shared_data_1d::b_dosy==true)
        {
            std::string filename=cmdline.query("-z_gradient");
            std::ifstream file(filename);
            if (!file)
            {
                std::cerr << "Cannot open z gradient file " << filename << std::endl;
                shared_data_1d::b_dosy=false;
            }

            double z_gradient;
            while (file >> z_gradient)
            {
                shared_data_1d::z_gradients.push_back(z_gradient);
            }

            file.close();

        }


        int n_err=std::stoi(cmdline.query("-n_err"));
        if(n_err>=2)
        {
            int zf=std::stoi(cmdline.query("-zf"));
            x.init_error(zf,n_err);
        }

        if(x.init_all_spectra(file_names,b_negative)) //read spectra.
        {
            if(x.peak_reading(peakfname))
            {
                double spectrum_begin=std::stod(cmdline.query("-spectrum-begin"));
                double spectrum_end=std::stod(cmdline.query("-spectrum-end"));
                x.peak_fitting(spectrum_begin,spectrum_end); //fitting
            }
            else
            {
                std::cout<<"Error: Failed to read input peaks."<<std::endl;
            }
            x.output(outfname); //output

            if(b_out_json)
            {
                std::string outfname_json=cmdline.query("-out_json_fname");
                std::cout<<"Write json file: " << outfname_json << std::endl;
                x.output_json(outfname_json,b_individual_peaks);
            }

            if(b_recon)
            {
                std::string folder_name=cmdline.query("-folder");
                if(folder_name=="")
                {
                    std::cout<<"Warning: Folder name is empty. Reconstructed spectrum will not be saved."<<std::endl;
                }
                else
                {
                    x.write_recon(folder_name);
                    x.get_data_of_recon(0);
                }
            }
        }
        else
        {
            std::cout<<"Error: Failed to read input spectra."<<std::endl;
        }
    }

    gettimeofday(&time,NULL);
    double time_2= (double)time.tv_sec + (double)time.tv_usec * .000001;

    std::cout<<"Total wall time is "<<time_2-time_1<<" seconds."<<std::endl;
    return 0;
};