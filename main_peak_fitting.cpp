#include <complex>
#include <vector>
#include <valarray>
#include <fstream>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "omp.h"


#include "commandline.h"
#include "spectrum_fit.h"

int shared_data::n_verbose=1;

int main(int argc, char **argv)
{
    
    std::cout<<"Last update: Oct. 2021"<<std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-v");
    args2.push_back("1");
    args3.push_back("verbose level (0: minimal, 1:normal)");
    
    args.push_back("-f");
    args2.push_back("none");
    args3.push_back("Parguments file");

    args.push_back("-method");
    args2.push_back("gaussian");
    args3.push_back("Peak shape: gaussian or voigt.");   

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("user defined noise scale factor for minimal peak intensity");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("user defined noise floor scale factor");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value. Noise level will be estimated from sepctrum if input is 0.0");

    args.push_back("-in");
    args2.push_back("04.csv");
    args3.push_back("input spectral file names. Multiple files should be seprated by space");

    args.push_back("-peak_in");
    args2.push_back(" picked_peaks.json");
    args3.push_back("Read peaks list from this file. Support .tab or .list format");
    
    args.push_back("-out");
    args2.push_back("fitted.tab");
    args3.push_back("output fitted peaks file name. Multiple files should be seprated by space. Support .tab or .list format");

    args.push_back("-recon");
    args2.push_back("yes");
    args3.push_back("save reconstructed and differential spectra files in pipe format? (yes)");

    args.push_back("-folder");
    args2.push_back("./sim_diff");
    args3.push_back("save reconstructed and differential spectra files in this folder.");

    args.push_back("-maxround");
    args2.push_back("100");
    args3.push_back("maximal rounds in iterative fitting process(50)");

    args.push_back("-combine");
    args2.push_back("0.04");
    args3.push_back("Cutoff to combine tightly overlapping peaks, 0.04 (0.08,0.12) for high(medium,low) quality spectrum");

    args.push_back("-wx");
    args2.push_back("0.0");
    args3.push_back("maximal FWHH in direct dimension in ppm, 0 means using information in the input peaks (0.0)");

    args.push_back("-wy");
    args2.push_back("0.0");
    args3.push_back("maximal FWHH in indirect dimension in ppm, 0 means using information in the input peaks  (0.0)");

    args.push_back("-n_err");
    args2.push_back("0");
    args3.push_back("Round of addtional fitting with artifact noise for the estimation of fitting errors");

    args.push_back("-zf1");
    args2.push_back("1");
    args3.push_back("Time of zero filling along direct dimension. Skipped if -n_err is less than 5");
    
    args.push_back("-zf2");
    args2.push_back("1");
    args3.push_back("Time of zero filling along indirect dimension. Skipped if -n_err is less than 5");
    


    cmdline.init(args, args2, args3);
    if(!cmdline.pharse(argc, argv)) return 1;

    std::string infname,outfname,peak_file;

    float water;
    int maxround;
    double user,user2;
    double wx,wy;
    double smooth;
    double too_near_cutoff;
    double noise_level;
    double removal_cutoff;
    bool b_read_success=false;
    
    noise_level=atof(cmdline.query("-noise_level").c_str());
    peak_file=cmdline.query("-peak_in");
    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");
    maxround=atoi(cmdline.query("-maxround").c_str());
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());
    removal_cutoff=atof(cmdline.query("-combine").c_str());
    wx=atof(cmdline.query("-wx").c_str());
    wy=atof(cmdline.query("-wy").c_str());
    too_near_cutoff=0.1;

    shared_data::n_verbose=atoi(cmdline.query("-v").c_str());

   
    int i_method=2;
    if(cmdline.query("-method") == "gaussian") i_method=1;
    else if(cmdline.query("-method").rfind("g", 0) == 0) i_method=1;
    else if(cmdline.query("-method").rfind("G", 0) == 0) i_method=1;
    else if (cmdline.query("-method") == "voigt") i_method=2;
    else if (cmdline.query("-method").rfind("v", 0) == 0) i_method=2;
    else if (cmdline.query("-method").rfind("V", 0) == 0) i_method=2;
    else 
    {
        std::cout<<"Unrecognized fitting line shape. Exit."<<std::endl;
        return 1;
    }

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_fit x;
        std::cout<<"Input files are "<<infname<<std::endl;
        if(i_method==1) std::cout<<"Fitting line shape is Gaussian"<<std::endl;
        else std::cout<<"Fitting line shape is Voigt"<<std::endl;
        std::cout<<std::endl;
        
        std::istringstream iss;
        iss.str(infname);
        std::string p;
        std::vector<std::string> file_names;

        while (iss >> p)
        {
            file_names.push_back(p);
        }
        
        x.initflags_fit(maxround,removal_cutoff,too_near_cutoff,i_method,0);
        x.set_scale(user,user2);

        if (x.init_all_spectra(file_names))
        {
            if (wx > 0.0 || wy > 0.0)
            {
                x.set_peak_width(wx, wy);
            }
            if (noise_level>1e-20) { x.set_noise_level(noise_level);}

            if(std::stoi(cmdline.query("-n_err"))>=5)
            {
                x.init_error(1,std::stoi(cmdline.query("-zf1")),std::stoi(cmdline.query("-zf2")),std::stoi(cmdline.query("-n_err")));
            }

            if (x.peak_reading(peak_file))
            {
                 x.peak_fitting();
            }
            x.print_peaks(outfname,cmdline.query("-recon")=="yes",cmdline.query("-folder"));
            std::cout<<"Finish print peaks."<<std::endl;
            // x.clear_memory();
        }
        else
        {
            std::cout<<"Canot read any spectrum file."<<std::endl;
        }
        
    }
    return 0;
}
