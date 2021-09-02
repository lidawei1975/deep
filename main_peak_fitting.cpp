#include <complex>
#include <vector>
#include <valarray>
#include <fstream>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "omp.h"


#include "commandline.h"
#include "spectrum_fit.h"


int main(int argc, char **argv)
{
    
    std::cout<<"Last update: Apr.21,2021"<<std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");
    
    args.push_back("-method");
    args2.push_back("voigt");
    args3.push_back("Peak shape: gaussian or voigt (voigt).");   

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("user defined noise scale factor for minimal peak intensity (5.5)");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("user defined noise floor scale factor (3.0)");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value. Noise level will be estimated from sepctrum if input is 0.0 (0.0)");


    args.push_back("-in");
    args2.push_back("input.ft2");
    args3.push_back("input spectral file names. Multiple files should be seprated by space");

    // args.push_back("-zf");
    // args2.push_back("0");
    // args3.push_back("addtional fold of zero filling, 0 or 1 (0)");

    args.push_back("-peak_in");
    args2.push_back("peaks_picked.tab");
    args3.push_back("Read peaks list from this file. Support .tab or .list format (input_peaks.tab)");
    
    args.push_back("-out");
    args2.push_back("peaks.tab");
    args3.push_back("output fitted peaks file name. Multiple files should be seprated by space. Support .tab or .list format");

    args.push_back("-recon");
    args2.push_back("yes");
    args3.push_back("save reconstructed and differential spectra files in pipe format? (yes)");

    args.push_back("-maxround");
    args2.push_back("50");
    args3.push_back("maximal rounds in iterative fitting process(50)");

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
    cmdline.pharse(argc, argv);

    std::string infname,outfname,peak_file;

    float water;
    int maxround;
    int zf;
    double user,user2;
    double wx,wy;
    double smooth;
    double too_near_cutoff;
    double noise_level;
    bool b_read_success=false;
    
    zf=atoi(cmdline.query("-zf").c_str());
    if(zf!=1) zf=0;
    noise_level=atof(cmdline.query("-noise_level").c_str());
    peak_file=cmdline.query("-peak_in");
    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");
    maxround=atoi(cmdline.query("-maxround").c_str());
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());
    wx=atof(cmdline.query("-wx").c_str());
    wy=atof(cmdline.query("-wy").c_str());
    too_near_cutoff=0.1;

   
    int i_method=2;
    if(cmdline.query("-method") == "gaussian") i_method=1;
    else if (cmdline.query("-method") == "voigt") i_method=2;

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_fit x;
        std::cout<<"Input files are "<<infname<<std::endl<<std::endl;
        std::istringstream iss;
        iss.str(infname);
        std::string p;
        std::vector<std::string> file_names;

        while (iss >> p)
        {
            file_names.push_back(p);
        }
        
        x.initflags_fit(maxround,too_near_cutoff,i_method,zf);
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

            if (x.peak_reading(peak_file) && x.peak_fitting())
            {
                x.print_peaks(outfname);
                std::cout<<"Finish print peaks."<<std::endl;
                if(cmdline.query("-recon")=="yes")
                {
                    x.generate_recon_and_diff_spectrum();
                    std::cout<<"Finish generate recon and diff spectral files."<<std::endl;
                }
            }
        }
        x.clear_memory();
    }
    return 0;
}
