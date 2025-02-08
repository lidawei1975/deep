#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <set>

#include "json/json.h"
#include "commandline.h"
#include "contour.h"
#include "spectrum_prediction.h"
#include "DeepConfig.h"

#include "dosy.h"
#include "peak_manipulation.h"

int main(int argc, char **argv)
{
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;
    std::cout << "This program will fit a diffussion constant D* from 1D DOSY (pseudo-2D) fitted peaks by VF_1D " << std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1:normal)");

    args.push_back("-f");
    args2.push_back("arguments_dosy_fit.txt");
    args3.push_back("read arguments from file");

    args.push_back("-in");
    args2.push_back("fitted.tab");
    args3.push_back("input fitted peak list with pseudo-2D DOSY peaks");

    args.push_back("-z");
    args2.push_back("z_gradients.txt");
    args3.push_back("input z gradients for each trace");

    args.push_back("-out");
    args2.push_back("result.txt");
    args3.push_back("output fitted D* result in text format");

    args.push_back("-out-tab");
    args2.push_back("fitted_dosy.tab");
    args3.push_back("output fitted peak list in .tab format with D*");

    args.push_back("-baseline");
    args2.push_back("no");
    args3.push_back("include a constant shift in fitting (no)");

    args.push_back("-combine");
    args2.push_back("no");
    args3.push_back("combine peaks in a multiplet for fitting (no)");

    /**
     * Dosy paramters: Diffusion time, length of the gradient and delay time for gradient recovery
    */
    args.push_back("-t_diffusion");
    args2.push_back("80");
    args3.push_back("DOSY diffusion time in ms");

    args.push_back("-g_length");
    args2.push_back("1.6");
    args3.push_back("DOSY gradient length in ms");

    args.push_back("-delay");
    args2.push_back("0.2");
    args3.push_back("DOSY gradient delay for recovery in ms");

    /**
     * Alternatively, use can provide a custome rescale factor for D* fitting
    */
    args.push_back("-rescale");
    args2.push_back("1454078.85082151");
    args3.push_back("rescale factor for Diffusion constant fitting. Negative value means calculating from DOSY parameters");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    bool b_include_baseline = cmdline.query("-baseline")[0] == 'y' || cmdline.query("-baseline")[0] == 'Y';
    bool b_combine_peaks = cmdline.query("-combine")[0] == 'y' || cmdline.query("-combine")[0] == 'Y';

    CDosyFit dosy_fit(b_include_baseline);

    if (dosy_fit.read_z_gradients(cmdline.query("-z")) 
        && dosy_fit.read_fitted_peaks(cmdline.query("-in")) 
        && dosy_fit.fit_diffusion_constant(b_combine_peaks) )
    {
        /**
         * Scale D* if rescale factor is provided
        */
        double rescale_factor = std::stod(cmdline.query("-rescale"));

        /**
         * If rescale factor is negative, calculate from DOSY parameters
        */
        if(rescale_factor <= 0)
        {
            double t_diffusion = std::stod(cmdline.query("-t_diffusion"));
            double g_length = std::stod(cmdline.query("-g_length"));
            double delay = std::stod(cmdline.query("-delay"));
            /**
             * Error if any of the DOSY parameters <=0 or t_diffusion <= g_length/3 + delay/2
            */
            if(t_diffusion <= 0 || g_length <= 0 || delay <= 0)
            {
                std::cerr << "DOSY parameters must be positive, set rescale_factor to 1.0 " << std::endl;
                rescale_factor = 1.0;
            }
            else if( t_diffusion <= g_length/3 + delay/2)
            {
                std::cerr << "DOSY diffusion time must be larger than g_length/3 + delay/2, set rescale_factor to 1.0 " << std::endl;
                rescale_factor = 1.0;
            }
            else
            {
                rescale_factor = (2 * M_PI * 4257.7 * g_length) * (2 * M_PI * 4257.7 * g_length) * (t_diffusion - g_length/3 - delay/2) * 1e-5;
                std::cout<<" Rescale factor is calculated from DOSY parameters: "<<rescale_factor<<std::endl;
            }
        }
        dosy_fit.rescale(rescale_factor);


        /**
         * Get output filenames. Multiple output files are separated by space
        */
        std::string output_files = cmdline.query("-out");

        std::istringstream iss(output_files);
        std::string output_file;
        while (iss >> output_file)
        {
            dosy_fit.write_result(output_file);
        }

        /**
         * Write peak file if output file is in .tab format. 
         * Do not write if filename does not contain .tab
        */
        std::string output_peak_file = cmdline.query("-out-tab");
        if(output_peak_file.find(".tab") != std::string::npos)
        {
            dosy_fit.write_peak_file(output_peak_file);
        }
        
        std::cout << "Fitting D* from 1D DOSY peaks finished." << std::endl;
    }
    else
    {
        std::cerr << "Fitting D* from 1D DOSY peaks failed." << std::endl;
    }

    return 0;
}
