
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <time.h>
#include <sys/time.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>	
#include <Eigen/SparseQR>


#include "kiss_fft.h"
#include "json/json.h"
#include "ldw_math.h"
#include "commandline.h"
#include "fid_1d.h"
#include "spectrum_baseline_1d.h"



int main(int argc, char **argv)
{

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-v");
    args2.push_back("1");
    args3.push_back("verbose level: 0 (minimal) 1 (normal)");

    args.push_back("-in");
    args2.push_back("test.ft1");
    args3.push_back("input spectral file names.");

    args.push_back("-baseline_in");
    args2.push_back("none");
    args3.push_back("input baseline file names. If provided, apply it to the input spectra and skip the baseline calculation.");

    args.push_back("-a");
    args2.push_back("1e14");
    args3.push_back("baseline smoothness parameter. Default is 1e14.");
    
    args.push_back("-b");
    args2.push_back("1.25");
    args3.push_back("baseline below signal parameter. Default is 1.25.");

    args.push_back("-n_water");
    args2.push_back("50");
    args3.push_back("number of data points along both side of center to be excluded for baseline calculation. Default is 50.");
    
    args.push_back("-out");
    args2.push_back("test_baseline.ft1");
    args3.push_back("output file names for baseline corrected spectra in pipe format.");

    args.push_back("-baseline_out");
    args2.push_back("baseline.txt");
    args3.push_back("output file names for baseline in binary of text (only when filename extension is txt) format.");

    /**
     * At this time, these two methods don't give identical results. To be investigated later.
    */
    args.push_back("-method");
    args2.push_back("0");
    args3.push_back("method: 0 (normal) 1 (sparse) matrix based algorithm");

   
    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();



    if (cmdline.query("-h") != "yes")
    {
        std::string infname=cmdline.query("-in");
        std::string outfname=cmdline.query("-out");
        std::string outfname_baseline=cmdline.query("-baseline_out");

        class spectrum_baseline_1d x;
        x.n_verbose=std::stoi(cmdline.query("-v"));
        x.init(10.0,3.0,0.0); //user_scale,user_scale2,noise_level. First two are not used in this program.
        x.read_spectrum(infname);

        std::string baseline_infname=cmdline.query("-baseline_in");

        /**
         * if baseline_infname is none or no, then do the baseline calculation. Otherwise, read the baseline from the file.
         * convert to lower case first.
        */
        std::transform(baseline_infname.begin(), baseline_infname.end(), baseline_infname.begin(), ::tolower);

        if(baseline_infname=="none" || baseline_infname=="no")
        {
            double a0=std::stod(cmdline.query("-a")); //baseline smoothness parameter
            double b0=std::stod(cmdline.query("-b")); //baseline below signal parameter
            int n_water=std::stoi(cmdline.query("-n_water")); //number of data points along both side of center to be excluded for baseline calculation

            int method=std::stoi(cmdline.query("-method"));

            if(method!=1)
            {
                method=0;
                std::cout<<"method is set to 0 (normal) matrix based algorithm"<<std::endl;
            }
            else
            {
                std::cout<<"method is set to 1 (sparse) matrix based algorithm"<<std::endl;
            }
            x.work(a0,b0,n_water,method,outfname_baseline);
        }
        else
        {
            std::cout<<"read baseline from file: "<<baseline_infname<<std::endl;
            x.read_baseline(baseline_infname);
        }

        /**
         * write the baseline corrected spectra to the output file.
        */
        x.write_spectrum(outfname); 
    }

    return 0;
}