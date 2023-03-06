
#include <vector>
#include <array>
#include <valarray>
#include <fstream>
#include <iostream>

#include "ceres/ceres.h"
#include "glog/logging.h"


using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include "json/json.h"
#include "commandline.h"
#include "spectrum_fit_3d.h"


int main(int argc, char **argv)
{
    std::cout<<"Last update: Apr.21,2021"<<std::endl;
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("Print help message then quit (no)");

    args.push_back("-f");
    args2.push_back("arguments_vf_3d.txt");
    args3.push_back("Read arguments from file (arguments_vf_3d.txt)");

    args.push_back("-method");
    args2.push_back("voigt");
    args3.push_back("Peak shape: gaussian or voigt (voigt).");

    args.push_back("-rmax");
    args2.push_back("10");
    args3.push_back("Max round of interation in fitting (10))");

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("User defined minimal peak intensity scale for peak picking (5.5)");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("User defined noise floor scale factor for peak picking (3.0)");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value, estimate from sepctrum if input is 0.0 (0.0)");

    args.push_back("-in1");
    args2.push_back("test001.ft3");
    args3.push_back("Name of first input file (test001.ft3)");

    args.push_back("-in2");
    args2.push_back("test256.ft3");
    args3.push_back("Name of last input file (test155.ft3)");

    args.push_back("-peak_in");
    args2.push_back("peaks.list");
    args3.push_back("Name of input peak list file (peaks.list)");

    args.push_back("-out");
    args2.push_back("fitted.list");
    args3.push_back("Output file name in Sparky format. (fitted.list)");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname1,infname2,outfname,peakin_fname;
    
    infname1 = cmdline.query("-in1");
    infname2 = cmdline.query("-in2");
    outfname = cmdline.query("-out");
    peakin_fname = cmdline.query("-peak_in");

    double noise_level=atof(cmdline.query("-noise_level").c_str());
    double user=atof(cmdline.query("-scale").c_str());
    double user2=atof(cmdline.query("-scale2").c_str());

    int rmax=atoi(cmdline.query("-rmax").c_str());

    int i_method=2;
    if(cmdline.query("-method") == "gaussian") i_method=1;
    else if (cmdline.query("-method") == "voigt") i_method=2;
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_fit_3d x;
        x.init_parameters(user,user2,noise_level,1 /*model*/,0 /*zf*/,true /*b_negative*/); //model selection, zf and b_negative are all useless for 3d fitting
        x.init_fitting_parameter(rmax,i_method);
        x.read_for_fitting(infname1,infname2);
        x.peak_reading(peakin_fname);
        x.work();
        x.print_fitted_peaks(outfname);
    }
    return 0;
}
