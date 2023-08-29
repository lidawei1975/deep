
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>


#include "json/json.h"
#include "commandline.h"
#include "spectrum_pick_1d.h"




int main(int argc, char **argv)
{
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("user defined noise level scale factor for peak picking (5.5)");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("user defined noise level scale factor for peak fitting(3.0)");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value, estimate from sepctrum if input is 0.0 (0.0)");

    args.push_back("-in");
    args2.push_back("test.ft1");
    args3.push_back("input file name (test.ft1)");

    args.push_back("-stride");
    args2.push_back("1");
    args3.push_back("stride factor for spectrum (3)");

    
    args.push_back("-out");
    args2.push_back("peaks.tab");
    args3.push_back("output file name");

    args.push_back("-model");
    args2.push_back("1");
    args3.push_back("Model selection for ANN picker, 1: FWHH 6-20, 2: FWHH 4-12");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("Also pick negative peaks (no)");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname,outfname;
    double user,user2;
    double max_width;
    int model_selection;
    int n_stride=1;
    bool b_negative=false;


    model_selection=atoi(cmdline.query("-model").c_str());
    if(model_selection!=1 && model_selection!=2) model_selection=3;

    double noise_level=atof(cmdline.query("-noise_level").c_str());
    infname = cmdline.query("-in");
    
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());
    n_stride=atoi(cmdline.query("-stride").c_str());
    b_negative=cmdline.query("-negative")=="yes";
   
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_pick_1d x;
        x.init(user,user2,noise_level);
        x.init_mod(model_selection);
        if(x.read_spectrum(infname,b_negative)) //read
        {
            x.stride_spectrum(n_stride);
            x.work2(b_negative); //picking 
            x.print_peaks(outfname); //output
        }
        std::cout<<"Done!"<<std::endl;
    }
    return 0;
};