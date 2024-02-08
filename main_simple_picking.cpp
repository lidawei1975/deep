
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_simple_picking.h"


#include "DeepConfig.h"

int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("Print help message then quit");

    args.push_back("-f");
    args2.push_back("none");
    args3.push_back("Parguments file");


    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("User defined minimal peak intensity scale for peak picking");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value, estimate from sepctrum if input is 0.0");

    args.push_back("-in");
    args2.push_back("test.ft2");
    args3.push_back("Input file name");

    args.push_back("-out");
    args2.push_back("peaks.tab");
    args3.push_back("Output file names");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("Pick both negative and positive peaks.");
    

    cmdline.init(args, args2, args3);
    if(!cmdline.pharse(argc, argv)) return 1;

    std::string infname,outfname;
    double user;
    bool b_negative;


    b_negative=false;
    b_negative=cmdline.query("-negative")=="yes" || cmdline.query("-negative")=="y";



    double noise_level=atof(cmdline.query("-noise_level").c_str());
    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());

   
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_simple_pick x;
        x.set_scale(user,3.0);
        if(x.init(infname)) //read and zero filling
        {
            if (noise_level>1e-20) { x.set_noise_level(noise_level);}
            x.simple_peak_picking(b_negative); //picking
            x.print_peaks_picking(outfname);
        }  
    }
    return 0;
}
