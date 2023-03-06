
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "json/json.h"
#include "commandline.h"
#include "spectrum_pick_3d.h"



int main(int argc, char **argv)
{
    std::cout<<"Last update: Feb. ,2023"<<std::endl;
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("Print help message then quit (no)");

    args.push_back("-f");
    args2.push_back("arguments_dp_3d.txt");
    args3.push_back("Read arguments from file (arguments_dp_3d.txt)");

    args.push_back("-debug_mode");
    args2.push_back("0");
    args3.push_back("Special debug mode (0:normal,1:special peak,2: normal peak, 3:simple)");

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
    args3.push_back("Name of last input file (test512.ft3)");

    args.push_back("-out");
    args2.push_back("peaks.list");
    args3.push_back("Output file name in Sparky format. (peaks.list)");

    args.push_back("-peak_in");
    args2.push_back("input_peaks.list");
    args3.push_back("Intermediate peak list file, used in debug mode only");

    args.push_back("-model");
    args2.push_back("2");
    args3.push_back("Model selection for ANN picker, 1: FWHH 6-20, 2: FWHH 4-12 (1)");

    args.push_back("-negative");
    args2.push_back("yes");
    args3.push_back("Pick negative peaks too (no)");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname1,infname2,outfname,peakinfname;
    double user,user2;
    int model_selection;
    int debug_mode;
    bool b_negative;

    b_negative=false;
    b_negative=cmdline.query("-negative")=="yes" || cmdline.query("-negative")=="y";

    debug_mode=atoi(cmdline.query("-debug_mode").c_str());


    model_selection=atoi(cmdline.query("-model").c_str());
    if(model_selection!=1) model_selection=2;

    double noise_level=atof(cmdline.query("-noise_level").c_str());
    infname1 = cmdline.query("-in1");
    infname2 = cmdline.query("-in2");
    
    peakinfname=cmdline.query("-peak_in");
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());

    // FILE *fp=fopen("for_debug_peaks.txt","w");
    // if(fp==NULL)
    // {
    //     std::cout<<"can't open for_debug_peaks.txt to write."<<std::endl;
    //     exit(0);
    // }
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_pick_3d x;
        x.init_parameters(user,user2,noise_level,model_selection,0,b_negative); //zf=0
        x.read_for_picking(infname1,infname2);

        if(debug_mode==1) //debug mode 1, load normal picking peaks then check special peaks
        {
            x.peak_reading(peakinfname);
            x.special_case_peaks();
            x.print_peaks(outfname);
        }
        else if(debug_mode==2) //debug mode 2, normal picking only
        {
            x.peak_picking();
            x.print_peaks(outfname);
        }
        else if(debug_mode==3) //simple mathmatical max peaks picking, as baseline model
        {
            x.simple_peak_picking();
            x.print_peaks(outfname);
        }
        else //normal picking followed by special case
        {
            x.peak_picking();
            x.special_case_peaks();
            x.print_peaks(outfname);
        }
    }
    return 0;
}
