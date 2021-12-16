
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_pick.h"


int main(int argc, char **argv)
{
    std::cout<<"Last update: Dec. 2021"<<std::endl;
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("Print help message then quit");

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("User defined minimal peak intensity scale for peak picking");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("User defined noise floor scale factor for peak picking");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value, estimate from sepctrum if input is 0.0");

    args.push_back("-in");
    args2.push_back("test.ft2");
    args3.push_back("Input file name");

    args.push_back("-out");
    args2.push_back("peaks.tab");
    args3.push_back("Output file names");

    args.push_back("-model");
    args2.push_back("1");
    args3.push_back("Model selection for ANN picker, 1: PPP 6-20, 2: PPP 4-12, 3: 3-9");

    args.push_back("-t1_noise");
    args2.push_back("no");
    args3.push_back("Remove possible t1 noise peaks");

    args.push_back("-debug_flag1");
    args2.push_back("0");
    args3.push_back("Reserved debug flag.(0:normal, 1:no special peak, 2: test new method)");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("Pick negative peaks instead of positive");
    

    cmdline.init(args, args2, args3);
    if(!cmdline.pharse(argc, argv)) return 1;

    std::string infname,outfname;
    double user,user2;
    double max_width;
    int model_selection;
    int debug_flag1;
    int t1_flag=0;
    bool b_negative;

    if(cmdline.query("-t1_noise")=="yes" || cmdline.query("-t1_noise")=="y") 
    {
        t1_flag=1;
    }

    b_negative=false;
    b_negative=cmdline.query("-negative")=="yes" || cmdline.query("-negative")=="y";

    debug_flag1=atoi(cmdline.query("-debug_flag1").c_str());
    model_selection=atoi(cmdline.query("-model").c_str());
    if(model_selection!=1) model_selection=2;

    double noise_level=atof(cmdline.query("-noise_level").c_str());
    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());

   
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_pick x;
        x.set_scale(user,user2);
        x.set_model_selection(model_selection);

        if(b_negative==true) 
        {
            x.set_negative_mode();
        }

        if(x.init(infname)) //read and zero filling
        {
            if (noise_level>1e-20) { x.set_noise_level(noise_level);}
            x.ann_peak_picking(debug_flag1,0,t1_flag); //picking
            x.print_peaks_picking(outfname);
        }
        // x.clear_memory();
        // std::ofstream fnoise("noise_level.txt");
        // fnoise<<x.get_noise_level()<<std::endl;
        // fnoise.close();
        
    }
    return 0;
}
