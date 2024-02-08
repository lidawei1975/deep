
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_pick.h"

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
    args2.push_back("arguments_dp.txt");
    args3.push_back("Parguments file");

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

    args.push_back("-auto_ppp");
    args2.push_back("yes");
    args3.push_back("Adjust peak width automatically (yes) using cubic spline interpolation");

    args.push_back("-model");
    args2.push_back("1");
    args3.push_back("Model selection for ANN picker, 1: PPP 6-20, 2: PPP 4-12");

    args.push_back("-t1_noise");
    args2.push_back("no");
    args3.push_back("Remove possible t1 noise peaks");

    args.push_back("-debug_flag1");
    args2.push_back("0");
    args3.push_back("Reserved debug flag.(0:normal, 1:no special peak, 2: test new method)");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("Pick both negative and positive peaks.");
    

    cmdline.init(args, args2, args3);
    if(!cmdline.pharse(argc, argv)) return 1;

    std::string infname,outfname;
    double user,user2;
    double max_width;
    int model_selection;
    int debug_flag1;
    int t1_flag=0;


    if(cmdline.query("-t1_noise")=="yes" || cmdline.query("-t1_noise")=="y") 
    {
        t1_flag=1;
    }



    debug_flag1=atoi(cmdline.query("-debug_flag1").c_str());
    model_selection=atoi(cmdline.query("-model").c_str());
    if(model_selection!=1) model_selection=2;

    /**
     *if  noise level > 0.0, pass to spectrum_pick (0: auto est, >0.0: user defined)
     * set it to 0.0 if user provide a negative value
    */
    double noise_level=atof(cmdline.query("-noise_level").c_str());
    if(noise_level<0.0){
        std::cout<<"Error: noise level must be >= 0.0, set it to 0.0"<<std::endl;
        noise_level=0.0;
    }


    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");

    /**
     * peak height cutoff, user defined. must be > 0.0
    */
    user=atof(cmdline.query("-scale").c_str());
    if(user<=0.0)
    {
        std::cout<<"Error: peak height cutoff must be > 0.0, set it to 5.5"<<std::endl;
        user=5.5;
    }

    /**
     * noise floor cutoff, user defined. must be > 0.0
    */
    user2=atof(cmdline.query("-scale2").c_str());
    if(user2<=0.0)
    {
        std::cout<<"Error: noise floor cutoff must be > 0.0, set it to 3.0"<<std::endl;
        user2=3.0;
    }

    bool b_auto_ppp=cmdline.query("-auto_ppp")=="yes" || cmdline.query("-auto_ppp")=="Yes" || cmdline.query("-auto_ppp")=="YES" || cmdline.query("-auto_ppp")=="y" || cmdline.query("-auto_ppp")=="Y";
    bool b_negative=cmdline.query("-negative")=="yes" || cmdline.query("-negative")=="Yes" || cmdline.query("-negative")=="YES" || cmdline.query("-negative")=="y" || cmdline.query("-negative")=="Y";
    

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_pick x;
        x.set_scale(user,user2);
        x.set_model_selection(model_selection);
        if(x.init(infname)) //read and zero filling
        {
            if (noise_level>1e-20) { x.set_noise_level(noise_level);}

             /**
             * Auto peak width adjustment
             */
            if(b_auto_ppp)
            {
                double target_width=8.0;
                if(model_selection==1) target_width=12.0; //model 1, optimal width is 12
                else target_width=6.0; //model 2, optimal width is 8
                x.adjust_ppp_of_spectrum(target_width);
            }

            x.ann_peak_picking(debug_flag1,t1_flag,b_negative); //picking
            x.print_peaks_picking(outfname);
        }
    }
    return 0;
}
