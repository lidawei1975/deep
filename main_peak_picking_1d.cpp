
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>


#include "json/json.h"
#include "commandline.h"
#include "spectrum_pick_1d.h"




#include "DeepConfig.h"

int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-f");
    args2.push_back("arguments_peak_picking_1d.txt");
    args3.push_back("Arguments file");

    args.push_back("-v");
    args2.push_back("1");
    args3.push_back("verbose level (0 or 1)");

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
    
    args.push_back("-out");
    args2.push_back("peaks.tab");
    args3.push_back("output file name");

    args.push_back("-model");
    args2.push_back("2");
    args3.push_back("Model selection for ANN picker, 1: FWHH 6-20, 2: FWHH 4-12");

    args.push_back("-auto_ppp");
    args2.push_back("yes");
    args3.push_back("Adjust peak width automatically using cubic spline interpolation. Suppressed if user_ppp is not 1.0");

    args.push_back("-interp_step");
    args2.push_back("1.0");
    args3.push_back("User defined spectrum interpolation step. 1 means no interpolation. 0.5 means 2x interpolation.");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("Also pick negative peaks (no)");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname,outfname;
    double user,user2;
    double max_width;
    int model_selection;
    bool b_negative=cmdline.query("-negative")=="yes" || cmdline.query("-negative")=="Yes" || cmdline.query("-negative")=="YES" || cmdline.query("-negative")=="y" || cmdline.query("-negative")=="Y";
    bool b_auto_ppp=cmdline.query("-auto_ppp")=="yes" || cmdline.query("-auto_ppp")=="Yes" || cmdline.query("-auto_ppp")=="YES" || cmdline.query("-auto_ppp")=="y" || cmdline.query("-auto_ppp")=="Y";

    shared_data_1d::n_verbose=atoi(cmdline.query("-v").c_str());

    double interp_step=std::stod(cmdline.query("-interp_step"));

    /**
     * A value of 1.0 means no interpolation. 0.5 means 2x interpolation. 2.0 means 0.5x interpolation.
     * A value other than 1.0 will suppress auto peak width adjustment.
    */
    if(interp_step>1.01 || interp_step<0.99) 
    {
        std::cout<<"User defined spectrum interpolation step: "<<interp_step<<std::endl;
        if(b_auto_ppp)
        {
            b_auto_ppp=false;
            std::cout<<"Auto peak width adjustment is suppressed."<<std::endl;
        }
    }


    model_selection=atoi(cmdline.query("-model").c_str());
    if(model_selection!=1 && model_selection!=2) model_selection=3;

    double noise_level=atof(cmdline.query("-noise_level").c_str());
    infname = cmdline.query("-in");
    
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());
    
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_pick_1d x;
        x.init(user,user2,noise_level);
        x.init_mod(model_selection);
        if(x.read_spectrum(infname,b_negative)) //read
        {
            if(interp_step>1.01 || interp_step<0.99)
            {
                x.interpolate_spectrum(interp_step);
            }
            /**
             * Auto peak width adjustment
             */
            else if(b_auto_ppp)
            {
                double target_width=8.0;
                if(model_selection==1) target_width=12.0; //model 1, optimal width is 12
                else target_width=6.0; //model 2, optimal width is 8
                /**
                 * We may need multiple iterations to get the optimal width because fwhh estimation is not perfect, especially when far away from the optimal width
                */
                x.adjust_ppp_of_spectrum(target_width);
                x.adjust_ppp_of_spectrum(target_width);
            }
            x.spectrum_pick_1d_work(b_negative); //picking 
            x.print_peaks(outfname); //output
        }
        std::cout<<"Done!"<<std::endl;
    }
    return 0;
};